from fastapi import FastAPI, HTTPException
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import os
import random
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Database Path
# Docker: /app/data/strava_warehouse.duckdb
# Local: ../data/strava_warehouse.duckdb
# We assume this script is running from api/ directory or similar depth
# os.path.dirname(__file__) -> .../api
# os.path.dirname(...) -> .../root
DB_PATH = os.getenv('DB_PATH', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'strava_warehouse.duckdb'))

def get_db_connection():
    try:
        # read_only=True is safer for API queries
        return duckdb.connect(DB_PATH, read_only=True)
    except Exception as e:
        # If DB doesn't exist yet
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

@app.get("/users/")
def get_users():
    try:
        con = get_db_connection()
        # Check if table exists
        try:
            users = con.execute("SELECT * FROM dim_athlete").fetchall()
        except:
             return []
             
        columns = [desc[0] for desc in con.description]
        result = []
        for user in users:
            result.append(dict(zip(columns, user)))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calculate_training_status_logic(user_id: int):
    con = get_db_connection()
    
    # 1. Get Daily TRIMP Sums for this user
    try:
        df = con.execute("""
            SELECT 
                da.start_date_local::DATE as activity_date,
                SUM(ae.trimp_banister) as daily_load,
                AVG(ae.efficiency_factor) as daily_ef,
                AVG(ae.aerobic_decoupling) as daily_decoup
            FROM activity_effectiveness ae
            JOIN dim_activity da ON ae.activity_id = da.activity_id
            WHERE da.athlete_id = ?
            GROUP BY 1
            ORDER BY 1
        """, [user_id]).fetchdf()
    except Exception as e:
        # Table might not exist
        print(f"Error querying effectiveness: {e}")
        return None
    
    if df.empty:
        return None

    # 2. Reindex
    start_date = df['activity_date'].min()
    end_date = pd.Timestamp(datetime.now().date())
    
    # Handle if start date is future (unlikely)
    if start_date > end_date:
        start_date = end_date

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_data = pd.DataFrame({'date': date_range.date})
    daily_data['activity_date'] = pd.to_datetime(daily_data['date'])
    df['activity_date'] = pd.to_datetime(df['activity_date'])
    
    merged = pd.merge(daily_data, df, on='activity_date', how='left').fillna(0)
    
    # 3. Calculate CTL, ATL, TSB
    CTL_DECAY = 42
    ATL_DECAY = 7
    
    ctl = [0.0]
    atl = [0.0]
    loads = merged['daily_load'].values
    
    for i in range(len(loads)):
        daily_strain = loads[i]
        # Exponential Weighted Moving Average
        # CTL_only = CTL_prev + (Load - CTL_prev) / time_constant
        curr_ctl = ctl[-1] + (daily_strain - ctl[-1]) / CTL_DECAY
        curr_atl = atl[-1] + (daily_strain - atl[-1]) / ATL_DECAY
        ctl.append(curr_ctl)
        atl.append(curr_atl)
        
    merged['CTL'] = ctl[1:]
    merged['ATL'] = atl[1:]
    merged['TSB'] = merged['CTL'] - merged['ATL']
    
    # Calculate Rolling Averages for Insights (7-day)
    merged['EF_7d'] = merged['daily_ef'].rolling(window=7, min_periods=1).mean()
    merged['Decoup_7d'] = merged['daily_decoup'].rolling(window=7, min_periods=1).mean()
    
    today_stats = merged.iloc[-1]
    tsb = today_stats['TSB']

    # Map TSB to Category
    if tsb > 5:
        target_category = "Anaerobic"
    elif -10 <= tsb <= 5:
        target_category = "VO2Max"
    elif -30 <= tsb < -10:
        target_category = "Aerobic"
    else:
        target_category = "Recovery"
    
    # Helper to clean NNs for JSON
    def clean_val(val, decimals=1):
        if pd.isna(val) or val is None: return None
        return round(float(val), decimals)

    # Get Last 7 days history
    history_df = merged.tail(7)
    history_list = []
    for _, row in history_df.iterrows():
        history_list.append({
            "date": str(row['date']),
            "fitness": clean_val(row['CTL']),
            "fatigue": clean_val(row['ATL']),
            "form": clean_val(row['TSB']),
            "daily_ef": clean_val(row['daily_ef'], 2),
            "daily_decoup": clean_val(row['daily_decoup'], 1)
        })

    return {
        "date": str(today_stats['date']),
        "fitness_ctl": clean_val(today_stats['CTL']),
        "fatigue_atl": clean_val(today_stats['ATL']),
        "form_tsb":    clean_val(today_stats['TSB']),
        "target_category": target_category,
        "efficiency_factor_7d": clean_val(today_stats.get('EF_7d'), 2),
        "aerobic_decoupling_7d": clean_val(today_stats.get('Decoup_7d'), 1),
        "latest_daily_ef": clean_val(today_stats.get('daily_ef'), 2),
        "latest_daily_decoup": clean_val(today_stats.get('daily_decoup'), 1),
        "history": history_list
    }

def get_ai_insight(stats, context="status", workout=None):
    try:
        history_str = "\n".join([f"  - {h['date']}: Form (TSB): {h['form']}, Load (ATL): {h['fatigue']}" for h in stats.get('history', [])])

        # Construct Prompt
        if context == "status":
            prompt = (
                f"Analyze the athlete's current training status and give a 2-3 sentence recommendation on what they should focus on.\n\n"
                f"Current Metrics:\n"
                f"- Fitness (CTL, 42-day moving average - higher means fitter): {stats.get('fitness_ctl')}\n"
                f"- Fatigue (ATL, 7-day avg load - higher means more tired): {stats.get('fatigue_atl')}\n"
                f"- Form (TSB, Fitness minus Fatigue - negative means fatigued, positive means rested): {stats.get('form_tsb')}\n"
                f"- Efficiency Factor 7d Avg (EF, higher is better aerobic efficiency): {stats.get('efficiency_factor_7d')}\n"
                f"- Aerobic Decoupling 7d Avg (Heart rate drift, lower is better, <5% is great): {stats.get('aerobic_decoupling_7d')}%\n\n"
                f"Recent 7 Day Trend:\n{history_str}\n\n"
                f"Talk directly to them as an expert endurance coach. Be concise and provide actionable advice based on their current load, form, and aerobic efficiency."
            )
        elif context == "workout":
            prompt = (
                f"The athlete currently has a Form/TSB of {stats.get('form_tsb')} (Negative TSB means fatigued/in heavy training, positive means rested/tapering).\n"
                f"Their 7-day Aerobic Decoupling is {stats.get('aerobic_decoupling_7d')}% (Under 5% indicates good base aerobic fitness).\n\n"
                f"We are recommending this workout: {workout.get('name')} ({workout.get('category')}).\n"
                f"Description: {workout.get('description')}\n\n"
                f"Briefly explain in 2-3 sentences why this specific workout is appropriate for their current training state, acting as their professional endurance coach. Talk directly to them."
            )
            
        # Call LM Studio local API
        url = "http://localhost:1234/api/v1/chat"
            
        payload = {
            "model": "mistralai/ministral-3-3b",
            "system_prompt": "You are an expert running coach and one of your athletes is training for a half marathon that is on May 2nd 2026. The athlete is 28, is 5'11 and weighs 220lbs. His goal is to run a sub 2 hour half marathon. Explain what the metrics mean in a concise way and give actionable advice.",
            "input": prompt
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Assuming the response format matches the custom local LM studio formats
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0 and "message" in data["choices"][0]:
            return data["choices"][0]["message"]["content"]
        elif "output" in data and len(data["output"]) > 0 and "content" in data["output"][0]:
            return data["output"][0]["content"]
        elif "message" in data:
            return data["message"]
        elif "response" in data:
            return data["response"]
        else:
            return str(data)
            
    except Exception as e:
        print(f"Error getting AI insight: {e}")
        return None
    except Exception as e:
        print(f"Error getting AI insight: {e}")
        return None

@app.get("/status/{user_id}")
def get_status(user_id: int):
    status = calculate_training_status_logic(user_id)
    if not status:
        raise HTTPException(status_code=404, detail="No training data found for user (Did you run analyze_effectiveness.py?)")
    
    # Add AI Insight
    status['ai_insight'] = get_ai_insight(status, context="status")
    
    return status

@app.get("/recommend/{user_id}")
def get_recommendation(user_id: int):
    status = calculate_training_status_logic(user_id)
    if not status:
        raise HTTPException(status_code=404, detail="No training data found for user")
    
    tsb_val = status.get('form_tsb')
    tsb = float(tsb_val) if tsb_val is not None else 0.0
    
    # Map TSB to Category
    if tsb > 5:
        target_category = "Anaerobic"
    elif -10 <= tsb <= 5:
        target_category = "VO2Max"
    elif -30 <= tsb < -10:
        target_category = "Aerobic"
    else:
        target_category = "Recovery"
        
    con = get_db_connection()
    try:
        workouts = con.execute("SELECT name, description, url, category FROM dim_workouts WHERE category = ?", [target_category]).fetchall()
    except Exception as e:
        # Table might not exist
        return {"error": "dim_workouts table not found. Run ingest_workouts.py first."}

    # Fallback logic
    if not workouts:
         fallback_map = {
             "Anaerobic": "VO2Max",
             "VO2Max": "Threshold", # Or Aerobic
             "Threshold": "Aerobic",
             "Aerobic": "Recovery",
             "Recovery": "Aerobic" 
         }
         # Try fallback
         orig_target = target_category
         target_category = fallback_map.get(target_category, "Aerobic")
         workouts = con.execute("SELECT name, description, url, category FROM dim_workouts WHERE category = ?", [target_category]).fetchall()
         
    if not workouts:
         # Final fallback to anything
         workouts = con.execute("SELECT name, description, url, category FROM dim_workouts LIMIT 5").fetchall()
         if not workouts:
            return {"message": "No workouts found in library at all."}

    w = random.choice(workouts)
    
    w_dict = {
            "name": w[0],
            "description": w[1],
            "url": w[2],
            "category": w[3]
        }
    
    # Add AI Reasoning
    ai_reasoning = get_ai_insight(status, context="workout", workout=w_dict)
    
    return {
        "user_id": user_id,
        "current_tsb": tsb,
        "recommended_category": target_category,
        "ai_reasoning": ai_reasoning,
        "workout": w_dict
    }
