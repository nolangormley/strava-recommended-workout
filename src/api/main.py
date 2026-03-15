from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import os
import random
import requests
from dotenv import load_dotenv
from typing import Optional

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.metrics import calculate_ctl_atl, calculate_tsb, get_target_category, calculate_vo2max_from_df, clean_val, calculate_acwr

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    merged = pd.merge(daily_data, df, on='activity_date', how='left')
    merged['daily_load'] = merged['daily_load'].fillna(0)
    
    # Treat 0s as NaNs for EF and decoup so they don't skew the average or display as 0
    merged['daily_ef'] = merged['daily_ef'].replace(0, np.nan)
    merged['daily_decoup'] = merged['daily_decoup'].replace(0, np.nan)
    
    # 3. Calculate CTL, ATL, TSB
    loads = merged['daily_load'].values
    ctl, atl = calculate_ctl_atl(loads)
        
    merged['CTL'] = ctl
    merged['ATL'] = atl
    merged['TSB'] = calculate_tsb(merged['CTL'], merged['ATL'])
    merged['ACWR'] = calculate_acwr(merged['CTL'], merged['ATL'])
    
    # Calculate Rolling Averages for Insights (7-day)
    merged['EF_7d'] = merged['daily_ef'].rolling(window=7, min_periods=1).mean()
    merged['Decoup_7d'] = merged['daily_decoup'].rolling(window=7, min_periods=1).mean()
    
    # Get last known actual values
    merged['latest_ef'] = merged['daily_ef'].ffill()
    merged['latest_decoup'] = merged['daily_decoup'].ffill()
    
    today_stats = merged.iloc[-1]
    tsb = today_stats['TSB']

    # Map TSB to Category
    target_category = get_target_category(tsb)
    
    # Get VO2 Max estimate
    vo2max_data = calculate_vo2max(user_id)
    latest_vo2_max = vo2max_data.get('latest_vo2_max') if vo2max_data else None
    
    # Get Athlete Info
    try:
        athlete_data = con.execute("SELECT weight, sex FROM dim_athlete WHERE athlete_id = ?", [user_id]).fetchone()
        weight_kg = athlete_data[0] if athlete_data else None
        weight_lbs = round(weight_kg * 2.20462) if weight_kg else None
        sex = athlete_data[1] if athlete_data else None
    except:
        weight_lbs = None
        sex = None

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
        "acwr":        clean_val(today_stats['ACWR'], 2),
        "target_category": target_category,
        "efficiency_factor_7d": clean_val(today_stats.get('EF_7d'), 2),
        "aerobic_decoupling_7d": clean_val(today_stats.get('Decoup_7d'), 1),
        "latest_daily_ef": clean_val(today_stats.get('latest_ef'), 2),
        "latest_daily_decoup": clean_val(today_stats.get('latest_decoup'), 1),
        "latest_vo2_max": latest_vo2_max,
        "weight_lbs": weight_lbs,
        "sex": sex,
        "history": history_list
    }

def get_ai_insight(stats, context="status", workout=None):
    try:
        history_str = "\n".join([f"  - {h['date']}: Form (TSB): {h['form']}, Load (ATL): {h['fatigue']}" for h in stats.get('history', [])])

        # Construct Prompt
        if context == "status":
            prompt = (
                f"Explain the user's current training status based on these metrics:\n"
                f"- Fitness (CTL): {stats.get('fitness_ctl')}\n"
                f"- Fatigue (ATL): {stats.get('fatigue_atl')}\n"
                f"- Form (TSB): {stats.get('form_tsb')} (Target: {stats.get('target_category')})\n"
                f"- Acute-to-Chronic Workload Ratio (ACWR): {stats.get('acwr')}\n"
                f"- Estimated VO2 Max: {stats.get('latest_vo2_max', 'N/A')}\n"
                f"- 7-Day Efficiency Factor: {stats.get('efficiency_factor_7d')}\n"
                f"- 7-Day Aerobic Decoupling: {stats.get('aerobic_decoupling_7d')}%\n\n"
                f"Give a short assessment of their readiness. Mention their ACWR: the sweet spot for injury prevention is 0.8-1.3, while over 1.5 is the danger zone. Give actionable advice."
            )
        elif context == "workout":
            prompt = (
                f"The athlete currently has a Form/TSB of {stats.get('form_tsb')} (Negative TSB means fatigued/in heavy training, positive means rested/tapering).\n"
                f"Their 7-day Aerobic Decoupling is {stats.get('aerobic_decoupling_7d')}% (Under 5% indicates good base aerobic fitness).\n"
                f"Their Estimated VO2 Max is: {stats.get('latest_vo2_max', 'N/A')}\n\n"
                f"We are recommending this workout: {workout.get('name')} ({workout.get('category')}).\n"
                f"Description: {workout.get('description')}\n\n"
                f"Briefly explain in 2-3 sentences why this specific workout is appropriate for their current training state, acting as their professional endurance coach. Talk directly to them."
            )
            
        llm_provider = os.getenv("LLM_PROVIDER", "local").lower()
        
        weight_str = f"weighs {stats.get('weight_lbs')}lbs" if stats.get('weight_lbs') else "weighs 220lbs"
        sex_str = stats.get('sex', 'male')
        if sex_str == 'M': sex_str = 'male'
        elif sex_str == 'F': sex_str = 'female'
        
        system_prompt = f"You are an expert running coach and one of your athletes is training for a half marathon that is on May 2nd 2026. The athlete is 28, is 5'11, {sex_str}, and {weight_str}. Their goal is to run a sub 2 hour half marathon. Explain what the metrics mean in a concise way and give actionable advice. Allow a bit of overtraining, they are very committed and love to push themselves. Don't be too conservative with the advice."

        if llm_provider == "groq":
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                return "Groq API key not set in environment (GROQ_API_KEY)."
                
            url = "https://api.groq.com/openai/v1/chat/completions"
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                return str(data)
            except requests.exceptions.HTTPError as e:
                print(f"HTTPError on Groq API call: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Groq error response body: {e.response.text}")
                raise
        else:
            # Call LM Studio local API
            url = "http://localhost:1234/api/v1/chat"
                
            payload = {
                "model": "mistralai/ministral-3-3b",
                "system_prompt": system_prompt,
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
    
    target_category = get_target_category(tsb)
        
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
        "current_acwr": status.get("acwr"),
        "recommended_category": target_category,
        "latest_vo2_max": status.get("latest_vo2_max"),
        "ai_reasoning": ai_reasoning,
        "workout": w_dict
    }

def calculate_vo2max(user_id: int):
    con = get_db_connection()
    try:
        max_hr_query = con.execute("SELECT MAX(max_heartrate) FROM dim_activity WHERE athlete_id=?", [user_id]).fetchone()
        hr_max = max_hr_query[0] if max_hr_query and max_hr_query[0] and max_hr_query[0] > 100 else 190
    except:
        hr_max = 190

    hr_rest = 60
    
    try:
        query = """
        SELECT 
            a.activity_id,
            a.start_date_local::DATE as activity_date,
            v.value as speed,
            h.value as hr
        FROM dim_activity a
        JOIN stream_velocity v ON a.activity_id = v.activity_id
        JOIN stream_heartrate h ON a.activity_id = h.activity_id AND v.time_offset = h.time_offset
        WHERE a.athlete_id = ? AND a.type = 'Run'
          AND v.value > 1.5
          AND h.value > 100
        """
        df = con.execute(query, [user_id]).fetchdf()
    except Exception as e:
        print(f"Error querying streams: {e}")
        return None
        
    if df.empty:
        return None
        
    df = calculate_vo2max_from_df(df, hr_max, hr_rest)
    if df is None or df.empty:
        return None
    
    # Calculate median per activity date
    results = df.groupby('activity_date')['vo2_max_est'].median().reset_index()
    results = results.sort_values('activity_date')
    
    # Roll 7 day average of the estimations
    results['vo2_max_rolling_7d'] = results['vo2_max_est'].rolling(window=7, min_periods=1).mean()

    latest = results.iloc[-1]
    
    history_list = []
    # Return last 14 days
    for _, row in results.tail(14).iterrows():
        history_list.append({
            "date": str(row['activity_date']),
            "vo2_max_estimate": clean_val(row['vo2_max_est'], 2),
            "vo2_max_rolling_7d": clean_val(row['vo2_max_rolling_7d'], 2)
        })

    return {
        "user_id": user_id,
        "latest_vo2_max": clean_val(latest['vo2_max_rolling_7d'], 2),
        "history": history_list
    }

@app.get("/vo2max/{user_id}")
def get_vo2max(user_id: int):
    vo2max_data = calculate_vo2max(user_id)
    if not vo2max_data:
        raise HTTPException(status_code=404, detail="No stream data found to calculate VO2 Max for this user.")
    return vo2max_data

def get_ai_training_plan(stats, race_date: str, race_type: str, goal_time: Optional[str] = None, pace_zones: Optional[dict] = None):
    try:
        current_date_str = stats.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        prompt = (
            f"The athlete is racing a {race_type} on {race_date}.\n"
        )
        if goal_time:
            prompt += f"Their goal time is {goal_time}.\n"
        else:
            prompt += "They have not provided a goal time. Please recommend a challenging but realistic goal time based on their current fitness level (especially VO2 max and Threshold pace) and include it in your response.\n"
            
        prompt += (
            f"Today's date is {current_date_str}.\n"
            f"Please create a daily training plan from tomorrow until the race date.\n\n"
            f"Current fitness stats to consider:\n"
            f"- Fitness (CTL): {stats.get('fitness_ctl', 'N/A')}\n"
            f"- Fatigue (ATL): {stats.get('fatigue_atl', 'N/A')}\n"
            f"- Form (TSB): {stats.get('form_tsb', 'N/A')}\n"
            f"- Estimated VO2 Max: {stats.get('latest_vo2_max', 'N/A')}\n"
            f"- 7-Day Efficiency Factor: {stats.get('efficiency_factor_7d', 'N/A')}\n"
            f"- 7-Day Aerobic Decoupling: {stats.get('aerobic_decoupling_7d', 'N/A')}%\n\n"
            f"Based on these metrics, tailor the difficulty and volume of the plan. "
            f"If fatigue is high, perhaps start with recovery. If fitness is high, challenge them appropriately.\n"
        )
        
        if pace_zones and "pace_zones_min_per_mile" in pace_zones:
            prompt += "\nUse the following exact pace zones based on their recent data when recommending how fast they should run:\n"
            for z_name, z_range in pace_zones["pace_zones_min_per_mile"].items():
                prompt += f"- {z_name}: {z_range}\n"
            prompt += "\nMake sure to explicitly mention these paces in the workout 'description' so the user knows what speed to run at. Just use the mean of the pace zone range when providing the pace in the description.\n\n"
        
        llm_provider = os.getenv("LLM_PROVIDER", "local").lower()
        
        weight_str = f"weighs {stats.get('weight_lbs')}lbs" if stats.get('weight_lbs') else "weighs 220lbs"
        sex_str = stats.get('sex', 'male')
        if sex_str == 'M': sex_str = 'male'
        elif sex_str == 'F': sex_str = 'female'
        
        system_prompt = (
            f"You are an expert running coach. Your athlete is {sex_str} and {weight_str}. "
            "You must respond STRICTLY with a valid JSON object. Do not include markdown formatting like ```json or any text outside the JSON framework. "
            "The JSON object must have three keys:\n"
            "1. 'blurb': a short introductory and motivating message about the plan.\n"
            "2. 'recommended_goal_time': a string recommending a goal time based on their metrics. If they provided a goal time, just restate it.\n"
            "3. 'plan': a dictionary where keys are the dates in 'YYYY-MM-DD' format, and values are objects with keys: "
            "'type_of_workout' (e.g., Easy Run, Tempo, Long Run, Rest), 'training_focus' (the purpose of the workout), "
            "'approximate_distance' (e.g., '5 miles'), and 'description' (detailed instructions)."
        )

        if llm_provider == "groq":
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                return {"error": "Groq API key not set in environment (GROQ_API_KEY)."}
                
            url = "https://api.groq.com/openai/v1/chat/completions"
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            }
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    import json
                    content = data["choices"][0]["message"]["content"]
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return {"error": "Failed to parse JSON response from LLM", "raw_content": content}
                return {"error": "Invalid response format from Groq API."}
            except requests.exceptions.HTTPError as e:
                print(f"HTTPError on Groq API call: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Groq error response body: {e.response.text}")
                return {"error": str(e)}
        else:
            return {"error": "This feature currently requires the LLM_PROVIDER to be set to groq in order to return properly formatted JSON."}

    except Exception as e:
        print(f"Error getting AI training plan: {e}")
        return {"error": str(e)}

@app.get("/schedule/{user_id}")
def get_race_schedule(user_id: int, race_date: str, race_type: str, goal_time: Optional[str] = None):
    # Validations
    try:
        datetime.strptime(race_date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid race_date format. Must be YYYY-MM-DD.")
        
    valid_race_types = ["5k", "5 mile", "10k", "10 mile", "half marathon", "marathon"]
    if race_type.lower() not in valid_race_types:
        raise HTTPException(status_code=400, detail=f"Invalid race_type. Must be one of: {', '.join(valid_race_types)}")
        
    status = calculate_training_status_logic(user_id)
    if not status:
        # Fallback to an empty stats dict if they have no training history
        status = {"user_id": user_id}
        
    pace_zones = calculate_pace_zones(user_id)
        
    plan = get_ai_training_plan(status, race_date, race_type, goal_time, pace_zones)
    
    if "error" in plan:
        raise HTTPException(status_code=500, detail=plan["error"])
        
    return {
        "user_id": user_id,
        "race_date": race_date,
        "race_type": race_type,
        "goal_time": goal_time,
        "pace_zones": pace_zones,
        "schedule": plan
    }

def speed_to_pace_str(speed_ms):
    if speed_ms <= 0 or pd.isna(speed_ms):
        return "N/A"
    mins = 26.8224 / speed_ms
    m = int(mins)
    s = int((mins - m) * 60)
    return f"{m}:{s:02d} /mi"

def calculate_pace_zones(user_id: int):
    con = get_db_connection()
    try:
        max_hr_query = con.execute("SELECT MAX(max_heartrate) FROM dim_activity WHERE athlete_id=?", [user_id]).fetchone()
        hr_max = max_hr_query[0] if max_hr_query and max_hr_query[0] and max_hr_query[0] > 100 else 190
    except:
        hr_max = 190

    # Limit to recent activities for more accurate current zones (e.g. last 90 days)
    try:
        query = """
        SELECT 
            v.value as speed,
            h.value as hr
        FROM dim_activity a
        JOIN stream_velocity v ON a.activity_id = v.activity_id
        JOIN stream_heartrate h ON a.activity_id = h.activity_id AND v.time_offset = h.time_offset
        WHERE a.athlete_id = ? AND a.type = 'Run'
          AND v.value > 1.8 AND v.value < 8.0
          AND h.value > 80 AND h.value <= ?
          AND a.start_date_local >= current_date - interval '90 days'
        """
        df = con.execute(query, [user_id, hr_max + 10]).fetchdf()
    except Exception as e:
        print(f"Error querying streams for pace zones: {e}")
        return None
        
    if df.empty:
        return None

    # Calculate Threshold Speed (T_speed)
    # Target HR for threshold is typically 85%-92% of HR Max.
    # To avoid hills throwing off the pace, we take the 80th percentile of speed in this HR range.
    threshold_df = df[(df['hr'] >= hr_max * 0.85) & (df['hr'] <= hr_max * 0.92) & (df['speed'] > 2.0)]
    
    if len(threshold_df) >= 30:
        t_speed = float(threshold_df['speed'].quantile(0.80))
    else:
        # Fallback to general high end speed if no threshold data
        t_speed = float(df['speed'].quantile(0.90))

    # Convert t_speed to pace in seconds per mile
    t_pace_sec = 26.8224 / t_speed * 60  # convert m/s to seconds per mile (1609.34 / t_speed)
    
    # Coros Percentages of Threshold Pace:
    # Recovery: > 140%
    # Aerobic Endurance: 119% - 140%
    # Aerobic Power: 106% - 119%
    # Threshold: 94.5% - 106%
    # Anaerobic Endurance: 85% - 94.5%
    # Anaerobic Power: < 85%
    
    def format_pace(sec):
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}'{s:02d}\""

    p140 = t_pace_sec * 1.40
    p119 = t_pace_sec * 1.19
    p106 = t_pace_sec * 1.06
    p0945 = t_pace_sec * 0.945
    p085 = t_pace_sec * 0.85
    
    # We subtract 1 second from the upper bound to prevent overlap, just like Coros
    result = {
        "Recovery": f">{format_pace(p140)}",
        "Aerobic Endurance": f"{format_pace(p119)}-{format_pace(p140)}",
        "Aerobic Power": f"{format_pace(p106)}-{format_pace(p119 - 1)}",
        "Threshold": f"{format_pace(p0945)}-{format_pace(p106 - 1)}",
        "Anaerobic Endurance": f"{format_pace(p085)}-{format_pace(p0945 - 1)}",
        "Anaerobic Power": f"<{format_pace(p085)}"
    }

    return {
        "user_id": user_id,
        "hr_max_used": hr_max,
        "pace_zones_min_per_mile": result
    }

@app.get("/pace_zones/{user_id}")
def get_pace_zones(user_id: int):
    zones = calculate_pace_zones(user_id)
    if not zones:
        raise HTTPException(status_code=404, detail="No stream data found to calculate Pace Zones for this user.")
    return zones

