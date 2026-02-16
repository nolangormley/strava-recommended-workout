from fastapi import FastAPI, HTTPException
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import os
import random

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
                SUM(ae.trimp_banister) as daily_load
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
    
    return {
        "date": str(today_stats['date']),
        "fitness_ctl": round(float(today_stats['CTL']), 1),
        "fatigue_atl": round(float(today_stats['ATL']), 1),
        "form_tsb": round(float(today_stats['TSB']), 1),
        "target_category": target_category
    }

@app.get("/status/{user_id}")
def get_status(user_id: int):
    status = calculate_training_status_logic(user_id)
    if not status:
        raise HTTPException(status_code=404, detail="No training data found for user (Did you run analyze_effectiveness.py?)")
    return status

@app.get("/recommend/{user_id}")
def get_recommendation(user_id: int):
    status = calculate_training_status_logic(user_id)
    if not status:
        raise HTTPException(status_code=404, detail="No training data found for user")
    
    tsb = status['form_tsb']
    
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
    
    return {
        "user_id": user_id,
        "current_tsb": tsb,
        "recommended_category": target_category,
        "workout": {
            "name": w[0],
            "description": w[1],
            "url": w[2],
            "category": w[3]
        }
    }
