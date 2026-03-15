from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import os
import random
import requests
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.metrics import calculate_ctl_atl, calculate_tsb, get_target_category, calculate_vo2max_from_df, clean_val

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Templates and Static Files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Fallback in case they don't exist yet:
os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

DB_PATH = os.getenv('DB_PATH', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'strava_warehouse.duckdb'))

def get_db_connection():
    try:
        # read_only=False allows API to perform CRUD operations.
        # Connections must be closed quickly to prevent holding the writer lock.
        return duckdb.connect(DB_PATH, read_only=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

# ==========================================
# FRONTEND ROUTES
# ==========================================

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/workout_streams/{activity_id}")
def view_workout_streams(request: Request, activity_id: int):
    return templates.TemplateResponse("streams.html", {"request": request, "activity_id": activity_id})

# ==========================================
# CRUD FOR ACTIVITIES (Read, Delete)
# ==========================================

@app.get("/api/activities")
def get_activities():
    con = get_db_connection()
    try:
        activities = con.execute("SELECT * FROM dim_activity ORDER BY start_date DESC").fetchall()
        columns = [desc[0] for desc in con.description]
        return [dict(zip(columns, row)) for row in activities]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()

@app.get("/api/activities/{activity_id}")
def get_activity(activity_id: int):
    con = get_db_connection()
    try:
        activity = con.execute("SELECT * FROM dim_activity WHERE activity_id = ?", [activity_id]).fetchone()
        if not activity:
            raise HTTPException(status_code=404, detail="Activity not found")
        columns = [desc[0] for desc in con.description]
        return dict(zip(columns, activity))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()

@app.get("/api/activities/{activity_id}/streams")
def get_activity_streams(activity_id: int):
    con = get_db_connection()
    try:
        # Join streams
        query = """
        SELECT 
            t.time_offset,
            v.value as velocity,
            h.value as heartrate,
            a.value as altitude,
            c.value as cadence,
            w.value as watts
        FROM (
            SELECT DISTINCT time_offset FROM stream_velocity WHERE activity_id = ?
            UNION SELECT DISTINCT time_offset FROM stream_heartrate WHERE activity_id = ?
        ) t
        LEFT JOIN stream_velocity v ON v.activity_id = ? AND v.time_offset = t.time_offset
        LEFT JOIN stream_heartrate h ON h.activity_id = ? AND h.time_offset = t.time_offset
        LEFT JOIN stream_altitude a ON a.activity_id = ? AND a.time_offset = t.time_offset
        LEFT JOIN stream_cadence c ON c.activity_id = ? AND c.time_offset = t.time_offset
        LEFT JOIN stream_watts w ON w.activity_id = ? AND w.time_offset = t.time_offset
        ORDER BY t.time_offset ASC
        """
        params = [activity_id] * 7
        df = con.execute(query, params).fetchdf()
        
        # Replace NaNs with None so JSON serialization works
        df = df.replace({np.nan: None})
        
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()

@app.delete("/api/activities/{activity_id}")
def delete_activity(activity_id: int):
    con = get_db_connection()
    try:
        # Delete from all tables sequentially.
        tables = [
            "stream_altitude", "stream_cadence", "stream_heartrate", "stream_moving",
            "stream_temperature", "stream_velocity", "stream_watts", "activity_effectiveness",
            "dim_activity"
        ]
        for table in tables:
            try:
                con.execute(f"DELETE FROM {table} WHERE activity_id = ?", [activity_id])
            except Exception as e:
                print(f"Failed deleting from {table}: {e}")
        return {"message": "Activity deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()

# ==========================================
# CRUD FOR WORKOUTS (Create, Read, Update, Delete)
# ==========================================
class WorkoutCreate(BaseModel):
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = "Manual"

@app.get("/api/workouts")
def get_workouts():
    con = get_db_connection()
    try:
        workouts = con.execute("SELECT * FROM dim_workouts ORDER BY workout_id DESC").fetchall()
        columns = [desc[0] for desc in con.description]
        return [dict(zip(columns, row)) for row in workouts]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()

@app.post("/api/workouts")
def create_workout(workout: WorkoutCreate):
    con = get_db_connection()
    try:
        # Auto-incrementing workout_id via MAX + 1
        max_id_row = con.execute("SELECT MAX(workout_id) FROM dim_workouts").fetchone()
        next_id = (max_id_row[0] or 0) + 1 if max_id_row and max_id_row[0] is not None else 1
        
        con.execute(
            "INSERT INTO dim_workouts (workout_id, name, description, category, tags, url, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [next_id, workout.name, workout.description, workout.category, workout.tags, workout.url, workout.source]
        )
        return {"message": "Workout created successfully", "workout_id": next_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()

@app.put("/api/workouts/{workout_id}")
def update_workout(workout_id: int, workout: WorkoutCreate):
    con = get_db_connection()
    try:
        con.execute(
            "UPDATE dim_workouts SET name=?, description=?, category=?, tags=?, url=?, source=? WHERE workout_id=?",
            [workout.name, workout.description, workout.category, workout.tags, workout.url, workout.source, workout_id]
        )
        return {"message": "Workout updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()

@app.delete("/api/workouts/{workout_id}")
def delete_workout(workout_id: int):
    con = get_db_connection()
    try:
        con.execute("DELETE FROM dim_workouts WHERE workout_id=?", [workout_id])
        return {"message": "Workout deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        con.close()


# ==========================================
# ORIGINAL ENDPOINTS & LOGIC (Refactored DB close)
# ==========================================

@app.get("/users/")
def get_users():
    con = get_db_connection()
    try:
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
    finally:
        con.close()

def calculate_training_status_logic(user_id: int):
    con = get_db_connection()
    try:
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
            print(f"Error querying effectiveness: {e}")
            return None
        
        if df.empty:
            return None

        # 2. Reindex
        start_date = df['activity_date'].min()
        end_date = pd.Timestamp(datetime.now().date())
        
        if start_date > end_date:
            start_date = end_date

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        daily_data = pd.DataFrame({'date': date_range.date})
        daily_data['activity_date'] = pd.to_datetime(daily_data['date'])
        df['activity_date'] = pd.to_datetime(df['activity_date'])
        
        merged = pd.merge(daily_data, df, on='activity_date', how='left')
        merged['daily_load'] = merged['daily_load'].fillna(0)
        
        # Treat 0s as NaNs
        merged['daily_ef'] = merged['daily_ef'].replace(0, np.nan)
        merged['daily_decoup'] = merged['daily_decoup'].replace(0, np.nan)
        
        loads = merged['daily_load'].values
        ctl, atl = calculate_ctl_atl(loads)
            
        merged['CTL'] = ctl
        merged['ATL'] = atl
        merged['TSB'] = calculate_tsb(merged['CTL'], merged['ATL'])
        
        merged['EF_7d'] = merged['daily_ef'].rolling(window=7, min_periods=1).mean()
        merged['Decoup_7d'] = merged['daily_decoup'].rolling(window=7, min_periods=1).mean()
        
        merged['latest_ef'] = merged['daily_ef'].ffill()
        merged['latest_decoup'] = merged['daily_decoup'].ffill()
        
        today_stats = merged.iloc[-1]
        tsb = today_stats['TSB']

        target_category = get_target_category(tsb)
        
        vo2max_data = calculate_vo2max(user_id)
        latest_vo2_max = vo2max_data.get('latest_vo2_max') if vo2max_data else None

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
            "latest_daily_ef": clean_val(today_stats.get('latest_ef'), 2),
            "latest_daily_decoup": clean_val(today_stats.get('latest_decoup'), 1),
            "latest_vo2_max": latest_vo2_max,
            "history": history_list
        }
    finally:
        con.close()

def get_ai_insight(stats, context="status", workout=None):
    try:
        history_str = "\n".join([f"  - {h['date']}: Form (TSB): {h['form']}, Load (ATL): {h['fatigue']}" for h in stats.get('history', [])])

        if context == "status":
            prompt = (
                f"Analyze the athlete's current training status and give a 2-3 sentence recommendation on what they should focus on.\n\n"
                f"Current Metrics:\n"
                f"- Fitness (CTL, 42-day moving average - higher means fitter): {stats.get('fitness_ctl')}\n"
                f"- Fatigue (ATL, 7-day avg load - higher means more tired): {stats.get('fatigue_atl')}\n"
                f"- Form (TSB, Fitness minus Fatigue - negative means fatigued, positive means rested. During hard training, a TSB of -10 to -30 is common. -30 or lower indicates high risk of injury): {stats.get('form_tsb')}\n"
                f"- Estimated VO2 Max: {stats.get('latest_vo2_max', 'N/A')}\n"
                f"- Efficiency Factor 7d Avg (EF, higher is better aerobic efficiency): {stats.get('efficiency_factor_7d')}\n"
                f"- Aerobic Decoupling 7d Avg (Heart rate drift, lower is better, <5% is great): {stats.get('aerobic_decoupling_7d')}%\n\n"
                f"Recent 7 Day Trend:\n{history_str}\n\n"
                f"Talk directly to them as an expert endurance coach. Be concise and provide actionable advice based on their current load, form, and aerobic efficiency."
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
        system_prompt = "You are an expert running coach and one of your athletes is training for a half marathon that is on May 2nd 2026. The athlete is 28, is 5'11 and weighs 220lbs. His goal is to run a sub 2 hour half marathon. Explain what the metrics mean in a concise way and give actionable advice. Allow a bit of overtraining, they are very committed and love to push themselves. Don't be too conservative with the advice."

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
            url = "http://localhost:1234/api/v1/chat"
            payload = {
                "model": "mistralai/ministral-3-3b",
                "system_prompt": system_prompt,
                "input": prompt
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
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
        try:
            workouts = con.execute("SELECT name, description, url, category FROM dim_workouts WHERE category = ?", [target_category]).fetchall()
        except Exception as e:
            return {"error": "dim_workouts table not found. Run ingest_workouts.py first."}

        if not workouts:
             fallback_map = {
                 "Anaerobic": "VO2Max",
                 "VO2Max": "Threshold", 
                 "Threshold": "Aerobic",
                 "Aerobic": "Recovery",
                 "Recovery": "Aerobic" 
             }
             orig_target = target_category
             target_category = fallback_map.get(target_category, "Aerobic")
             workouts = con.execute("SELECT name, description, url, category FROM dim_workouts WHERE category = ?", [target_category]).fetchall()
             
        if not workouts:
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
        
        ai_reasoning = get_ai_insight(status, context="workout", workout=w_dict)
        
        return {
            "user_id": user_id,
            "current_tsb": tsb,
            "recommended_category": target_category,
            "ai_reasoning": ai_reasoning,
            "workout": w_dict
        }
    finally:
        con.close()

def calculate_vo2max(user_id: int):
    con = get_db_connection()
    try:
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
        
        results = df.groupby('activity_date')['vo2_max_est'].median().reset_index()
        results = results.sort_values('activity_date')
        
        results['vo2_max_rolling_7d'] = results['vo2_max_est'].rolling(window=7, min_periods=1).mean()

        latest = results.iloc[-1]
        
        history_list = []
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
    finally:
        con.close()

@app.get("/vo2max/{user_id}")
def get_vo2max(user_id: int):
    vo2max_data = calculate_vo2max(user_id)
    if not vo2max_data:
        raise HTTPException(status_code=404, detail="No stream data found to calculate VO2 Max for this user.")
    return vo2max_data
