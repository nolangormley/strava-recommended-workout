import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import os
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'strava_warehouse.duckdb')

def calculate_training_load():
    con = duckdb.connect(DB_PATH)
    
    # 1. Get Daily TRIMP Sums
    # We need to aggregate loads by day, because you might do 2 workouts a day.
    df = con.execute("""
        SELECT 
            da.start_date_local::DATE as activity_date,
            SUM(ae.trimp_banister) as daily_load
        FROM activity_effectiveness ae
        JOIN dim_activity da ON ae.activity_id = da.activity_id
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()
    
    if df.empty:
        print("No training data found.")
        return

    # 2. Reindex to include rest days (0 load)
    # Start from first activity date to today
    start_date = df['activity_date'].min()
    end_date = datetime.now().date()
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_data = pd.DataFrame({'date': date_range.date})
    
    # Ensure types match for merge
    daily_data['activity_date'] = pd.to_datetime(daily_data['date'])
    df['activity_date'] = pd.to_datetime(df['activity_date'])
    
    merged = pd.merge(daily_data, df, on='activity_date', how='left').fillna(0)
    
    # 3. Calculate CTL (Fitness), ATL (Fatigue), TSB (Form)
    # Constants
    CTL_DECAY = 42 # days
    ATL_DECAY = 7  # days
    
    # Decaying average factors: alpha = 2 / (N + 1) or simplified 1/N for EWMA often used in cycling
    # Coggan formula: CTL_today = CTL_yesterday + (TRIMP_today - CTL_yesterday) / TimeConstant
    
    ctl = [0.0]
    atl = [0.0]
    
    # Initialize with 0 assuming user was inactive before. 
    # In reality, you'd seed this with an estimate if history is short.
    
    loads = merged['daily_load'].values
    dates = merged['date'].values
    
    for i in range(len(loads)):
        daily_strain = loads[i]
        
        # Previous values
        prev_ctl = ctl[-1]
        prev_atl = atl[-1]
        
        # Update
        curr_ctl = prev_ctl + (daily_strain - prev_ctl) / CTL_DECAY
        curr_atl = prev_atl + (daily_strain - prev_atl) / ATL_DECAY
        
        ctl.append(curr_ctl)
        atl.append(curr_atl)
        
    # Remove initial seed
    merged['CTL'] = ctl[1:]
    merged['ATL'] = atl[1:]
    merged['TSB'] = merged['CTL'] - merged['ATL']
    
    # 4. Display Status
    print("\n" + "="*60)
    print("CURRENT TRAINING STATUS")
    print("="*60)
    
    today_stats = merged.iloc[-1]
    formatted_date = today_stats['date'].strftime("%Y-%m-%d")
    
    print(f"Date:    {formatted_date}")
    print(f"Fitness (CTL): {today_stats['CTL']:.1f}")
    print(f"Fatigue (ATL): {today_stats['ATL']:.1f}")
    print(f"Form    (TSB): {today_stats['TSB']:.1f}")
    
    print("-" * 60)
    
    # 5. Recommendation Logic
    tsb = today_stats['TSB']
    print("RECOMMENDATION:")
    
    if tsb > 5:
        print("[!] You are very fresh (Positive TSB).")
        print("    -> Good time for a Time Trial, Race, or very hard Interval session.")
        print("    -> Alternatively, increase volume to build fitness.")
    elif -10 <= tsb <= 5:
        print("[+] You are in the 'Grey Zone' (Neutral TSB).")
        print("    -> Maintain training. You can handle a standard workout.")
        print("    -> Focus on specific adaptations (Tempo, Threshold).")
    elif -30 <= tsb < -10:
        print("[*] You are in the 'Optimal Training Zone'.")
        print("    -> You are accumulating fatigue at a sustainable rate.")
        print("    -> Keep overloading progressively, but monitor sleep/recovery.")
        print("    -> Recommended: Endurance Level 2 or Sweet Spot.")
    else: # tsb < -30
        print("[!] HIGH FATIGUE WARNING (TSB < -30).")
        print("    -> You are at risk of overtraining or illness.")
        print("    -> Recommended: REST DAY or active recovery (Zone 1 spin/jog).")
        category = "Recovery"

    # 6. Specific Workout Recommendation (Wrapper)
    print("\n" + "="*60)
    print("RECOMMENDED WORKOUT (from Coros Library)")
    print("="*60)
    
    # Map TSB to Workout Category
    if tsb > 5:
        target_category = "Anaerobic" # or VO2Max
    elif -10 <= tsb <= 5:
        target_category = "VO2Max" # or Threshold if available
    elif -30 <= tsb < -10:
        target_category = "Aerobic"
    else:
        target_category = "Recovery"
        
    # Query Database
    workouts = con.execute(f"SELECT name, description, url FROM dim_workouts WHERE category = '{target_category}'").fetchall()
    
    # Fallback logic if specific category is empty
    if not workouts:
        if target_category == "Anaerobic": target_category = "VO2Max"
        elif target_category == "VO2Max": target_category = "Threshold"
        elif target_category == "Recovery": target_category = "Aerobic"
        workouts = con.execute(f"SELECT name, description, url FROM dim_workouts WHERE category = '{target_category}'").fetchall()
    
    if workouts:
        import random
        w = random.choice(workouts)
        print(f"Workout: {w[0]}")
        print(f"Type:    {target_category}")
        print(f"Goal:    {w[1]}")
        print(f"Link:    {w[2]}")
    else:
        print("No specific workout found in library for this category.")

    # 7. Recent History Table
    print("\nRecent History (Last 7 Days):")
    print(f"{'Date':<12} | {'Load':<6} | {'Fitness':<8} | {'Fatigue':<8} | {'Form':<6}")
    print("-" * 50)
    
    recent = merged.tail(7)
    for index, row in recent.iterrows():
        d = row['date'].strftime("%m-%d")
        print(f"{d:<12} | {row['daily_load']:<6.0f} | {row['CTL']:<8.1f} | {row['ATL']:<8.1f} | {row['TSB']:<6.1f}")

if __name__ == "__main__":
    calculate_training_load()
