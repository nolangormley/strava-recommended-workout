import duckdb
import math
import pandas as pd
import numpy as np

# Constants (You can adjust these or we can make them arguments)
DEFAULT_MAX_HR = 192
DEFAULT_REST_HR = 60
IS_MALE = True # Needed for Banister's constant (b=1.92 for men, 1.67 for women)

import os
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'strava_warehouse.duckdb')

def calculate_trimp_banister(duration_sec, avg_hr, max_hr, rest_hr, is_male):
    """
    Calculates Banister's Training Impulse (TRIMP).
    Source: Banister, E.W. (1991). Modeling Elite Athletic Performance.
    
    TRIMP = Duration(min) * HR_ratio * X
    Where X = 0.64 * exp(Y * HR_ratio)
    Y = 1.92 for men, 1.67 for women
    HR_ratio = (HR_ex - HR_rest) / (HR_max - HR_rest)
    """
    if max_hr <= rest_hr: return 0
    
    duration_min = duration_sec / 60.0
    hr_ratio = (avg_hr - rest_hr) / (max_hr - rest_hr)
    
    # Ensure ratio is within bounds (0 to 1) for sanity, though Banister allows >1 technically (anaerobic)
    hr_ratio = max(0, hr_ratio)
    
    y_factor = 1.92 if is_male else 1.67
    weighting_factor = 0.64 * math.exp(y_factor * hr_ratio)
    
    return duration_min * hr_ratio * weighting_factor

def calculate_trimp_edwards(hr_series, max_hr):
    """
    Calculates Edwards' TRIMP based on time in zones.
    Source: Edwards, S. (1993). The Heart Rate Monitor Book.
    
    Zone 1: 50-60% Max HR (1 pt)
    Zone 2: 60-70% Max HR (2 pts)
    Zone 3: 70-80% Max HR (3 pts)
    Zone 4: 80-90% Max HR (4 pts)
    Zone 5: 90-100% Max HR (5 pts)
    """
    if len(hr_series) == 0: return 0
    
    zones = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    total_score = 0
    
    # We assume 1 second per point if it's raw stream data usually diff is 1s, but we should check time.
    # For now, simplistic counting of seconds in zone.
    
    # Vectorized approach with numpy/pandas is faster
    hr_array = np.array(hr_series)
    
    # Create masks for zones
    # Zone 1
    z1 = ((hr_array >= max_hr * 0.5) & (hr_array < max_hr * 0.6))
    z2 = ((hr_array >= max_hr * 0.6) & (hr_array < max_hr * 0.7))
    z3 = ((hr_array >= max_hr * 0.7) & (hr_array < max_hr * 0.8))
    z4 = ((hr_array >= max_hr * 0.8) & (hr_array < max_hr * 0.9))
    z5 = (hr_array >= max_hr * 0.9)
    
    # TRIMP = Sum(minutes_in_zone * zone_weight)
    # Our data is seconds (assuming 1 row = 1 sec approx).
    # Edwards formula uses minutes.
    
    score = (np.sum(z1) * 1) + (np.sum(z2) * 2) + (np.sum(z3) * 3) + (np.sum(z4) * 4) + (np.sum(z5) * 5)
    return score / 60.0

def calculate_time_in_zones(hr_series, max_hr):
    """
    Returns the time (in seconds) spent in each of the 5 Edwards zones.
    """
    if len(hr_series) == 0: return [0, 0, 0, 0, 0]
    
    hr_array = np.array(hr_series)
    z1 = ((hr_array >= max_hr * 0.5) & (hr_array < max_hr * 0.6))
    z2 = ((hr_array >= max_hr * 0.6) & (hr_array < max_hr * 0.7))
    z3 = ((hr_array >= max_hr * 0.7) & (hr_array < max_hr * 0.8))
    z4 = ((hr_array >= max_hr * 0.8) & (hr_array < max_hr * 0.9))
    z5 = (hr_array >= max_hr * 0.9)
    
    # Assuming 1 second per point (approx)
    return [int(np.sum(z1)), int(np.sum(z2)), int(np.sum(z3)), int(np.sum(z4)), int(np.sum(z5))]

def calculate_aerobic_decoupling(hr_times, hr_values, vel_times, vel_values):
    """
    Calculates Aerobic Decoupling (Pa:HR ratio drift).
    Decoupling = (EF_first_half - EF_second_half) / EF_first_half
    Where EF (Efficiency Factor) = Avg Speed / Avg HR.
    """
    if not hr_values or not vel_values:
        return None
        
    # Align data by converting to pandas series or dataframe for easier time-based indexing
    # Create simple dataframes
    df_hr = pd.DataFrame({'time': hr_times, 'hr': hr_values})
    df_vel = pd.DataFrame({'time': vel_times, 'vel': vel_values})
    
    # Merge on time (approximate match or exact if streams are consistent)
    # Strava streams usually have exact matching offsets if requested together
    # But let's just use exact match on 'time'
    df = pd.merge(df_hr, df_vel, on='time', how='inner')
    
    if len(df) < 60: # Too short to be meaningful
        return None
    
    # Filter out stops (velocity near 0)
    df = df[df['vel'] > 0.1] # moving only
    
    if len(df) < 60:
        return None
        
    duration = df['time'].max()
    midpoint = duration / 2
    
    first_half = df[df['time'] <= midpoint]
    second_half = df[df['time'] > midpoint]
    
    if len(first_half) == 0 or len(second_half) == 0:
        return None
        
    avg_hr1 = first_half['hr'].mean()
    avg_vel1 = first_half['vel'].mean()
    
    avg_hr2 = second_half['hr'].mean()
    avg_vel2 = second_half['vel'].mean()
    
    if avg_hr1 == 0 or avg_hr2 == 0:
        return None
        
    # Calculate EF (Speed / HR)
    # Using m/min for speed to make numbers nicer, though ratio is unitless essentially
    ef1 = (avg_vel1 * 60) / avg_hr1
    ef2 = (avg_vel2 * 60) / avg_hr2
    
    if ef1 == 0: return 0
    
    # TODO: Research how to handle negative decoupling (e.g., -5%).
    # Currently it means efficiency IMPROVED in the second half (negative drift).
    # This often happens during interval sets where rest periods lower average HR,
    # or if the user significantly speeds up (negative split) without corresponding HR spike.
    # Consider filtering out values < -10% or handling interval workouts differently.
    return ((ef1 - ef2) / ef1) * 100

def analyze_effectiveness():
    con = duckdb.connect(DB_PATH)
    
    # 1. Create Analysis Table
    # Recreate to ensure schema update
    con.execute("DROP TABLE IF EXISTS activity_effectiveness")
    con.execute("""
        CREATE TABLE IF NOT EXISTS activity_effectiveness (
            activity_id UBIGINT PRIMARY KEY,
            trimp_banister DOUBLE,
            trimp_edwards DOUBLE,
            efficiency_factor DOUBLE, -- Speed/HR Ratio
            intensity_factor DOUBLE,  -- Avg HR / Max HR
            aerobic_decoupling DOUBLE, -- Pa:HR Drift %
            zone_1_sec INTEGER,
            zone_2_sec INTEGER,
            zone_3_sec INTEGER,
            zone_4_sec INTEGER,
            zone_5_sec INTEGER,
            effectiveness_score DOUBLE, -- Custom composite
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (activity_id) REFERENCES dim_activity(activity_id)
        );
    """)
    
    # 2. Get Activities with Heart Rate Data
    activities = con.execute("""
        SELECT 
            da.activity_id, 
            da.name, 
            da.start_date_local,
            da.max_heartrate,
            da.average_speed
        FROM dim_activity da
        JOIN stream_heartrate sh ON da.activity_id = sh.activity_id
        GROUP BY da.activity_id, da.name, da.start_date_local, da.max_heartrate, da.average_speed
    """).fetchall()
    
    print(f"Analyzing {len(activities)} activities with HR data...")
    
    for act in activities:
        act_id, name, date, reported_max_hr, avg_speed = act
        
        # Use reported Max HR if available and reasonable (>150), else default
        user_max_hr = reported_max_hr if (reported_max_hr and reported_max_hr > 150) else DEFAULT_MAX_HR
        
        # Fetch HR stream (time, value)
        hr_data = con.execute(f"""
            SELECT value, time_offset 
            FROM stream_heartrate 
            WHERE activity_id = {act_id} 
            ORDER BY time_offset
        """).fetchall()
        
        hr_values = [h[0] for h in hr_data]
        times = [h[1] for h in hr_data]
        
        if not hr_values:
            continue
            
        # Calculate duration in seconds (max time offset)
        duration_sec = times[-1] - times[0] if times else 0
        if duration_sec == 0: duration_sec = len(hr_values) # fallback
        
        avg_hr = sum(hr_values) / len(hr_values)
        
        # 1. Banister TRIMP
        # We calculate it point-by-point (more accurate) or avg (easier).
        # Typically point-by-point ("integrating" the curve) is better for interval workouts.
        # Let's do point-by-point summation.
        trimp_b_sum = 0
        prev_time = times[0]
        
        for i in range(1, len(hr_values)):
            dt_min = (times[i] - prev_time) / 60.0
            if dt_min > 5: # gap check
                 dt_min = 0 
            
            hr_curr = hr_values[i]
            # Single point calculation
            val = calculate_trimp_banister(dt_min * 60, hr_curr, user_max_hr, DEFAULT_REST_HR, IS_MALE)
            trimp_b_sum += val
            prev_time = times[i]
            
        # 2. Edwards TRIMP & Zones
        trimp_e = calculate_trimp_edwards(hr_values, user_max_hr)
        zones = calculate_time_in_zones(hr_values, user_max_hr)
        
        # 3. Efficiency Factor (EF)
        # Normalized Speed (m/min) / Avg HR
        # Speed is m/s -> m/min = speed * 60
        avg_speed_m_min = (avg_speed or 0) * 60
        ef = avg_speed_m_min / avg_hr if avg_hr > 0 else 0
        
        # 4. Intensity Factor (Simple)
        intensity = avg_hr / user_max_hr

        # 5. Aerobic Decoupling
        # Fetch velocity stream
        vel_stream = con.execute(f"""
            SELECT value, time_offset FROM stream_velocity WHERE activity_id = {act_id} ORDER BY time_offset
        """).fetchall()
        
        decoupling = None
        if vel_stream:
            vel_values = [v[0] for v in vel_stream]
            vel_times = [v[1] for v in vel_stream]
            decoupling = calculate_aerobic_decoupling(times, hr_values, vel_times, vel_values)
        
        # Persist
        # Persist
        con.execute("""
            INSERT INTO activity_effectiveness (
                activity_id, trimp_banister, trimp_edwards, efficiency_factor, intensity_factor, 
                aerobic_decoupling, zone_1_sec, zone_2_sec, zone_3_sec, zone_4_sec, zone_5_sec, 
                effectiveness_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [act_id, trimp_b_sum, trimp_e, ef, intensity, decoupling, zones[0], zones[1], zones[2], zones[3], zones[4], trimp_b_sum])
        
        decoupling_str = f"{decoupling:.1f}%" if decoupling is not None else "N/A"
        print(f"Analyzed {name}: TRIMP(b)={trimp_b_sum:.1f}, EF={ef:.2f}, Decoup={decoupling_str}")

    # Summary Report
    print("\n" + "="*60)
    print(f"EFFECTIVENESS RANKING (Sorted by Banister TRIMP)")
    print("="*60)
    results = con.execute("""
        SELECT 
            da.name, 
            da.type, 
            da.start_date_local, 
            ae.trimp_banister, 
            ae.trimp_edwards, 
            ae.efficiency_factor,
            ae.aerobic_decoupling,
            ae.zone_5_sec
        FROM activity_effectiveness ae
        JOIN dim_activity da ON ae.activity_id = da.activity_id
        ORDER BY da.start_date_local DESC
        LIMIT 20
    """).fetchall()
    
    print(f"{'Activity':<25} | {'Date':<10} | {'Banister':<8} | {'EF':<5} | {'Decoup':<7} | {'Z5 (min)':<8}")
    print("-" * 80)
    for res in results:
        name = (res[0][:22] + '..') if len(res[0]) > 22 else res[0]
        date = str(res[2])[:10]
        decoup = f"{res[6]:.1f}%" if res[6] is not None else "-"
        z5_min = round(res[7]/60, 1) if res[7] else 0
        print(f"{name:<25} | {date:<10} | {res[3]:<8.1f} | {res[5]:<5.2f} | {decoup:<7} | {z5_min:<8}")

if __name__ == "__main__":
    analyze_effectiveness()
