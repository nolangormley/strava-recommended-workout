import os
import sys
import webbrowser
import requests
import json
import math
import time
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv, set_key

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from src.metrics import calculate_ctl_atl, calculate_tsb, get_target_category

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

CLIENT_ID = os.getenv('STRAVA_CLIENT_ID')
CLIENT_SECRET = os.getenv('STRAVA_CLIENT_SECRET')

REDIRECT_PORT = 5000
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"
SCOPES = "activity:read_all,activity:read"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, 'strava_warehouse.duckdb')

STREAMS_TO_FETCH = ["time", "heartrate", "velocity_smooth", "cadence", "watts", "temp", "altitude", "moving"]

class OAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query_components = parse_qs(urlparse(self.path).query)
        code = query_components.get('code', [None])[0]
        if code:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Authorization Successful!</h1><p>You can close this window now.</p></body></html>")
            self.server.auth_code = code
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Authorization failed.")
            
    def log_message(self, format, *args):
        pass # minimize output

class StravaManager:
    def __init__(self):
        if not CLIENT_ID or not CLIENT_SECRET:
            print("Error: STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET must be set in .env file.")
            sys.exit(1)
            
    def save_refresh_token(self, refresh_token):
        set_key(env_path, "STRAVA_REFRESH_TOKEN", refresh_token)
        load_dotenv(env_path, override=True)

    def run_oauth_flow(self):
        auth_url = f"https://www.strava.com/oauth/authorize?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&response_type=code&scope={SCOPES}"
        print(f"Please open URL to authorize: {auth_url}")
        try: webbrowser.open(auth_url)
        except: pass
        
        server = HTTPServer(('localhost', REDIRECT_PORT), OAuthHandler)
        server.auth_code = None
        server.handle_request()
        if server.auth_code: return self.exchange_code_for_token(server.auth_code)
        return None

    def exchange_code_for_token(self, code):
        data = requests.post("https://www.strava.com/oauth/token", data={
            'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET, 'code': code, 'grant_type': 'authorization_code'
        }).json()
        self.save_refresh_token(data['refresh_token'])
        return data['access_token']

    def get_access_token(self):
        refresh_token = os.getenv('STRAVA_REFRESH_TOKEN')
        if not refresh_token:
            print("No refresh token found. Starting OAuth flow...")
            return self.run_oauth_flow()
        try:
            print("Refreshing access token...")
            resp = requests.post("https://www.strava.com/oauth/token", data={
                'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET, 'refresh_token': refresh_token, 'grant_type': 'refresh_token'
            })
            if resp.status_code in [400, 401]:
                print(f"Refresh token invalid (status {resp.status_code}). Starting OAuth flow...")
                return self.run_oauth_flow()
            resp.raise_for_status()
            data = resp.json()
            if data.get('refresh_token') and data['refresh_token'] != refresh_token:
                print("New refresh token received, saving...")
                self.save_refresh_token(data['refresh_token'])
            return data['access_token']
        except Exception as e:
            print(f"Error getting access token: {e}")
            return None

    def fetch_activities(self, access_token, count=5, page=1, after=None):
        params = {'per_page': count, 'page': page}
        if after: params['after'] = after
        resp = requests.get("https://www.strava.com/api/v3/athlete/activities", headers={'Authorization': f'Bearer {access_token}'}, params=params)
        if resp.status_code == 401: raise Exception("Unauthorized")
        resp.raise_for_status()
        return resp.json()

    def fetch_activity_streams(self, access_token, activity_id, keys=STREAMS_TO_FETCH):
        print(f"  Fetching streams for activity {activity_id}...")
        resp = requests.get(f"https://www.strava.com/api/v3/activities/{activity_id}/streams", headers={'Authorization': f'Bearer {access_token}'}, params={'keys': ",".join(keys), 'key_by_type': 'true'})
        if resp.status_code == 401: raise Exception("Unauthorized")
        if resp.status_code == 404:
            print(f"  Streams not found for activity {activity_id}")
            return None
        resp.raise_for_status()
        return resp.json()

class StravaDB:
    def __init__(self, db_path=DB_PATH):
        self.con = duckdb.connect(db_path)
        self.init_schema()
        
    def init_schema(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS dim_athlete (
                athlete_id UBIGINT PRIMARY KEY, username VARCHAR, firstname VARCHAR, lastname VARCHAR,
                city VARCHAR, state VARCHAR, country VARCHAR, sex VARCHAR, weight DOUBLE, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        try:
            self.con.execute("ALTER TABLE dim_athlete ADD COLUMN weight DOUBLE;")
        except duckdb.CatalogException:
            pass
        except duckdb.BinderException:
            pass
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS dim_activity (
                activity_id UBIGINT PRIMARY KEY, athlete_id UBIGINT, name VARCHAR, type VARCHAR,
                start_date TIMESTAMP, start_date_local TIMESTAMP, distance DOUBLE, moving_time INTEGER,
                elapsed_time INTEGER, total_elevation_gain DOUBLE, average_speed DOUBLE, max_speed DOUBLE,
                average_heartrate DOUBLE, max_heartrate DOUBLE, average_cadence DOUBLE, calories DOUBLE,
                device_name VARCHAR, description VARCHAR, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (athlete_id) REFERENCES dim_athlete(athlete_id)
            );
        """)
        def create_stream_table(name, val_type="DOUBLE"):
            self.con.execute(f"CREATE TABLE IF NOT EXISTS stream_{name} (activity_id UBIGINT, time_offset INTEGER, value {val_type}, FOREIGN KEY (activity_id) REFERENCES dim_activity(activity_id));")
            
        create_stream_table("heartrate", "INTEGER")
        create_stream_table("velocity", "DOUBLE")
        create_stream_table("cadence", "INTEGER")
        create_stream_table("watts", "INTEGER")
        create_stream_table("temperature", "INTEGER")
        create_stream_table("altitude", "DOUBLE")
        create_stream_table("moving", "BOOLEAN")

    def upsert_athlete(self, a):
        self.con.execute("""
            INSERT INTO dim_athlete (athlete_id, username, firstname, lastname, city, state, country, sex, weight) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (athlete_id) DO UPDATE SET username=EXCLUDED.username, firstname=EXCLUDED.firstname, lastname=EXCLUDED.lastname, city=EXCLUDED.city, state=EXCLUDED.state, country=EXCLUDED.country, sex=EXCLUDED.sex, weight=EXCLUDED.weight, updated_at=now();
        """, [a.get('id'), a.get('username'), a.get('firstname'), a.get('lastname'), a.get('city'), a.get('state'), a.get('country'), a.get('sex'), a.get('weight')])

    def upsert_activity(self, a):
        self.con.execute("""
            INSERT INTO dim_activity (activity_id, athlete_id, name, type, start_date, start_date_local, distance, moving_time, elapsed_time, total_elevation_gain, average_speed, max_speed, average_heartrate, max_heartrate, average_cadence, calories, device_name, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (activity_id) DO UPDATE SET name=EXCLUDED.name, type=EXCLUDED.type, distance=EXCLUDED.distance, moving_time=EXCLUDED.moving_time, elapsed_time=EXCLUDED.elapsed_time, total_elevation_gain=EXCLUDED.total_elevation_gain, average_speed=EXCLUDED.average_speed, max_speed=EXCLUDED.max_speed, average_heartrate=EXCLUDED.average_heartrate, updated_at=now();
        """, [a.get('id'), a.get('athlete', {}).get('id'), a.get('name'), a.get('type'), a.get('start_date'), a.get('start_date_local'), a.get('distance'), a.get('moving_time'), a.get('elapsed_time'), a.get('total_elevation_gain'), a.get('average_speed'), a.get('max_speed'), a.get('average_heartrate'), a.get('max_heartrate'), a.get('average_cadence'), a.get('calories'), a.get('device_name'), a.get('description')])

    def insert_streams(self, activity_id, streams):
        if not streams or 'time' not in streams: return
        for t in ["stream_heartrate", "stream_velocity", "stream_cadence", "stream_watts", "stream_temperature", "stream_altitude", "stream_moving"]:
            self.con.execute(f"DELETE FROM {t} WHERE activity_id = ?", [activity_id])
        time_data = streams['time']['data']
        def proc(key, tbl):
            if key in streams:
                self.con.executemany(f"INSERT INTO stream_{tbl} VALUES (?, ?, ?)", list(zip([activity_id]*len(time_data), time_data, streams[key]['data'])))
        proc('heartrate', 'heartrate')
        proc('velocity_smooth', 'velocity')
        proc('cadence', 'cadence')
        proc('watts', 'watts')
        proc('temp', 'temperature')
        proc('altitude', 'altitude')
        proc('moving', 'moving')

    def activity_exists(self, act_id): return self.con.execute("SELECT 1 FROM dim_activity WHERE activity_id = ?", [act_id]).fetchone() is not None
    def activity_has_streams(self, act_id): return self.con.execute("SELECT 1 FROM stream_velocity WHERE activity_id = ? LIMIT 1", [act_id]).fetchone() is not None

def ingest_data(manager, db, access_token, lookback_days=42):
    after_ts = int((datetime.now() - timedelta(days=lookback_days)).timestamp())
    print(f"Limiting ingestion to activities after {datetime.fromtimestamp(after_ts)} ({lookback_days} days lookback)")
    
    try:
        print("Fetching athlete profile...")
        athlete = requests.get("https://www.strava.com/api/v3/athlete", headers={'Authorization': f'Bearer {access_token}'}).json()
        db.upsert_athlete(athlete)
        print(f"Athlete: {athlete.get('firstname')} {athlete.get('lastname')}")
    except Exception as e:
        print(f"Warning: Could not fetch athlete profile: {e}")

    after_timestamp = int(time.time() - (lookback_days * 24 * 60 * 60))

    page = 1
    processed_count = 0
    while True:
        print(f"Fetching activities page {page}...")
        try:
            activities = manager.fetch_activities(access_token, count=50, page=page, after=after_ts)
        except Exception as e:
            print(f"Error fetching activities on page {page}: {e}")
            if "Unauthorized" in str(e):
                print("Token expired during ingest, re-authenticating...")
                access_token = manager.run_oauth_flow()
                if not access_token: break
                continue
            break
            
        if not activities:
            print("No more activities found.")
            break
        
        print(f"Found {len(activities)} activities on page {page}.")
        for act in activities:
            act_id = act['id']
            act_name = act['name']
            exists = db.activity_exists(act_id)
            has_streams = db.activity_has_streams(act_id)
            
            if not exists or not has_streams:
                if not exists:
                    print(f"  New activity: {act_name} ({act_id})")
                else:
                    print(f"  Activity exists but missing streams: {act_name} ({act_id})")
                
                db.upsert_activity(act)
                try:
                    streams = manager.fetch_activity_streams(access_token, act_id)
                    if streams:
                        db.insert_streams(act_id, streams)
                        print(f"    Streams saved.")
                except Exception as str_e:
                    print(f"    Error fetching streams for {act_id}: {str_e}")
                    if "Unauthorized" in str(str_e):
                        print("    Token expired during stream fetch, re-authenticating...")
                        access_token = manager.run_oauth_flow()
                        if access_token:
                            streams = manager.fetch_activity_streams(access_token, act_id)
                            if streams: db.insert_streams(act_id, streams)
                processed_count += 1
            else:
                # Debug: skip message
                # print(f"  Skipping {act_name} (already indexed)")
                pass
                
        if len(activities) < 50:
            print("Reached last page of activities.")
            break
        page += 1
    return processed_count

def calculate_trimp_banister(duration_sec, avg_hr, max_hr, rest_hr, is_male):
    if max_hr <= rest_hr: return 0
    hr_ratio = max(0, (avg_hr - rest_hr) / (max_hr - rest_hr))
    return (duration_sec / 60.0) * hr_ratio * (0.64 * math.exp((1.92 if is_male else 1.67) * hr_ratio))

def calculate_trimp_edwards(hr_series, max_hr):
    if not hr_series: return 0
    hr_array = np.array(hr_series)
    return ((np.sum((hr_array >= max_hr * 0.5) & (hr_array < max_hr * 0.6)) * 1) + (np.sum((hr_array >= max_hr * 0.6) & (hr_array < max_hr * 0.7)) * 2) + (np.sum((hr_array >= max_hr * 0.7) & (hr_array < max_hr * 0.8)) * 3) + (np.sum((hr_array >= max_hr * 0.8) & (hr_array < max_hr * 0.9)) * 4) + (np.sum(hr_array >= max_hr * 0.9) * 5)) / 60.0

def calculate_time_in_zones(hr_series, max_hr):
    if not hr_series: return [0,0,0,0,0]
    hr_array = np.array(hr_series)
    return [int(np.sum((hr_array >= max_hr * 0.5) & (hr_array < max_hr * 0.6))), int(np.sum((hr_array >= max_hr * 0.6) & (hr_array < max_hr * 0.7))), int(np.sum((hr_array >= max_hr * 0.7) & (hr_array < max_hr * 0.8))), int(np.sum((hr_array >= max_hr * 0.8) & (hr_array < max_hr * 0.9))), int(np.sum(hr_array >= max_hr * 0.9))]

def calculate_aerobic_decoupling(hr_times, hr_values, vel_times, vel_values):
    if not hr_values or not vel_values: return None
    df = pd.merge(pd.DataFrame({'time': hr_times, 'hr': hr_values}), pd.DataFrame({'time': vel_times, 'vel': vel_values}), on='time', how='inner')
    df = df[df['vel'] > 0.1]
    if len(df) < 60: return None
    duration = df['time'].max()
    midpoint = duration / 2
    fh, sh = df[df['time'] <= midpoint], df[df['time'] > midpoint]
    if len(fh) == 0 or len(sh) == 0: return None
    ef1, ef2 = (fh['vel'].mean() * 60) / fh['hr'].mean() if fh['hr'].mean() > 0 else 0, (sh['vel'].mean() * 60) / sh['hr'].mean() if sh['hr'].mean() > 0 else 0
    if ef1 == 0: return 0
    return ((ef1 - ef2) / ef1) * 100

def run_analyze_effectiveness(db):
    con = db.con
    con.execute("DROP TABLE IF EXISTS activity_effectiveness")
    con.execute("CREATE TABLE IF NOT EXISTS activity_effectiveness (activity_id UBIGINT PRIMARY KEY, trimp_banister DOUBLE, trimp_edwards DOUBLE, efficiency_factor DOUBLE, intensity_factor DOUBLE, aerobic_decoupling DOUBLE, zone_1_sec INTEGER, zone_2_sec INTEGER, zone_3_sec INTEGER, zone_4_sec INTEGER, zone_5_sec INTEGER, effectiveness_score DOUBLE, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (activity_id) REFERENCES dim_activity(activity_id));")
    
    acts = con.execute("SELECT da.activity_id, da.name, da.start_date_local, da.max_heartrate, da.average_speed, ath.sex FROM dim_activity da JOIN stream_heartrate sh ON da.activity_id = sh.activity_id JOIN dim_athlete ath ON da.athlete_id = ath.athlete_id GROUP BY 1,2,3,4,5,6").fetchall()
    
    processed = 0
    total = len(acts)
    for i, (act_id, name, date, rep_max_hr, avg_speed, sex) in enumerate(acts):
        if i % 10 == 0:
            print(f"  Processing {i}/{total}: {name}...")
        
        hr_data = con.execute(f"SELECT value, time_offset FROM stream_heartrate WHERE activity_id = {act_id} ORDER BY time_offset").fetchall()
        if not hr_data: continue
        hr_vals, times = [h[0] for h in hr_data], [h[1] for h in hr_data]
        user_max_hr = rep_max_hr if (rep_max_hr and rep_max_hr > 150) else 192
        avg_hr = sum(hr_vals) / len(hr_vals)
        
        is_male = (sex == 'M')
        trimp_b = 0
        prev_t = times[0]
        for idx in range(1, len(hr_vals)):
            dt = (times[idx] - prev_t) / 60.0
            if dt > 5: dt = 0
            trimp_b += calculate_trimp_banister(dt * 60, hr_vals[idx], user_max_hr, 60, is_male)
            prev_t = times[idx]
            
        trimp_e = calculate_trimp_edwards(hr_vals, user_max_hr)
        zones = calculate_time_in_zones(hr_vals, user_max_hr)
        ef = ((avg_speed or 0) * 60) / avg_hr if avg_hr > 0 else 0
        intensity = avg_hr / user_max_hr
        
        vel_data = con.execute(f"SELECT value, time_offset FROM stream_velocity WHERE activity_id = {act_id} ORDER BY time_offset").fetchall()
        decoupling = calculate_aerobic_decoupling(times, hr_vals, [v[1] for v in vel_data], [v[0] for v in vel_data]) if vel_data else None
        
        con.execute("INSERT INTO activity_effectiveness (activity_id, trimp_banister, trimp_edwards, efficiency_factor, intensity_factor, aerobic_decoupling, zone_1_sec, zone_2_sec, zone_3_sec, zone_4_sec, zone_5_sec, effectiveness_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [act_id, trimp_b, trimp_e, ef, intensity, decoupling, zones[0], zones[1], zones[2], zones[3], zones[4], trimp_b])
        processed += 1
    return processed

def run_analyze_training_load(db):
    con = db.con
    df = con.execute("SELECT da.start_date_local::DATE as activity_date, SUM(ae.trimp_banister) as daily_load, AVG(ae.efficiency_factor) as daily_ef, AVG(ae.aerobic_decoupling) as daily_decoup FROM activity_effectiveness ae JOIN dim_activity da ON ae.activity_id = da.activity_id GROUP BY 1 ORDER BY 1").fetchdf()
    if df.empty: return None
    
    start_date, end_date = df['activity_date'].min(), datetime.now().date()
    daily_data = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date, freq='D').date})
    daily_data['activity_date'] = pd.to_datetime(daily_data['date'])
    df['activity_date'] = pd.to_datetime(df['activity_date'])
    
    merged = pd.merge(daily_data, df, on='activity_date', how='left')
    merged['daily_load'] = merged['daily_load'].fillna(0)
    ctl, atl = calculate_ctl_atl(merged['daily_load'].values)
    merged['CTL'], merged['ATL'] = ctl, atl
    merged['TSB'] = calculate_tsb(ctl, atl)
    
    today = merged.iloc[-1]
    return {"CTL": today['CTL'], "ATL": today['ATL'], "TSB": today['TSB']}

def main():
    manager = StravaManager()
    access_token = manager.get_access_token()
    if not access_token:
        print("Failed to authenticate with Strava.")
        return
        
    db = StravaDB()
    print("Ingesting Strava data (past 42 days)...")
    count = ingest_data(manager, db, access_token, lookback_days=42)
    print(f"Ingestion complete ({count} new activities).")
    
    print("Analyzing effectiveness...", end='', flush=True)
    eff_count = run_analyze_effectiveness(db)
    print(f" Done ({eff_count} activities processed).")
    
    print("Calculating training load...", end='', flush=True)
    stats = run_analyze_training_load(db)
    if stats:
        print(f" Done.\n-> Status: CTL: {stats['CTL']:.1f}, ATL: {stats['ATL']:.1f}, TSB: {stats['TSB']:.1f} ({get_target_category(stats['TSB'])})")
        print(" Failed (no data).")
        
    db.con.close()

if __name__ == "__main__":
    main()
