import duckdb
import requests
import json
import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fetch_strava_data import StravaManager

# Connect to database (or create it)
# By default, files in 'data' directory in project root
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, 'strava_warehouse.duckdb')

STREAMS_TO_FETCH = ["time", "heartrate", "velocity_smooth", "cadence", "watts", "temp", "altitude", "moving"]

class StravaDB:
    def __init__(self, db_path=DB_PATH):
        self.con = duckdb.connect(db_path)
        self.init_schema()
        
    def init_schema(self):
        # 1. Dimension Tables
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS dim_athlete (
                athlete_id UBIGINT PRIMARY KEY,
                username VARCHAR,
                firstname VARCHAR,
                lastname VARCHAR,
                city VARCHAR,
                state VARCHAR,
                country VARCHAR,
                sex VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS dim_activity (
                activity_id UBIGINT PRIMARY KEY,
                athlete_id UBIGINT,
                name VARCHAR,
                type VARCHAR,
                start_date TIMESTAMP,
                start_date_local TIMESTAMP,
                distance DOUBLE,       -- meters
                moving_time INTEGER,   -- seconds
                elapsed_time INTEGER,  -- seconds
                total_elevation_gain DOUBLE, -- meters
                average_speed DOUBLE,  -- m/s
                max_speed DOUBLE,      -- m/s
                average_heartrate DOUBLE,
                max_heartrate DOUBLE,
                average_cadence DOUBLE,
                calories DOUBLE,
                device_name VARCHAR,
                description VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (athlete_id) REFERENCES dim_athlete(athlete_id)
            );
        """)

        # 2. Fact Tables (Time Series Streams)
        # Using a unified time series table or separate?
        # User requested separate tables.
        
        # Generic function for stream tables
        def create_stream_table(table_name, value_type="DOUBLE"):
             self.con.execute(f"""
                CREATE TABLE IF NOT EXISTS stream_{table_name} (
                    activity_id UBIGINT,
                    time_offset INTEGER, -- seconds from start
                    value {value_type},
                    FOREIGN KEY (activity_id) REFERENCES dim_activity(activity_id)
                );
            """)
            
        create_stream_table("heartrate", "INTEGER")
        create_stream_table("velocity", "DOUBLE") # velocity_smooth
        create_stream_table("cadence", "INTEGER")
        create_stream_table("watts", "INTEGER")
        create_stream_table("temperature", "INTEGER")
        create_stream_table("altitude", "DOUBLE")
        create_stream_table("moving", "BOOLEAN")
        
        # Also create a unified view for convenience
        # (Optional, but good practice)

    def upsert_athlete(self, athlete_data):
        self.con.execute("""
            INSERT INTO dim_athlete (athlete_id, username, firstname, lastname, city, state, country, sex)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (athlete_id) DO UPDATE SET
                username = EXCLUDED.username,
                firstname = EXCLUDED.firstname,
                lastname = EXCLUDED.lastname,
                city = EXCLUDED.city,
                state = EXCLUDED.state,
                country = EXCLUDED.country,
                sex = EXCLUDED.sex,
                updated_at = now();
        """, [
            athlete_data.get('id'),
            athlete_data.get('username'),
            athlete_data.get('firstname'),
            athlete_data.get('lastname'),
            athlete_data.get('city'),
            athlete_data.get('state'),
            athlete_data.get('country'),
            athlete_data.get('sex')
        ])

    def upsert_activity(self, activity):
        self.con.execute("""
            INSERT INTO dim_activity (
                activity_id, athlete_id, name, type, start_date, start_date_local,
                distance, moving_time, elapsed_time, total_elevation_gain,
                average_speed, max_speed, average_heartrate, max_heartrate,
                average_cadence, calories, device_name, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (activity_id) DO UPDATE SET
                name = EXCLUDED.name,
                type = EXCLUDED.type,
                distance = EXCLUDED.distance,
                moving_time = EXCLUDED.moving_time,
                elapsed_time = EXCLUDED.elapsed_time,
                total_elevation_gain = EXCLUDED.total_elevation_gain,
                average_speed = EXCLUDED.average_speed,
                max_speed = EXCLUDED.max_speed,
                average_heartrate = EXCLUDED.average_heartrate,
                updated_at = now();
        """, [
            activity.get('id'),
            activity.get('athlete', {}).get('id'),
            activity.get('name'),
            activity.get('type'),
            activity.get('start_date'),
            activity.get('start_date_local'),
            activity.get('distance'),
            activity.get('moving_time'),
            activity.get('elapsed_time'),
            activity.get('total_elevation_gain'),
            activity.get('average_speed'),
            activity.get('max_speed'),
            activity.get('average_heartrate'),
            activity.get('max_heartrate'),
            activity.get('average_cadence'),
            activity.get('calories'),
            activity.get('device_name'),
            activity.get('description')
        ])

    def insert_streams(self, activity_id, streams):
        if not streams:
            return

        # Clear existing streams for this activity to avoid duplicates
        tables = ["stream_heartrate", "stream_velocity", "stream_cadence", 
                  "stream_watts", "stream_temperature", "stream_altitude", "stream_moving"]
        for t in tables:
            self.con.execute(f"DELETE FROM {t} WHERE activity_id = ?", [activity_id])

        # Get time stream (required for mapping)
        if 'time' not in streams:
            print(f"Warning: No time stream for activity {activity_id}")
            return
            
        time_data = streams['time']['data']
        
        # Helper to insert
        def process_stream(key, table_name):
            if key in streams:
                data = streams[key]['data']
                # Zip with time
                rows = list(zip([activity_id]*len(time_data), time_data, data))
                self.con.executemany(f"INSERT INTO stream_{table_name} VALUES (?, ?, ?)", rows)
                print(f"   Saved {len(rows)} points to stream_{table_name}")

        process_stream('heartrate', 'heartrate')
        process_stream('velocity_smooth', 'velocity')
        process_stream('cadence', 'cadence')
        process_stream('watts', 'watts')
        process_stream('temp', 'temperature')
        process_stream('altitude', 'altitude')
        process_stream('moving', 'moving')

    def activity_exists(self, activity_id):
        res = self.con.execute("SELECT 1 FROM dim_activity WHERE activity_id = ?", [activity_id]).fetchone()
        return res is not None

    def activity_has_streams(self, activity_id):
        # Check if we have data in key stream tables (e.g. heartrate or velocity)
        # Note: some activities might justifiably have no streams (manual entry), 
        # but this checks if we attempted ingestion.
        # A simple check is to see if there are any rows in any stream table for this id.
        # We'll check velocity since it's most common.
        res = self.con.execute("SELECT 1 FROM stream_velocity WHERE activity_id = ? LIMIT 1", [activity_id]).fetchone()
        if res: return True
        # Check heartrate
        res = self.con.execute("SELECT 1 FROM stream_heartrate WHERE activity_id = ? LIMIT 1", [activity_id]).fetchone()
        return res is not None

def get_activities_missing_streams(manager, db, access_token):
    """
    Generator that yields activities that are in DB but missing streams.
    This effectively re-scans history to backfill streams.
    """
    page = 1
    per_page = 50 
    
    while True:
        try:
            activities = manager.fetch_activities(access_token, count=per_page, page=page)
        except Exception as e:
            # reuse auth logic if needed/abstract it, but simple catch for now
            if "Unauthorized" in str(e):
                 # ... (same auth logic) ...
                 print(f"Unauthorized error on page {page}. Attempting to re-authenticate...")
                 new_token = manager.run_oauth_flow()
                 if new_token:
                    access_token = new_token
                    continue
            print(f"Error fetching page {page}: {e}")
            break
            
        if not activities:
            break
            
        for activity in activities:
            # If (New Activity) OR (Existing Activity AND Missing Streams)
            exists = db.activity_exists(activity['id'])
            if not exists:
                yield activity, access_token, "new"
            elif not db.activity_has_streams(activity['id']):
                # Only yield if it's a type of activity that should have streams (e.g. not manual weight training if no device)
                # But for now, we just try fetching.
                yield activity, access_token, "missing_streams"
        
        # We must ignore the "stop if all found" logic because we are backfilling streams
        # for potentially all historical activities.
        # However, Strava API limits are real.
        
        if len(activities) < per_page:
            break
            
        page += 1

def check_missing_activities(manager, db, access_token):
    """
    Scans Strava history and reports count of activities not in DB or missing streams.
    """
    print("Scanning for missing activities/streams...")
    count = 0
    for _, _, _ in get_activities_missing_streams(manager, db, access_token):
        count += 1
    print(f"Found {count} activities needing processing.")
    return count

def main():
    manager = StravaManager()
    db = StravaDB()
    
    # Authenticate
    access_token = manager.get_access_token()
    if not access_token:
        # Try flow
        access_token = manager.run_oauth_flow()
        if not access_token:
            print("Failed to authenticate.")
            return

    # 1. Fetch Athlete Profile (for dim_athlete)
    # We need a method in manager or just direct request
    print("Fetching athlete profile...")
    headers = {'Authorization': f'Bearer {access_token}'}
    try:
        athlete = requests.get("https://www.strava.com/api/v3/athlete", headers=headers).json()
        db.upsert_athlete(athlete)
        print(f"Upserted athlete: {athlete.get('firstname')} {athlete.get('lastname')}")
    except Exception as e:
        print(f"Error fetching athlete: {e}")

    # 2. Ingest Activities (New or Missing Streams)
    print("Starting ingestion of activities (New + Missing Streams)...")
    try:
        # We use the combined generator
        activity_gen = get_activities_missing_streams(manager, db, access_token)
        
        count = 0
        for activity, current_token, reason in activity_gen:
            # Update the access_token in main scope
            access_token = current_token
            
            print(f"Processing Activity ({reason}): {activity['name']} ({activity['start_date_local']})...")
            
            # Upsert Dimension (always safe to do)
            db.upsert_activity(activity)
            
            # Fetch Streams
            try:
                streams = manager.fetch_activity_streams(access_token, activity['id'], keys=STREAMS_TO_FETCH)
                if streams:
                    db.insert_streams(activity['id'], streams)
                else:
                    print("   No streams found.")
            except Exception as e:
                # If we get Unauthorized here, it implies the token expired betweeen fetching activity and streams
                if "Unauthorized" in str(e):
                    print("   Unauthorized during stream fetch. Refreshing token...")
                    access_token = manager.run_oauth_flow()
                    if access_token:
                        try:
                            streams = manager.fetch_activity_streams(access_token, activity['id'], keys=STREAMS_TO_FETCH)
                            if streams:
                                db.insert_streams(activity['id'], streams)
                            else:
                                print("   No streams found (after refresh).")
                        except Exception as inner_e:
                            print(f"   Error processing streams (after refresh): {inner_e}")
                else:
                    print(f"   Error processing streams: {e}")
            
            count += 1
            
        print(f"Ingestion complete. Processed {count} activities.")

    except Exception as e:
        print(f"Error in batch process: {e}")

if __name__ == "__main__":
    main()
