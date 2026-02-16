from src.fetch_strava_data import StravaManager
from src.ingest_strava_duckdb import main as ingest
from src.ingest_workouts import scrape_workouts_dom, ingest_to_duckdb
from src.analyze_effectiveness import analyze_effectiveness
from src.analyze_training_load import calculate_training_load
import sys

def main():
    print("=== Strava Workout Recommender ===")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--update-workouts':
        print("\n[Step 0] Updating Coros Workout Library (this takes a minute)...")
        workouts = scrape_workouts_dom()
        if workouts:
            ingest_to_duckdb(workouts)
            
    print("\n[Step 1] Ingesting New Strava Activities...")
    # Call the main function of ingest_strava_duckdb
    # Note: ingest_strava_duckdb.main() calls sys.exit() on error or clean run? 
    # Checking source: It prints and returns. Should be safe.
    try:
        ingest()
    except Exception as e:
        print(f"Ingestion Error (non-fatal): {e}")

    print("\n[Step 2] Analyzing Effectiveness (TRIMP/EF)...")
    analyze_effectiveness()
    
    print("\n[Step 3] Calculating Training Load & Recommendation...")
    calculate_training_load()

if __name__ == "__main__":
    main()
