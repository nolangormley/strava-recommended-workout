# Strava Workout Recommender

This project fetches your Strava activities, stores them in a local DuckDB database, analyzes your Training Stress Balance (Fitness/Fatigue/Form), and recommends a specific workout from the Coros training library.

## Setup

1.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    playwright install chromium
    ```

2.  **Configuration**:
    Create a `.env` file with your Strava API credentials:
    ```env
    STRAVA_CLIENT_ID=your_id
    STRAVA_CLIENT_SECRET=your_secret
    STRAVA_REFRESH_TOKEN=your_token (optional, script handles auth)
    ```

## Usage

**Run everything:**
```bash
python run.py
```

This will:
1.  Fetch latest Strava activities.
2.  Calculate training metrics (TRIMP, Efficiency Factor).
3.  Display your current Training Load status.
4.  Recommend a specific workout.

**Update Workout Library:**
To scrape new workouts from Coros (Run occasionally):
```bash
python run.py --update-workouts
```

## Data Warehouse
All data is stored in `data/strava_warehouse.duckdb`. You can query it using DuckDB CLI or DBeaver.

## Key Files
*   `run.py`: Main entry point.
*   `ingest_strava_duckdb.py`: Fetches Strava data.
*   `analyze_effectiveness.py`: Calculates effectiveness metrics.
*   `analyze_training_load.py`: Calculates TSB and recommends workouts.
*   `ingest_workouts.py`: Scrapes Coros website for workout plans.

## Running the Web API

The project includes a FastAPI backend that serves the data. You can start the server locally or via Docker.

**Start Locally using Uvicorn:**
```bash
python -m uvicorn src.api.main:app --reload
```

**Start using Docker Compose:**
```bash
docker-compose up --build
```

The API will be available at `http://127.0.0.1:8000`.
