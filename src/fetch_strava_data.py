import os
import sys
import webbrowser
import requests
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv, set_key

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

CLIENT_ID = os.getenv('STRAVA_CLIENT_ID')
CLIENT_SECRET = os.getenv('STRAVA_CLIENT_SECRET')

# Configuration
REDIRECT_PORT = 5000
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"
SCOPES = "activity:read_all,activity:read"

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

class StravaManager:
    def __init__(self):
        if not CLIENT_ID or not CLIENT_SECRET:
            print("Error: STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET must be set in .env file.")
            sys.exit(1)
            
    def save_refresh_token(self, refresh_token):
        """Saves the refresh token to the .env file."""
        set_key(env_path, "STRAVA_REFRESH_TOKEN", refresh_token)
        # Reload environment to ensure we have the latest
        load_dotenv(env_path, override=True)
        print(f"Saved new refresh token to {env_path}")

    def run_oauth_flow(self):
        """Initiates the OAuth 2.0 flow to get a new refresh token."""
        print("Initiating OAuth flow...")
        
        # 1. Authorize URL
        auth_url = (
            f"https://www.strava.com/oauth/authorize?"
            f"client_id={CLIENT_ID}&"
            f"redirect_uri={REDIRECT_URI}&"
            f"response_type=code&"
            f"scope={SCOPES}"
        )
        
        print(f"\nPlease open the following URL in your browser if it doesn't open automatically:")
        print(f"{auth_url}\n")
        try:
            webbrowser.open(auth_url)
        except:
            print("Could not open browser automatically.")
        
        # 2. Start local server to listen for callback
        print(f"Listening on port {REDIRECT_PORT} waiting for callback...")
        server = HTTPServer(('localhost', REDIRECT_PORT), OAuthHandler)
        server.auth_code = None
        
        # Handle a single request then close
        server.handle_request()
        
        if not server.auth_code:
            print("Failed to get authorization code.")
            return None
            
        print("Authorization code received.")
        return self.exchange_code_for_token(server.auth_code)

    def exchange_code_for_token(self, code):
        """Exchanges the authorization code for access and refresh tokens."""
        url = "https://www.strava.com/oauth/token"
        payload = {
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code'
        }
        
        response = requests.post(url, data=payload)
        response.raise_for_status()
        data = response.json()
        
        self.save_refresh_token(data['refresh_token'])
        return data['access_token']

    def get_access_token(self):
        """Gets a valid access token, refreshing if necessary."""
        refresh_token = os.getenv('STRAVA_REFRESH_TOKEN')
        
        if not refresh_token:
            print("No refresh token found. Starting initial authorization...")
            return self.run_oauth_flow()

        url = "https://www.strava.com/oauth/token"
        payload = {
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token'
        }
        
        try:
            print("Refreshing access token...")
            response = requests.post(url, data=payload)
            
            # If refresh token is invalid (e.g. revoked or expired), try OAuth flow
            if response.status_code == 401 or response.status_code == 400:
                print("Refresh token invalid or expired. Re-authorizing...")
                return self.run_oauth_flow()
                
            response.raise_for_status()
            data = response.json()
            
            # Save new refresh token if it changed
            if data.get('refresh_token') and data['refresh_token'] != refresh_token:
                self.save_refresh_token(data['refresh_token'])
                
            return data['access_token']
            
        except requests.exceptions.RequestException as e:
            print(f"Error refreshing token: {e}")
            return None

    def fetch_activities(self, access_token, count=5, page=1, after=None):
        """Fetches recent activities."""
        url = "https://www.strava.com/api/v3/athlete/activities"
        headers = {'Authorization': f'Bearer {access_token}'}
        params = {'per_page': count, 'page': page}
        if after:
            params['after'] = after
        
        print(f"Fetching activities (Page {page}, Count {count}, After {after})...")
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 401:
            raise Exception("Unauthorized")
            
        response.raise_for_status()
        return response.json()

    def fetch_activity_streams(self, access_token, activity_id, keys=["time", "heartrate", "velocity_smooth"]):
        """Fetches time-series data (streams) for a specific activity."""
        url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
        headers = {'Authorization': f'Bearer {access_token}'}
        params = {
            'keys': ",".join(keys),
            'key_by_type': 'true'
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 401:
             raise Exception("Unauthorized")
        
        if response.status_code == 404:
            return None
            
        response.raise_for_status()
        return response.json()

def main():
    manager = StravaManager()
    
    # Get Token
    access_token = manager.get_access_token()
    if not access_token:
        print("Failed to obtain access token.")
        return

    print(f"Access token secured: {access_token[:10]}...")

    # Fetch Data
    try:
        activities = manager.fetch_activities(access_token)
    except Exception as e:
        if "Unauthorized" in str(e):
            print("\n[!] Access token invalid or missing scopes.")
            print("    Triggering re-authorization flow...")
            access_token = manager.run_oauth_flow()
            if access_token:
                activities = manager.fetch_activities(access_token)
            else:
                print("Re-authorization failed.")
                return
        else:
            print(f"An error occurred fetching activities: {e}")
            return

    print(f"\nSuccessfully fetched {len(activities)} activities:")
    print("-" * 50)
    for activity in activities:
        # Convert meters to km
        distance_km = activity['distance'] / 1000
        print(f"{activity['start_date_local'][:10]} | {activity['name']} (ID: {activity['id']})")
        print(f"   Type: {activity['type']} | Distance: {distance_km:.2f} km")
        
        # Fetch detailed streams
        try:
            streams = manager.fetch_activity_streams(access_token, activity['id'])
            if streams:
                if 'heartrate' in streams:
                    hr_data = streams['heartrate']['data']
                    avg_hr = sum(hr_data) // len(hr_data) if hr_data else 0
                    print(f"   [+] Heart Rate: {len(hr_data)} points (Avg: {avg_hr} bpm)")
                    # print(f"       Sample: {hr_data[:10]}...")
                else:
                    print("   [-] No Heart Rate data available")

                if 'velocity_smooth' in streams:
                    vel_data = streams['velocity_smooth']['data']
                    # Calculate average pace (min/km) from avg velocity (m/s) if possible
                    # avg_vel = sum(vel_data) / len(vel_data) if vel_data else 0
                    print(f"   [+] Pace/Velocity: {len(vel_data)} data points")
                else:
                    print("   [-] No Pace/Velocity data available")
        except Exception as e:
            print(f"   [!] Could not fetch streams: {e}")
            
        print("-" * 50)

if __name__ == "__main__":
    main()
