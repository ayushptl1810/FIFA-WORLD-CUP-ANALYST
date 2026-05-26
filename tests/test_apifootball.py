import os
import requests
from dotenv import load_dotenv
load_dotenv()

def try_api_football():
    print("\n" + "="*60)
    print("DEMO: API-Football (dashboard.api-football.com)")
    print("="*60)

    api_key = os.getenv("APIFOOTBALL_KEY")
    base_url = os.getenv("APIFOOTBALL_ENDPOINT")
    
    headers = {
        "x-apisports-key": api_key
    }

    # Query /status endpoint to check key validity & daily limits
    print("\n[1] Checking API key status & daily limit...")
    try:
        status_url = f"{base_url}/status"
        response = requests.get(status_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            errors = data.get("errors", [])
            
            if errors:
                print(f"❌ API Error: {errors}")
                return
                
            response_data = data.get("response", {})
            account = response_data.get("account", {})
            requests_limits = response_data.get("requests", {})
            
            print(f"✅ Connection successful!")
            print(f"   Subscriber: {account.get('firstname')} {account.get('lastname')} ({account.get('email')})")
            print(f"   Subscription Plan: {account.get('plan')}")
            print(f"   Daily usage: {requests_limits.get('current')} / {requests_limits.get('limit_day')} requests used today.")
        else:
            print(f"❌ Failed to connect. HTTP Status Code: {response.status_code}")
            print(response.text)
            return
    except Exception as e:
        print(f"❌ Network connection error: {e}")
        return

    print("\n[2] Querying World Cup league details from the API...")
    try:
        leagues_url = f"{base_url}/leagues"
        params = {"search": "World Cup"}
        
        response = requests.get(leagues_url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            leagues = data.get("response", [])
            
            print(f"\nFound {len(leagues)} matches for 'World Cup' inside API-Football:")
            for item in leagues[:6]:
                league_info = item.get("league", {})
                country_info = item.get("country", {})
                print(f"   • League ID: {league_info.get('id'):<5} | Name: {league_info.get('name'):<22} | Country: {country_info.get('name')}")
        else:
            print(f"❌ Failed to fetch leagues data. HTTP Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error searching leagues: {e}")

if __name__ == "__main__":
    try_api_football()
