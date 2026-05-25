import json, requests

# Fetch directly — no clone needed
BASE = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026"

fixtures_raw = requests.get(f"{BASE}/worldcup.json").json()
teams_raw = requests.get(f"{BASE}/worldcup.teams.json").json()

# The fixtures file has a "matches" key which is a flat list of all matches
fixtures = []
for match in fixtures_raw.get("matches", []):
    fixtures.append({
        "round": match.get("round"),
        "date": match.get("date"),
        "time": match.get("time"),
        "team1": match.get("team1"),
        "team2": match.get("team2"),
        "group": match.get("group"),          # None for knockout rounds
        "ground": match.get("ground"),
        "score1": match.get("score1"),         # None until played
        "score2": match.get("score2"),
    })

# teams_raw is a flat list of team dictionaries
teams = [t["name"] for t in teams_raw]

print(f"Fixtures: {len(fixtures)}")
print(f"Teams: {len(teams)}")
print(f"\nSample fixture:\n{fixtures[0]}")
print(f"\nSample teams: {teams[:5]}")