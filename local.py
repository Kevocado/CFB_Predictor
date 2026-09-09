import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

env_path = project_root / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.startswith("CFBD_API_KEY="):
                key = line.strip().split("=", 1)[1].strip("'\"")
                if not key.startswith("Bearer "):
                    key = f"Bearer {key}"
                os.environ["CFBD_API_KEY"] = key

from cfb_predictor.api.routes import get_player_props, CURRENT_SEASON

def test_local():
    print(f"Fetching local player props for {CURRENT_SEASON} Week 1...")
    props = get_player_props(season=CURRENT_SEASON, week=1)
    
    if not props:
        print("No props returned.")
        return

    print(f"\nRetrieved {len(props)} total player predictions:")
    print("-" * 50)
    
    # Group and print by team
    by_team = {}
    for p in props:
        by_team.setdefault(p["recent_team"], []).append(p)
        
    for team, players in sorted(by_team.items()):
        print(f"\nTeam: {team} ({len(players)} players)")
        for p in sorted(players, key=lambda x: x["player_name"]):
            yards = p.get("receiving_yards", "N/A")
            print(f"  - {p['player_name']} ({p['position']}) | Proj Rec Yds: {yards}")

if __name__ == "__main__":
    test_local()