from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata
import requests


class SkillCorner:

    def load(self, match_id):

        def compute_age(birthday_str):
            if not birthday_str:
                return None
            try:
                birthdate = datetime.strptime(birthday_str, "%Y-%m-%d")
                return (datetime.today() - birthdate).days // 365
            except ValueError:
                return None  # or a default age if appropriate

        def remove_accents(text):
            # Normalize to decomposed form, then remove non-ASCII combining marks
            return "".join(
                c
                for c in unicodedata.normalize("NFD", text)
                if unicodedata.category(c) != "Mn"
            )

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        meta_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_match.json"

        # Save meta JSON
        meta_file = data_dir / f"{match_id}_meta.json"
        if not meta_file.exists():
            with open(meta_file, "w", encoding="utf-8") as f:
                f.write(requests.get(meta_url).text)

        metadata_json = pd.read_json(meta_file, lines=True)

        # Team info
        home_team = metadata_json["home_team"][0]
        away_team = metadata_json["away_team"][0]

        team = {home_team["id"]: home_team["name"], away_team["id"]: away_team["name"]}

        match_date = metadata_json["date_time"][0].date()
        match_name = f"{home_team['name']}_vs_{away_team['name']}_{match_date}"

        metadata_dic = {
            "match_date": match_date,
            "home_team": home_team["name"],
            "away_team": away_team["name"],
            "home_team_jersey_color": metadata_json["home_team_kit"][0]["jersey_color"],
            "home_team_number_color": metadata_json["home_team_kit"][0]["number_color"],
            "away_team_jersey_color": metadata_json["away_team_kit"][0]["jersey_color"],
            "away_team_number_color": metadata_json["away_team_kit"][0]["number_color"],
            "match_name": match_name,
        }
        metadata_df = pd.DataFrame([metadata_dic])

        # Lineup DataFrame
        players = [
            {
                "player_id": p["id"],
                "short_name": remove_accents(p["short_name"]),
                "team_name": team[p["team_id"]],
                "jersey_number": p["number"],
                "skillcorner_position": p["player_role"]["acronym"],
                "playing_time_ip": p["playing_time"]["total"]["minutes_tip"],
                "playing_time_op": p["playing_time"]["total"]["minutes_otip"],
                "start_time": p["start_time"],
                "end_time": p["end_time"],
                "age": compute_age(p["birthday"]),
            }
            for p in metadata_json["players"][0]
            if p["start_time"] is not None
        ]
        lineup_df = pd.DataFrame(players)

        # Tracking preprocessing
        home_side_map = {"left_to_right": 1, "right_to_left": -1}
        player2position = lineup_df.set_index("player_id")[
            "skillcorner_position"
        ].to_dict()
        player2team = lineup_df.set_index("player_id")["team_name"].to_dict()

        tracking_url = f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"

        # Save tracking JSON
        tracking_file = data_dir / f"{match_id}_tracking.jsonl"

        if not tracking_file.exists():
            with open(tracking_file, "w", encoding="utf-8") as f:
                f.write(requests.get(tracking_url).text)

        # Load data
        tracking_json = pd.read_json(tracking_file, lines=True)

        # Vectorized tracking DataFrame
        tracking_records = []
        gk_records = []
        for _, row in tracking_json.iterrows():
            possession = row["possession"]["group"]
            if possession not in ["home team", "away team"]:
                continue
            period = int(row["period"])
            home_team_side = home_side_map.get(
                metadata_json["home_team_side"][0][period - 1], 1
            )
            away_team_side = -home_team_side
            for p in row["player_data"]:
                team_name = player2team[p["player_id"]]
                side = (
                    home_team_side if team_name == home_team["name"] else away_team_side
                )
                possession_label = (
                    "In"
                    if (possession == "home team") == (team_name == home_team["name"])
                    else "Out"
                )
                record = {
                    "period": period,
                    "frame": row["frame"],
                    "timestamp": row["timestamp"],
                    "player_id": p["player_id"],
                    "team_name": team_name,
                    "possession": possession_label,
                    "x": p["x"] * side,
                    "y": p["y"] * side,
                }
                if player2position[p["player_id"]] == "GK":
                    gk_records.append(record)
                else:
                    tracking_records.append(record)
        tracking_df = pd.DataFrame(tracking_records)
        gk_df = pd.DataFrame(gk_records)
        # Parse events
        event_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_dynamic_events.csv"

        # Save meta JSON
        event_file = data_dir / f"{match_id}_dynamic_events.csv"
        if not event_file.exists():
            with open(event_file, "w", encoding="utf-8") as f:
                f.write(requests.get(event_url).text)

        event_df = pd.read_csv(event_file)

        


        # Parse Phases
        phases_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_phases_of_play.csv"

        # Save meta JSON
        phases_file = data_dir / f"{match_id}_phases_of_play.csv"
        if not phases_file.exists():
            with open(phases_file, "w", encoding="utf-8") as f:
                f.write(requests.get(phases_url).text)

        phases_df = pd.read_csv(phases_file)

        intervals = pd.IntervalIndex.from_arrays(
            phases_df["frame_start"], phases_df["frame_end"], closed="left"
        )

        # Map each frame to its corresponding phase (index in phases_df)
        phase_idx = intervals.get_indexer(tracking_df["frame"])

        # Initialize 'phase' column
        tracking_df["phase"] = np.nan

        # Identify valid matches
        valid = phase_idx != -1

        # Assign phases based on possession
        tracking_df.loc[valid, "phase"] = np.where(
            tracking_df.loc[valid, "possession"] == "In",
            phases_df.iloc[phase_idx[valid]]["team_in_possession_phase_type"].values,
            phases_df.iloc[phase_idx[valid]][
                "team_out_of_possession_phase_type"
            ].values,
        )

        # Merge lineup info directly into tracking_df
        tracking_df = tracking_df.merge(
            lineup_df[["player_id", "start_time", "skillcorner_position"]],
            on="player_id",
            how="left",
        )
        tracking_df["match_id"] = match_id


        print(f"{match_name} parsed ...")
        return metadata_df, lineup_df, tracking_df, event_df, phases_df, gk_df
