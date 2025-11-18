"""
Position module.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Dict, List, Sequence, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from PyPDF2 import PdfMerger
import seaborn as sns

import plotly.graph_objects as go
from sklearn.manifold import MDS
import plotly.express as px
from IPython.display import display
import mplcursors
import enum
import ast
from datetime import datetime

# -----------------------------
# Constants
# -----------------------------
POSITIONS_ORDER = [
    "LB",
    "LCB",
    "CB",
    "RCB",
    "RB",
    "LWB",
    "LDM",
    "CDM",
    "RDM",
    "RWB",
    "LM",
    "LCM",
    "CM",
    "RCM",
    "RM",
    "LWF",
    "LAM",
    "CAM",
    "RAM",
    "RWF",
    "LF",
    "LCF",
    "CF",
    "RCF",
    "RF",
]

SKILLCORNER_POSITIONS = [
    "LF",
    "CF",
    "RF",
    "LW",
    "AM",
    "RW",
    "LM",
    "CM",
    "RM",
    "LWB",
    "LDM",
    "DM",
    "RDM",
    "RWB",
    "LB",
    "LCB",
    "CB",
    "RCB",
    "RB",
]

five_x_five_positions = [
    "LF",
    "",
    "CF",
    "",
    "RF",
    "LW",
    "",
    "AM",
    "",
    "RW",
    "",
    "LM",
    "CM",
    "RM",
    "",
    "LWB",
    "LDM",
    "DM",
    "RDM",
    "RWB",
    "LB",
    "LCB",
    "CB",
    "RCB",
    "RB",
]

# quick index map for earlier helper
_SKILLCORNER_IDX = {p: i for i, p in enumerate(five_x_five_positions) if p}

# color map (white -> red) and default sizes
CMAP = mcolors.LinearSegmentedColormap.from_list("white_red", [(0, "white"), (1, "#C55D57")])
CELL_FONTSIZE = 18
TITLE_FONTSIZE = 22
SUBTITLE_FONTSIZE = 18


class Colors(enum.Enum):
    """Enum for the colors"""

    WHITE = "white"
    BLUE = "#215CAF"
    LIGHTBLUE = "#7A9DCF"
    RED = "#B7352D"
    LIGHTRED = "#D48681"

    GREY = "#E2E2E2"
    PITCH_GREY = "#A9A9A9"
    PITCH_MARKINGS = "#E2E2E2"

    GREEN = "#627313"
    LIGHTGREEN = "#C0C7A1"
    ORANGE = "#8E6713"
    LIGHTORANGE = "#D2C2A1"

    BREAK = "#8C8C8C"
    GREY_120 = "#575757"

# -----------------------------
# Utility functions
# -----------------------------

def return_percentage(ratio: float) -> str:
    """Return short string percentage similar to original behaviour."""
    percentage = round(float(ratio) * 100, 1)
    return str(int(percentage)) if percentage.is_integer() else f"{percentage}"


def skillcorner_position_to_idx(position: str) -> int:
    return _SKILLCORNER_IDX[position]


def hellinger_distance_matrix(dict_vectors: Sequence[Dict[str, float]],
                              index_order: Sequence[str] = POSITIONS_ORDER) -> np.ndarray:
    """Compute Hellinger distance matrix efficiently.

    Converts list-of-dicts into a (n, m) array aligned with index_order, then
    uses the identity H(p,q) = sqrt(1 - sum(sqrt(p * q))).
    """
    n = len(dict_vectors)
    m = len(index_order)
    arr = np.zeros((n, m), dtype=float)

    for i, d in enumerate(dict_vectors):
        # faster than repeated .get calls in Python loop is still acceptable here
        arr[i, :] = [d.get(k, 0.0) for k in index_order]

    # ensure numerical stability
    arr = np.clip(arr, 0.0, 1.0)

    sqrt_arr = np.sqrt(arr)
    # similarity matrix of Bhattacharyya coefficients
    S = sqrt_arr @ sqrt_arr.T
    # clip to [0,1]
    S = np.clip(S, 0.0, 1.0)
    H = np.sqrt(np.maximum(0.0, 1.0 - S))
    return H

# Helper alias for backward compatibility
hellinger_distance = lambda a, b: float(np.sqrt(np.maximum(0.0, 1.0 - np.sum(np.sqrt(np.array([a.get(k,0) for k in POSITIONS_ORDER]) * np.array([b.get(k,0) for k in POSITIONS_ORDER]))))))

# -----------------------------
# Plotting helpers
# -----------------------------

def _draw_position_grid(ax: matplotlib.axes.Axes,
                        position_profile: Dict[str, float],
                        cmap=CMAP,
                        cell_fontsize: int = CELL_FONTSIZE,
                        title: Optional[str] = None,
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None) -> None:
    """Draw a 5x5 grid of positions on 'ax' using values from position_profile.

    position_profile must contain entries for all values in POSITIONS_ORDER.
    """
    values = np.array([position_profile.get(p, 0.0) for p in POSITIONS_ORDER])
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(5):
        for j in range(5):
            pos = POSITIONS_ORDER[i * 5 + j]
            val = position_profile.get(pos, 0.0)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=cmap(norm(val))))
            ax.text(j, i, return_percentage(val), va="center", ha="center", color="black", fontsize=cell_fontsize)

    # grid lines
    for k in range(6):
        ax.plot([k - 0.5, k - 0.5], [-0.5, 4.5], color="black", linewidth=1)
        ax.plot([-0.5, 4.5], [k - 0.5, k - 0.5], color="black", linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=SUBTITLE_FONTSIZE)


def convert_to_mmss(time_str):
    # Parse the input time (HH:MM:SS)
    dt = datetime.strptime(time_str, "%H:%M:%S")
    
    # Convert hours and minutes to total minutes
    total_minutes = dt.hour * 60 + dt.minute
    
    # Use only whole seconds
    seconds = dt.second
    
    return f"{total_minutes:02d}:{seconds:02d}"

# -----------------------------
# Main Position class
# -----------------------------
class Position:
    def __init__(self, position_maps_dir: Path):
        self.position_maps_dir = Path(position_maps_dir)
        self.position_maps_dir.mkdir(parents=True, exist_ok=True)
        self.position_map_results: List[dict] = []
        self.skillcorner_position_profile: Dict[str, Dict[str, float]] = {}
        self.position_map_results_df: Optional[pd.DataFrame] = None

    # -------------------------
    # Vectorized dynamic position assignment
    # -------------------------
    @staticmethod
    def _vh2position(v_level: str, h_level: str) -> str:
        mapping = {
            "LD": "LWB",
            "LCD": "LDM",
            "CD": "CDM",
            "RCD": "RDM",
            "RD": "RWB",
            "LA": "LWF",
            "LCA": "LAM",
            "CA": "CAM",
            "RCA": "RAM",
            "RA": "RWF",
        }
        pos = f"{h_level}{v_level}"
        return mapping.get(pos, pos)

    @classmethod
    def assign_dynamic_positions(cls, tracking_df: pd.DataFrame) -> pd.DataFrame:
        """Assigns a discrete positional label based on player x/y within each (frame, team).

        This implementation avoids explicit Python loops by computing relative
        positions within each (frame, team) group and then binning with numpy.
        """
        df = tracking_df.copy()
        # ensure floats
        df["x"] = df["x"].astype(float)
        df["y"] = df["y"].astype(float)

        groups = df.groupby(["frame", "team_name"])

        # compute group-wise min/max
        df["min_x"] = groups["x"].transform("min")
        df["max_x"] = groups["x"].transform("max")
        df["min_y"] = groups["y"].transform("min")
        df["max_y"] = groups["y"].transform("max")

        # vertical weights and cumulative boundaries (normalized)
        v_weights = np.array([0.5, 1.0, 1.0, 1.0, 0.5])
        v_cum = np.cumsum(v_weights)
        v_cum = v_cum / v_cum[-1]  # 5 boundaries in (0,1]
        # include 0 at left
        v_edges = np.concatenate(([0.0], v_cum))  # length 6

        # compute relative position in group and digitize
        eps = 1e-9
        rel_x = (df["x"] - df["min_x"]) / (df["max_x"] - df["min_x"] + eps)
        # map to bins 0..4 -> labels B,D,M,A,F
        v_bin_idx = np.minimum(np.searchsorted(v_edges, rel_x, side="right") - 1, 4)
        v_labels = np.array(["B", "D", "M", "A", "F"])
        df["v_level"] = v_labels[v_bin_idx]

        # horizontal equally spaced bins using group min/max
        rel_y = (df["y"] - df["min_y"]) / (df["max_y"] - df["min_y"] + eps)
        # 5 bins edges: 0, 0.2, 0.4, 0.6, 0.8, 1.0
        h_edges = np.linspace(0.0, 1.0, 6)
        h_bin_idx = np.minimum(np.searchsorted(h_edges, rel_y, side="right") - 1, 4)
        h_labels = np.array(["R", "RC", "C", "LC", "L"])
        df["h_level"] = h_labels[h_bin_idx]

        # combine to position string and map with helper
        df["position"] = [cls._vh2position(v, h) for v, h in zip(df["v_level"], df["h_level"])]

        # drop helper cols
        df = df.drop(columns=["min_x", "max_x", "min_y", "max_y"])
        return df

    # -------------------------
    # Template creation (vectorized counting)
    # -------------------------
    def create_template(self, position_dfs: pd.DataFrame) -> None:
        """Create templates for each skillcorner position and save to templates.pdf.

        Expects position_dfs to have columns ['skillcorner_position', 'position', 'match_id', 'player_id'].
        """
        # precompute profiles using groupby (vectorized)
        grouped = (
            position_dfs.groupby(["skillcorner_position", "position"]).size().unstack(fill_value=0)
        )

        skillcorner_counts = (
            position_dfs[['match_id', 'player_id', 'skillcorner_position']]
            .drop_duplicates()
            ['skillcorner_position']
            .value_counts()
        )

        skillcorner_counts_dict = skillcorner_counts.to_dict()

        # normalize to proportions per skillcorner_position
        grouped = grouped.reindex(columns=POSITIONS_ORDER, fill_value=0)
        proportions = grouped.div(grouped.sum(axis=1).replace(0, 1), axis=0)

        self.skillcorner_position_profile = {k: proportions.loc[k].to_dict() for k in proportions.index}

        # plotting: create a single multi-panel figure with all templates
        n = len(SKILLCORNER_POSITIONS)
        rows, cols = 5, 5
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()

        for _, sc_pos in enumerate(SKILLCORNER_POSITIONS):
            i = skillcorner_position_to_idx(sc_pos)
            ax = axes[i]
            if sc_pos == "CM":
                fig.delaxes(axes[i])
                continue
            profile = self.skillcorner_position_profile.get(sc_pos)
            _draw_position_grid(ax, profile, title=f"{sc_pos} - {skillcorner_counts_dict[sc_pos]}")

        empty_indices = [i for i, p in enumerate(five_x_five_positions) if p == ""]
        for i in empty_indices:
            fig.delaxes(axes[i])

        fig.tight_layout()
        out = self.position_maps_dir / "templates.pdf"
        with PdfPages(out) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # -------------------------
    # Plotting maps per player (cleaner version)
    # -------------------------
    def plot_maps(self, metadata_df: pd.DataFrame, lineup_df: pd.DataFrame, tracking_df: pd.DataFrame, merger: PdfMerger) -> PdfMerger:
        """Create positional maps per player and append to provided PdfMerger.

        This implementation reduces repeated plotting code by using _draw_position_grid.
        """
        player2position = lineup_df.set_index("player_id")["skillcorner_position"].to_dict()
        player2team = lineup_df.set_index("player_id")["team_name"].to_dict()
        match_date = metadata_df["match_date"].iloc[0]

        # iterate players
        for _, player in lineup_df.iterrows():
            pid = player["player_id"]
            if player2position.get(pid) == "GK":
                continue

            player_df = tracking_df[tracking_df["player_id"] == pid]

            if player_df.empty:
                continue

            team_name = player2team.get(pid, "")
            player_name = player["short_name"]

            phases = [("All", None), ("In", "in"), ("Out", "out")]

            # create figure with rows = 1 + number of subphases
            sub_in = ["build_up", "create", "finish", "direct", "quick_break", "set_play", "transition", "chaotic"]
            sub_out = ["low_block", "mid_block", "high_block", "defending_direct", "defending_quick_break", "defending_set_play", "defending_transition", "chaotic"]
            nrows = 1 + len(sub_in)
            ncols = 3
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), constrained_layout=True)

            # helper to get phase-specific df
            def get_phase_df(p):
                if p == "All":
                    return player_df
                return player_df[player_df["possession"] == p]

            for col_idx, (phase, key) in enumerate(phases):
                ax_main = axes[0, col_idx]
                df_phase = get_phase_df(phase)
                pos_profile = df_phase.position.value_counts(normalize=True).reindex(POSITIONS_ORDER, fill_value=0).to_dict()
                _draw_position_grid(ax_main, pos_profile, title=f"{phase} ({round((player['playing_time_ip']+player['playing_time_op']) if phase=='All' else (player['playing_time_ip'] if phase=='In' else player['playing_time_op']),2)} mins)")

                if phase == "All":
                    our_prediction = min(
                        self.skillcorner_position_profile,
                        key=lambda skillcorner_position: hellinger_distance(
                            pos_profile, self.skillcorner_position_profile[skillcorner_position]
                        ),
                    )
                    
                    self.position_map_results.append({
                        "match_date": match_date,
                        "team_name": team_name,
                        "player_name": player_name,
                        "skillcorner_position": player2position[pid],
                        "our_prediction": our_prediction,
                        "position_profile": pos_profile,
                    })

                # if subphases
                sub_list = sub_in if phase == "In" else (sub_out if phase == "Out" else [])
                for ridx, sub in enumerate(sub_list, start=1):
                    ax_sub = axes[ridx, col_idx]
                    df_sub = player_df[(player_df["possession"] == phase) & (player_df["phase"] == sub)] if phase != "All" else pd.DataFrame(columns=player_df.columns)
                    profile_sub = df_sub.position.value_counts(normalize=True).reindex(POSITIONS_ORDER, fill_value=0).to_dict()
                    _draw_position_grid(ax_sub, profile_sub, title=sub.replace("_", " ").title())

            # remove extra axes rows first column as original behaviour
            for i in range(1, nrows):
                fig.delaxes(axes[i][0])

            # figure title
            start, end = player.get("start_time"), player.get("end_time")
            from_to = f"From {convert_to_mmss(start)}" if start and start != "00:00:00" else ""
            if end:
                from_to = f"{from_to + ' To ' if from_to else 'Until '}{convert_to_mmss(end)}"
            age_part = f" ({player.get('age')})" if player.get("age") else ""
            fig.suptitle(f"{player_name}{age_part}, {team_name}, {match_date}, skillcorner: {player2position.get(pid)}" + (f"\n{from_to}" if from_to else ""), fontsize=TITLE_FONTSIZE)

            # save to single PDF and append
            pdf_path = self.position_maps_dir / f"{player_name}_{player2position.get(pid)}_{match_date}.pdf"
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            merger.append(str(pdf_path))
            if pdf_path.exists():
                pdf_path.unlink()

        return merger

    # -------------------------
    # Confusion matrix and results saving
    # -------------------------
    def save_nearest_neighbor_confusion_matrix(self) -> None:
        results_df = pd.DataFrame(self.position_map_results)
        if results_df.empty:
            print("No results to save")
            return

        results_df["matched"] = results_df.apply(lambda r: r["our_prediction"] == r["skillcorner_position"], axis=1)

        results_df = results_df.sort_values(by="matched", ascending=True)

        total_cases = len(results_df)
        true_cases = results_df["matched"].sum()
        accuracy = true_cases / total_cases if total_cases > 0 else 0.0

        print(f"Number of total cases: {total_cases}")
        print(results_df["matched"].value_counts())
        print(f"Accuracy: {accuracy:.2%}")

        # save individual case pages and build confusion frequency matrix
        files: List[str] = []
        for _, row in results_df.iterrows():
            profile = row["position_profile"]
            skill_pos = row["skillcorner_position"]
            player_name = row["player_name"]
            our_pred = row["our_prediction"]
            match_date = row["match_date"]

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            _draw_position_grid(ax, profile, title=f"SkillCorner: {skill_pos}, Ours: {our_pred}", cell_fontsize=20)
            fig.text(0.5, 0.01, f"{player_name} on {match_date}", ha="center", fontsize=TITLE_FONTSIZE)
            fn = f"{skill_pos}_{player_name}_{match_date}.pdf"
            files.append(fn)
            with PdfPages(self.position_maps_dir / fn) as pdf:
                pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # merge and remove
        merger = PdfMerger()
        for f in files:
            merger.append(str(self.position_maps_dir / f))
        out = self.position_maps_dir / "cases.pdf"
        merger.write(str(out))
        merger.close()
        for f in files:
            (self.position_maps_dir / f).unlink(missing_ok=True)

        # frequency matrix
        frequency_matrix = pd.crosstab(results_df["skillcorner_position"], results_df["our_prediction"]).reindex(index=SKILLCORNER_POSITIONS, columns=SKILLCORNER_POSITIONS, fill_value=0)

        # plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = frequency_matrix == 0
        heatmap = sns.heatmap(frequency_matrix, annot=True, fmt="d", mask=mask, cmap="Reds", cbar_kws={"label": "Frequency", "shrink": 0.8, "pad": 0.02}, annot_kws={"size": 16})

        # Set the color bar label font size
        cbar = heatmap.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(20)

        # Increase the font size of the tick labels
        for label in cbar.ax.get_yticklabels():
            label.set_fontsize(15)  # Set the desired font size

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

        ax.set_xlabel("Nearest neighbor label", fontsize=20, labelpad=15)
        ax.set_ylabel("SkillCorner label", fontsize=20, labelpad=15)
        ax.set_aspect("equal")

        split_after_labels = ["RB", "RWB", "RM", "RF"]  # example group ends

        # Get index positions of those labels
        label_pos = {label: idx for idx, label in enumerate(SKILLCORNER_POSITIONS)}
        split_positions = [label_pos[label] + 1 for label in split_after_labels]

        # Draw thicker grid lines after the specified labels
        for pos in split_positions:
            ax.axvline(pos, color="black", linewidth=0.1)
            ax.axhline(pos, color="black", linewidth=0.1)

        fig.tight_layout()
        with PdfPages(self.position_maps_dir / "matching.pdf") as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        self.position_map_results_df = results_df

    # -------------------------
    # Embedding visualization (MDS)
    # -------------------------
    def create_interactive_embedding_visualization(self):
        if self.position_map_results_df is None or self.position_map_results_df.empty:
            print("No results to embed")
            return

        vectors = self.position_map_results_df["position_profile"].tolist()

        D = hellinger_distance_matrix(vectors)
        emb = MDS(n_components=2, dissimilarity="precomputed", random_state=42, metric=True)
        pts = emb.fit_transform(D)

        self.position_map_results_df["mds_embedding_x"] = pts[:, 0]
        self.position_map_results_df["mds_embedding_y"] = pts[:, 1]

        position_labels = self.position_map_results_df["skillcorner_position"].tolist()
        mask = ~self.position_map_results_df["mds_embedding_x"].isna()

        symbol_map = {
            "L": ["LB", "LWB", "LW", "LF"],
            "LC": ["LCB", "LDM", "LM"],
            "C": ["CB", "DM", "CM", "AM", "CF"],
            "RC": ["RCB", "RDM", "RM"],
            "R": ["RB", "RWB", "RM", "RW", "RF"],
        }
        symbol_lookup = {pos: sym for sym, group in symbol_map.items() for pos in group}

        BASE_MARKER_SIZE = 18

        def get_color(pos):
            if pos in ["LF", "CF", "RF"]:
                return Colors.RED.value
            elif pos in ["LW", "AM", "RW"]:
                return Colors.LIGHTRED.value
            elif pos in ["LWB", "LDM", "DM", "RDM", "RWB"]:
                return Colors.LIGHTBLUE.value
            elif "B" in pos:
                return Colors.BLUE.value
            else:
                return Colors.GREY.value

        traces = []

        # Main plot: letters
        for pos in np.unique(position_labels):
            df_pos = self.position_map_results_df[
                (self.position_map_results_df["skillcorner_position"] == pos) & mask
            ]
            
            x_coords = df_pos["mds_embedding_y"]
            y_coords = df_pos["mds_embedding_x"]

            hover_text = [
                f"Match: {df_pos.loc[idx, 'match_date']}<br>"
                f"Player: {df_pos.loc[idx, 'player_name']}<br>"
                f"Team: {df_pos.loc[idx, 'team_name']}<br>"
                f"SkillCorner: {df_pos.loc[idx, 'skillcorner_position']}<br>"
                f"Ours: {df_pos.loc[idx, 'our_prediction']}<br>"
                for idx in df_pos.index
            ]
            
            # Plot the actual points
            traces.append(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="text",
                    text=[symbol_lookup[pos]] * len(df_pos),
                    textfont=dict(size=BASE_MARKER_SIZE, color=get_color(pos)),
                    hovertext=hover_text,
                    hoverinfo="text",
                    showlegend=False
                )
            )

        # Add dummy points for legend with same symbol & color
        for pos in np.unique(position_labels):
            traces.append(
                go.Scatter(
                    x=[None],  # invisible point
                    y=[None],
                    mode="text",
                    text=[symbol_lookup[pos]],
                    textfont=dict(size=BASE_MARKER_SIZE, color=get_color(pos)),
                    name=pos,
                    showlegend=True
                )
            )


        # Get axis ranges to invert
        x_vals = self.position_map_results_df["mds_embedding_y"].dropna()
        y_vals = self.position_map_results_df["mds_embedding_x"].dropna()

        layout = go.Layout(
            title=dict(text="Interactive Player Position Map", font=dict(size=20)),
            xaxis=dict(
                title=dict(text="Component 2", font=dict(size=18)),
                showgrid=False,
                range=[x_vals.max(), x_vals.min()]  # flip x-axis
            ),
            yaxis=dict(
                title=dict(text="Component 1", font=dict(size=18)),
                showgrid=False,
                range=[y_vals.max(), y_vals.min()]  # flip y-axis
            ),
            legend=dict(title=dict(text="SkillCorner Labels"), font=dict(size=16)),
            width=1000,
            height=800,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        return go.Figure(data=traces, layout=layout)