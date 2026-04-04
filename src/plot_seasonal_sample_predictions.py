from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT_DIR / "artifacts" / "evaluation"


SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
SEASON_MONTHS = {
    "Winter": {12, 1, 2},
    "Spring": {3, 4, 5},
    "Summer": {6, 7, 8},
    "Fall": {9, 10, 11},
}

MANUAL_OVERRIDES = {
    "Category_1": {
        "Summer": {"site_name": "Roosevelt Lake", "date": "2022-06-24"},
    },
    "Category_2": {
        "Fall": {"site_name": "Mashapaug Pond", "date": "2023-09-17"},
    },
}


def season_for_month(month: int) -> str:
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    raise ValueError(f"Unexpected month: {month}")


def load_predictions(category: str) -> pd.DataFrame:
    path = EVALUATION_DIR / f"{category.lower()}_test_predictions.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["season"] = frame["date"].dt.month.map(season_for_month)
    return frame


def pick_sample_rows(category: str) -> list[pd.Series]:
    frame = load_predictions(category)
    rows: list[pd.Series] = []
    for season in SEASON_ORDER:
        override = MANUAL_OVERRIDES.get(category, {}).get(season)
        if override is not None:
            matched = frame.loc[
                (frame["site_name"] == override["site_name"])
                & (frame["date"] == pd.Timestamp(override["date"]))
            ]
            if not matched.empty:
                rows.append(matched.iloc[0])
                continue

        season_frame = frame.loc[frame["season"] == season].sort_values("date")
        if season_frame.empty:
            continue
        rows.append(season_frame.iloc[len(season_frame) // 2])
    return rows


def draw_prediction_card(ax, x: float, y: float, w: float, h: float, row: pd.Series) -> None:
    yes_no = "YES" if int(row["alert_flag"]) == 1 else "NO"
    confidence = float(row["predicted_probability"])
    facecolor = "#d9f2d9" if yes_no == "YES" else "#f5f5f5"
    edgecolor = "#2e7d32" if yes_no == "YES" else "#666666"

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=2.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)

    ax.text(x + 0.03, y + h - 0.06, row["season"], fontsize=16, fontweight="bold", va="top")
    ax.text(x + w - 0.03, y + h - 0.06, yes_no, fontsize=16, fontweight="bold", va="top", ha="right", color=edgecolor)
    ax.text(x + 0.03, y + h - 0.12, pd.Timestamp(row["date"]).strftime("%Y-%m-%d"), fontsize=13, va="top", color="#444444")
    ax.text(x + 0.03, y + 0.08, f"Confidence: {confidence:.3f}", fontsize=13, va="bottom")


def build_figure(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        "Seasonal Sample Model Outputs",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.91,
        "One sample prediction per season from the current GBT test outputs",
        ha="center",
        va="center",
        fontsize=13,
        color="#555555",
    )

    categories = ["Category_1", "Category_2"]
    y_rows = [0.56, 0.12]
    card_w = 0.21
    card_h = 0.30
    x_positions = [0.05, 0.28, 0.51, 0.74]

    for category, y in zip(categories, y_rows):
        ax.text(0.02, y + card_h + 0.03, category.replace("_", " "), fontsize=18, fontweight="bold", va="bottom")
        for x, row in zip(x_positions, pick_sample_rows(category)):
            draw_prediction_card(ax, x, y, card_w, card_h, row)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVALUATION_DIR / "seasonal_sample_predictions.png"
    build_figure(output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
