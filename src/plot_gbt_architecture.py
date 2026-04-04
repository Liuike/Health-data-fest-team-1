from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT_DIR / "artifacts" / "evaluation"


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    facecolor: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=2.0,
        edgecolor="#2f2f2f",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        x + w / 2,
        y + h * 0.32,
        body,
        ha="center",
        va="center",
        fontsize=12.5,
        wrap=True,
        color="#333333",
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=2.4,
            color="#444444",
            shrinkA=8,
            shrinkB=8,
        )
    )


def build_architecture_figure(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top_y = 0.55
    box_w = 0.16
    box_h = 0.22
    x_positions = [0.04, 0.24, 0.44, 0.64, 0.84]

    add_box(
        ax,
        x_positions[0],
        top_y,
        box_w,
        box_h,
        "Input Data",
        "Daily site panel\nWeather + sampling\nAdvisory labels",
        "#dceefb",
    )
    add_box(
        ax,
        x_positions[1],
        top_y,
        box_w,
        box_h,
        "Target",
        "Build `y_30d_onset`\nfrom future onset labels\nEligible rows only",
        "#e8f5e9",
    )
    add_box(
        ax,
        x_positions[2],
        top_y,
        box_w,
        box_h,
        "Features",
        "Remove leakage\nAdd 90-day lags\nAdd rolling history",
        "#fff3cd",
    )
    add_box(
        ax,
        x_positions[3],
        top_y,
        box_w,
        box_h,
        "Batch Split",
        "Random stratified split\nTrain + validation\nHeld-out test",
        "#f8d7da",
    )
    add_box(
        ax,
        x_positions[4],
        top_y,
        box_w,
        box_h,
        "Models",
        "Category_1 GBT\nCategory_2 GBT\nSeparate site encodings",
        "#e8e3ff",
    )

    lower_y = 0.18
    add_box(
        ax,
        0.30,
        lower_y,
        0.18,
        0.19,
        "Tuning",
        "Validation AUCPR\nEarly stopping\nThreshold selection",
        "#d1ecf1",
    )
    add_box(
        ax,
        0.56,
        lower_y,
        0.24,
        0.19,
        "Outputs",
        "Probability = confidence\nAlert flag from threshold\nModels, metrics, figures",
        "#d4edda",
    )

    for idx in range(len(x_positions) - 1):
        add_arrow(
            ax,
            (x_positions[idx] + box_w, top_y + box_h / 2),
            (x_positions[idx + 1], top_y + box_h / 2),
        )

    add_arrow(
        ax,
        (x_positions[3] + box_w * 0.5, top_y),
        (0.39, lower_y + 0.19),
    )
    add_arrow(
        ax,
        (x_positions[4] + box_w * 0.5, top_y),
        (0.68, lower_y + 0.19),
    )
    add_arrow(
        ax,
        (0.48, lower_y + 0.095),
        (0.56, lower_y + 0.095),
    )

    ax.text(
        0.5,
        0.94,
        "Gradient-Boosted Advisory Model",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.895,
        "Simplified batch-split training and inference flow",
        ha="center",
        va="center",
        fontsize=14,
        color="#444444",
    )

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_outline_markdown(output_path: Path) -> None:
    output_path.write_text(
        "\n".join(
            [
                "# Gradient-Boosted Tree Architecture",
                "",
                "1. Input: daily site-level panel with weather, sampling, and advisory data.",
                "2. Target: create `y_30d_onset` for advisory onset within the next 30 days.",
                "3. Features: remove leakage fields and add 90-day lag and rolling-history features.",
                "4. Split: use a stratified random batch split into train, validation, and test.",
                "5. Model: train one XGBoost classifier for `Category_1` and one for `Category_2`.",
                "6. Tuning: use validation data for hyperparameter search, early stopping, and threshold selection.",
                "7. Output: save confidence scores, alert flags, model files, and evaluation artifacts.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    image_path = EVALUATION_DIR / "gbt_architecture_outline.png"
    markdown_path = EVALUATION_DIR / "gbt_architecture_outline.md"
    build_architecture_figure(image_path)
    write_outline_markdown(markdown_path)
    print(f"Wrote {image_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
