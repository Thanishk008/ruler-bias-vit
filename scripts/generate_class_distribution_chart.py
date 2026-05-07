from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "isic2019" / "ISIC_2019_Training_GroundTruth.csv"
OUT_PATH = ROOT / "outputs" / "report_assets" / "class_distribution_chart.png"

CLASS_ORDER = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_LABELS = {
    "MEL": "MEL",
    "NV": "NV",
    "BCC": "BCC",
    "AK": "AK",
    "BKL": "BKL",
    "DF": "DF",
    "VASC": "VASC",
    "SCC": "SCC",
}


def load_counts() -> dict[str, int]:
    counts = {name: 0 for name in CLASS_ORDER}
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for class_name in CLASS_ORDER:
                if float(row.get(class_name, 0) or 0) == 1.0:
                    counts[class_name] += 1
    return counts


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = load_counts()

    labels = [CLASS_LABELS[name] for name in CLASS_ORDER]
    values = [counts[name] for name in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="white")
    colors = ["#547aa5", "#2a9d8f", "#4c78a8", "#f4a261", "#8ab17d", "#b56576", "#6c8ead", "#c97b63"]
    bars = ax.bar(labels, values, color=colors, edgecolor="#243447", linewidth=0.8)

    ax.set_title("ISIC 2019 Filtered Class Distribution", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Image Count", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    max_val = max(values)
    ax.set_ylim(0, max_val * 1.15)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max_val * 0.015,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    note = (
        "Counts exclude UNK and match the filtered dataset used in this project "
        f"(total = {sum(values):,} images)."
    )
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=10, color="#444444")

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
