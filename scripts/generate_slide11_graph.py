from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Update these entries as teammate selections are finalized.
SERIES = [
    {
        "label": "Baseline\nViT",
        "recall": 0.4867,
        "f1": 0.4048,
        "color": "#7A8799",
    },
    {
        "label": "Segmented ViT\n(Jinali)",
        "recall": 0.4029,
        "f1": 0.3755,
        "color": "#2AE058",
    },
    {
        "label": "Swin Technique 1\n(Thanishk)",
        "recall": 0.5960,
        "f1": 0.4471,
        "color": "#4C78A8",
    },
    {
        "label": "DeiT\n(Tommy)",
        "recall": 0.4835,
        "f1": 0.3614,
        "color": "#AF032E",
    },
    {
        "label": "Foundation\nCLIP",
        "recall": 0.1430,
        "f1": 0.1297,
        "color": "#E17C05",
    },
]

TITLE = "Best Model Comparison"
Y_LABEL = "Score"
Y_MAX = 0.7
OUTPUT = Path("outputs/report_assets/slide11_best_model_macro_f1.png")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    labels = [item["label"] for item in SERIES]
    recall_values = [item["recall"] for item in SERIES]
    f1_values = [item["f1"] for item in SERIES]
    colors = [item["color"] for item in SERIES]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    x = range(len(labels))
    width = 0.37
    recall_bars = ax.bar(
        [i - width / 2 for i in x],
        recall_values,
        width=width,
        color=colors,
        edgecolor="#4A4A4A",
        linewidth=1.2,
        label="Macro Recall",
    )
    f1_bars = ax.bar(
        [i + width / 2 for i in x],
        f1_values,
        width=width,
        color=colors,
        edgecolor="#4A4A4A",
        linewidth=1.2,
        hatch="//",
        alpha=0.9,
        label="Macro F1",
    )

    ax.set_xticks(list(x), labels, fontsize=10)
    ax.set_ylim(0, Y_MAX)
    ax.set_ylabel(Y_LABEL, fontsize=11)
    ax.set_title(TITLE, fontsize=14, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, item in zip(recall_bars, SERIES):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(item["recall"] + 0.015, Y_MAX - 0.02),
            f"R {item['recall']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
            color="#111111",
        )
    for bar, item in zip(f1_bars, SERIES):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(item["f1"] + 0.015, Y_MAX - 0.02),
            f"F1 {item['f1']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
            color="#111111",
        )

    legend_handles = [
        Patch(facecolor="#FFFFFF", edgecolor="#4A4A4A", label="Solid: Macro Recall"),
        Patch(facecolor="#FFFFFF", edgecolor="#4A4A4A", hatch="//", label="Hatched: Macro F1"),
        Patch(facecolor="#7A8799", edgecolor="#4A4A4A", label="Shared baseline"),
        Patch(facecolor="#2AE058", edgecolor="#4A4A4A", label="Jinali best model"),
        Patch(facecolor="#4C78A8", edgecolor="#4A4A4A", label="Thanishk best model"),
        Patch(facecolor="#AF032E", edgecolor="#4A4A4A", label="Tommy best model"),
        Patch(facecolor="#E17C05", edgecolor="#4A4A4A", label="Shared foundation"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved graph to {OUTPUT}")


if __name__ == "__main__":
    main()
