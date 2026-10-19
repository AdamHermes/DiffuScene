"""
Visualize iteration distribution histograms from collision_metrics_report.json
as bar charts for each room category.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    report_path = "collision_metrics_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rooms = data["rooms"]

    for room_name, metrics in rooms.items():
        hist = metrics["convergence"]["iteration_histogram"]
        n_scenes = metrics["n_scenes"]

        # Filter out empty buckets beyond the last non-zero one
        labels = []
        counts = []
        for label, count in hist.items():
            labels.append(label)
            counts.append(count)

        # Trim trailing zeros
        last_nonzero = 0
        for i, c in enumerate(counts):
            if c > 0:
                last_nonzero = i
        labels = labels[:last_nonzero + 1]
        counts = counts[:last_nonzero + 1]

        pcts = [c / n_scenes * 100 for c in counts]

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(labels))
        bars = ax.bar(x, pcts, color="#4C72B0", edgecolor="white", width=0.6)

        # Add count labels on top of bars
        for bar, count, pct in zip(bars, counts, pcts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{count}\n({pct:.1f}%)",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xlabel("Solver Iterations", fontsize=12)
        ax.set_ylabel("Scenes (%)", fontsize=12)
        ax.set_title(f"Iteration Distribution — {room_name.capitalize()} (n={n_scenes})",
                      fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, max(pcts) * 1.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        out_path = f"iteration_histogram_{room_name}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
