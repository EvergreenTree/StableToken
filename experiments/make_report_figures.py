#!/usr/bin/env python
"""Build illustrative figures for the reviewer-facing StableToken report."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "experiments" / "reports" / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "font.size": 10,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "legend.frameon": False,
        }
    )


def save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def chunking_tradeoff() -> None:
    ref_rows = read_csv(ROOT / "experiments/runs/chunking_advanced_20260517/chunking_reference_comparison.csv")
    lat_rows = read_csv(ROOT / "experiments/runs/chunking_advanced_20260517/chunking_commit_latency_metrics.csv")

    wanted = [
        ("stride_15s", "first", "15s stride / first"),
        ("stride_15s", "edge_voter", "15s stride / edge+voter"),
        ("stride_5s", "first", "5s stride / first"),
        ("stride_5s", "edge_voter", "5s stride / edge+voter"),
        ("stride_1s", "first", "1s stride / first"),
        ("stride_1s", "edge_voter", "1s stride / edge+voter"),
        ("stride_1s", "hidden_edge_voter", "1s stride / hidden edge+voter"),
    ]
    by_policy: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in ref_rows:
        if row["reference"] != "nonoverlap_majority":
            continue
        by_policy[(row["policy"], row["aggregation"])].append(float(row["mismatch_rate"]))

    labels = [label for _, _, label in wanted]
    values = [mean(by_policy[(policy, agg)]) for policy, agg, _ in wanted]

    best_by_latency: dict[tuple[str, float], list[float]] = defaultdict(list)
    coverage_by_latency: dict[tuple[str, float], list[float]] = defaultdict(list)
    grouped: dict[tuple[str, str, float], list[tuple[float, float]]] = defaultdict(list)
    for row in lat_rows:
        key = (row["reference"], row["aggregation"], float(row["commit_latency_seconds"]))
        grouped[key].append((float(row["mismatch_rate"]), float(row["coverage_vs_reference"])))
    refs = ["high_overlap_stride_1s_edge_voter", "nonoverlap_majority"]
    for ref in refs:
        for latency in sorted({float(row["commit_latency_seconds"]) for row in lat_rows}):
            candidates = [
                (mean([x[0] for x in vals]), mean([x[1] for x in vals]))
                for (local_ref, _agg, local_latency), vals in grouped.items()
                if local_ref == ref and local_latency == latency
            ]
            if candidates:
                mismatch, coverage = min(candidates, key=lambda item: item[0])
                best_by_latency[(ref, latency)].append(mismatch)
                coverage_by_latency[(ref, latency)].append(coverage)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = ["#315f72", "#7aa6a1", "#315f72", "#7aa6a1", "#315f72", "#7aa6a1", "#c05a4b"]
    axes[0].bar(range(len(labels)), values, color=colors)
    axes[0].set_title("Window policy mismatch vs non-overlap")
    axes[0].set_ylabel("Mean token mismatch")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylim(0, max(values) * 1.18)

    ref_labels = {
        "high_overlap_stride_1s_edge_voter": "best vs high-overlap consensus",
        "nonoverlap_majority": "best vs non-overlap",
    }
    for ref, color in zip(refs, ["#2f6f73", "#b85450"]):
        xs = sorted({lat for local_ref, lat in best_by_latency if local_ref == ref})
        ys = [best_by_latency[(ref, lat)][0] for lat in xs]
        axes[1].plot(xs, ys, marker="o", linewidth=2.2, color=color, label=ref_labels[ref])
    axes[1].set_title("Commit-latency tradeoff")
    axes[1].set_xlabel("Commit latency (s)")
    axes[1].set_ylabel("Best mismatch")
    axes[1].set_xticks([1, 3, 5, 10, 15])
    axes[1].set_ylim(0, 0.9)
    axes[1].legend(loc="upper right")
    save(FIG_DIR / "figure_chunking_tradeoff.png")


def degradation_vs_asr() -> None:
    rows = read_csv(ROOT / "experiments/analysis/token_asr_degradation_comparison.csv")
    labels = {
        "aac_32k": "AAC",
        "babble_4spk_10db": "Babble",
        "competing_speech_16db": "Competing",
        "gaussian_25db": "Gaussian",
        "reverb_small_room": "Reverb",
        "telephone_bandpass": "Telephone",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharex=False)
    for ax, lang, y_key, y_label in [
        (axes[0], "fr", "asr_mean_wer", "Whisper-small WER"),
        (axes[1], "zh", "asr_mean_cer", "Whisper-small CER"),
    ]:
        subset = [row for row in rows if row["language"] == lang]
        xs = [float(row["token_mean_ued"]) for row in subset]
        ys = [float(row[y_key]) for row in subset]
        ax.scatter(xs, ys, s=72, color="#2f6f73", alpha=0.9)
        for row, x, y in zip(subset, xs, ys):
            ax.annotate(labels.get(row["corruption"], row["corruption"]), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_title(f"{lang.upper()} token UED vs downstream ASR")
        ax.set_xlabel("Token UED")
        ax.set_ylabel(y_label)
        ax.set_xlim(0, max(xs) * 1.18)
        ax.set_ylim(0, max(ys) * 1.18)
    save(FIG_DIR / "figure_token_asr_mismatch.png")


def distribution_drift() -> None:
    rows = read_csv(ROOT / "experiments/runs/distribution_degradation_20260517/distribution_kl.csv")
    conditions = [
        ("gaussian_25db", "Gaussian"),
        ("reverb_small_room", "Reverb"),
        ("babble_4spk_10db", "Babble"),
        ("competing_speech_16db", "Competing"),
    ]
    langs = [("fr", "FR"), ("zh", "ZH")]
    values = {lang: [] for lang, _ in langs}
    for condition, _label in conditions:
        for lang, _ in langs:
            match = [
                row
                for row in rows
                if row["left"] == f"{lang}:clean" and row["right"] == f"{lang}:{condition}"
            ]
            values[lang].append(float(match[0]["kl_bits"]) if match else 0.0)

    x = np.arange(len(conditions))
    width = 0.36
    plt.figure(figsize=(8.5, 4.4))
    plt.bar(x - width / 2, values["fr"], width, label="FR", color="#315f72")
    plt.bar(x + width / 2, values["zh"], width, label="ZH", color="#c05a4b")
    plt.xticks(x, [label for _condition, label in conditions])
    plt.ylabel("KL(clean -> corruption), bits")
    plt.title("Hard corruptions drive larger token-distribution drift")
    plt.legend()
    save(FIG_DIR / "figure_distribution_drift.png")


def lfq_ablation_matrix() -> None:
    path = ROOT / "experiments/analysis/lfq_ablation_matrix_summary.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    labels = [row["variant"].replace("_", "\n") for row in rows]
    mean_ued = [float(row["mean_all_ued"]) for row in rows]
    unique_rate = [float(row["clean_unique_token_rate"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].bar(range(len(labels)), mean_ued, color="#315f72")
    axes[0].set_title("LFQ matrix: mean UED")
    axes[0].set_ylabel("Mean UED, lower is better")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    axes[0].set_ylim(0, max(mean_ued) * 1.18)

    axes[1].bar(range(len(labels)), unique_rate, color="#7aa6a1")
    axes[1].set_title("LFQ matrix: clean token usage")
    axes[1].set_ylabel("Unique-token rate")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    axes[1].set_ylim(0, 1.0)
    save(FIG_DIR / "figure_lfq_ablation_matrix.png")


def main() -> None:
    setup_style()
    chunking_tradeoff()
    degradation_vs_asr()
    distribution_drift()
    lfq_ablation_matrix()
    print(f"Wrote report figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
