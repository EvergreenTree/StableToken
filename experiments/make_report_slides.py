#!/usr/bin/env python
"""Generate an offline HTML slide deck from the StableToken report artifacts."""
from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SLIDE_DIR = ROOT / "experiments" / "reports" / "slides"
ASSET_DIR = SLIDE_DIR / "assets"

BG = "#f7f4ee"
INK = "#182026"
MUTED = "#607079"
BLUE = "#315f72"
TEAL = "#6fa39b"
RED = "#b85450"
GOLD = "#c58b3c"
GREEN = "#4f8a5b"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "font.size": 11,
            "axes.grid": True,
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "legend.frameon": False,
        }
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def copy_asset(src: Path, name: str) -> str:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    dst = ASSET_DIR / name
    shutil.copy2(src, dst)
    return f"assets/{name}"


def draw_boxes(path: Path, title: str, boxes: list[dict[str, Any]], links: list[tuple[int, int]] | None = None) -> str:
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.75)
    ax.axis("off")
    ax.text(0.5, 6.25, title, fontsize=22, weight="bold", color=INK)
    for idx, box in enumerate(boxes):
        x, y, w, h = box["xywh"]
        color = box.get("color", BLUE)
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="none", alpha=0.95)
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            box["text"],
            ha="center",
            va="center",
            color="white",
            fontsize=box.get("fontsize", 13),
            weight="bold",
            wrap=True,
        )
    if links:
        for left, right in links:
            x1, y1, w1, h1 = boxes[left]["xywh"]
            x2, y2, _w2, h2 = boxes[right]["xywh"]
            ax.annotate(
                "",
                xy=(x2, y2 + h2 / 2),
                xytext=(x1 + w1, y1 + h1 / 2),
                arrowprops=dict(arrowstyle="->", lw=2.3, color=MUTED),
            )
    savefig(path)
    return f"assets/{path.name}"


def title_visual() -> str:
    path = ASSET_DIR / "slide_01_title.png"
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.75)
    ax.axis("off")
    logo = ROOT / "assets" / "stabletoken-logo.png"
    if logo.exists():
        img = plt.imread(logo)
        ax.imshow(img, extent=(0.4, 3.2, 3.75, 6.2), aspect="auto")
    ax.text(0.55, 3.25, "StableToken", fontsize=30, weight="bold", color=INK)
    ax.text(0.55, 2.85, "Reviewer-grade local evaluation", fontsize=16, color=MUTED)
    xs = [4.2, 6.2, 8.2, 10.2]
    labels = ["30s audio", "Whisper\nencoder", "LFQ L16\n8192 codes", "25 Hz\ntokens"]
    colors = [BLUE, TEAL, GOLD, RED]
    for i, (x, label, color) in enumerate(zip(xs, labels, colors)):
        ax.add_patch(plt.Rectangle((x, 3.25), 1.45, 1.0, facecolor=color, edgecolor="none"))
        ax.text(x + 0.725, 3.75, label, ha="center", va="center", color="white", fontsize=12, weight="bold")
        if i < len(xs) - 1:
            ax.annotate("", xy=(x + 1.9, 3.75), xytext=(x + 1.45, 3.75), arrowprops=dict(arrowstyle="->", lw=2.2, color=MUTED))
    ax.text(4.2, 2.5, "Robustness | pseudo-streaming | downstream usefulness", fontsize=14, color=INK)
    savefig(path)
    return f"assets/{path.name}"


def why_evaluation_visual() -> str:
    path = ASSET_DIR / "slide_02_claim_map.png"
    boxes = [
        {"xywh": (0.7, 3.9, 2.2, 1.0), "text": "Robustness", "color": BLUE},
        {"xywh": (3.4, 3.9, 2.2, 1.0), "text": "Streaming", "color": TEAL},
        {"xywh": (6.1, 3.9, 2.2, 1.0), "text": "Distribution", "color": GOLD},
        {"xywh": (8.8, 3.9, 2.2, 1.0), "text": "Downstream", "color": RED},
        {"xywh": (2.0, 1.8, 2.2, 1.0), "text": "Config runs", "color": "#53656e"},
        {"xywh": (4.9, 1.8, 2.2, 1.0), "text": "CSV/JSON", "color": "#53656e"},
        {"xywh": (7.8, 1.8, 2.2, 1.0), "text": "Plots", "color": "#53656e"},
    ]
    return draw_boxes(path, "Claim to evidence map", boxes, [(0, 4), (1, 4), (2, 5), (3, 6)])


def stabletoken_pipeline_visual() -> str:
    path = ASSET_DIR / "slide_03_pipeline.png"
    boxes = [
        {"xywh": (0.7, 3.5, 1.8, 1.0), "text": "30s\nnon-causal\nchunk", "color": BLUE},
        {"xywh": (3.1, 3.5, 1.8, 1.0), "text": "Whisper\nencoder", "color": TEAL},
        {"xywh": (5.5, 3.5, 1.8, 1.0), "text": "LFQ\nlayer 16", "color": GOLD},
        {"xywh": (7.9, 3.5, 1.8, 1.0), "text": "8192\ncodes", "color": RED},
        {"xywh": (4.2, 1.65, 2.4, 0.9), "text": "25 Hz token stream", "color": GREEN},
    ]
    return draw_boxes(path, "Released tokenizer shape", boxes, [(0, 1), (1, 2), (2, 3), (2, 4)])


def harness_visual() -> str:
    path = ASSET_DIR / "slide_04_harness.png"
    boxes = [
        {"xywh": (0.6, 4.0, 2.1, 0.9), "text": "YAML\nconfigs", "color": BLUE},
        {"xywh": (3.3, 4.0, 2.1, 0.9), "text": "Fixed\ncheckpoint", "color": TEAL},
        {"xywh": (6.0, 4.0, 2.1, 0.9), "text": "Local\nsplits", "color": GOLD},
        {"xywh": (8.7, 4.0, 2.1, 0.9), "text": "Seeded\nruns", "color": RED},
        {"xywh": (2.0, 2.0, 2.5, 0.9), "text": "CSV / JSON", "color": "#53656e"},
        {"xywh": (5.2, 2.0, 2.5, 0.9), "text": "PNG plots", "color": "#53656e"},
        {"xywh": (8.4, 2.0, 2.5, 0.9), "text": "Git ledger", "color": "#53656e"},
    ]
    return draw_boxes(path, "Config-driven experiment harness", boxes, [(0, 4), (1, 4), (2, 5), (3, 6)])


def metric_cards_visual() -> str:
    path = ASSET_DIR / "slide_05_sanity_cards.png"
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.75)
    ax.text(0.55, 6.05, "Released checkpoint sanity", fontsize=22, weight="bold", color=INK)
    cards = [
        ("Token rate", "25.0 Hz", BLUE),
        ("30s crop", "750 tokens", TEAL),
        ("Vocabulary", "8192 codes", GOLD),
        ("Bitrate", "325 bps", RED),
    ]
    for i, (label, value, color) in enumerate(cards):
        x = 0.75 + i * 2.8
        ax.add_patch(plt.Rectangle((x, 2.4), 2.35, 1.8, facecolor=color, edgecolor="none"))
        ax.text(x + 1.175, 3.55, value, ha="center", va="center", color="white", fontsize=21, weight="bold")
        ax.text(x + 1.175, 2.85, label, ha="center", va="center", color="white", fontsize=12)
    savefig(path)
    return f"assets/{path.name}"


def codec_channel_visual() -> str:
    path = ASSET_DIR / "slide_07_codec_channel.png"
    rows = read_csv(ROOT / "experiments/runs/degradation_full_100_20260517/degradation_summary.csv")
    wanted = [("aac_32k", "AAC 32k"), ("mp3_32k", "MP3 32k"), ("opus_24k", "Opus 24k"), ("telephone_bandpass", "Telephone")]
    langs = ["fr", "zh"]
    values = {lang: [] for lang in langs}
    for corr, _label in wanted:
        for lang in langs:
            match = [r for r in rows if r["corruption"] == corr and r["language"] == lang]
            values[lang].append(float(match[0]["mean_ued"]) if match else 0.0)
    x = np.arange(len(wanted))
    width = 0.36
    plt.figure(figsize=(9.4, 5.1))
    plt.bar(x - width / 2, values["fr"], width, label="FR", color=BLUE)
    plt.bar(x + width / 2, values["zh"], width, label="ZH", color=RED)
    plt.xticks(x, [label for _corr, label in wanted])
    plt.ylabel("Mean UED")
    plt.title("Codec and channel stressors are measurable")
    plt.ylim(0, 0.32)
    plt.legend()
    savefig(path)
    return f"assets/{path.name}"


def streaming_windows_visual() -> str:
    path = ASSET_DIR / "slide_08_streaming_windows.png"
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.set_xlim(0, 75)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("Absolute time (s)")
    ax.set_yticks([3.3, 2.3, 1.3])
    ax.set_yticklabels(["30s non-overlap", "15s stride", "1s stride"])
    ax.set_title("Same audio span, different 30s windows")
    for start in [0, 30, 60]:
        ax.broken_barh([(start, min(30, 75 - start))], (3.0, 0.45), facecolors=BLUE)
    for start in [0, 15, 30, 45]:
        ax.broken_barh([(start, 30)], (2.0, 0.45), facecolors=TEAL)
    for start in range(0, 46, 5):
        ax.broken_barh([(start, 30)], (1.0, 0.36), facecolors=GOLD, alpha=0.55)
    ax.axvspan(29.5, 30.5, color=RED, alpha=0.18)
    ax.text(31, 3.75, "boundary region", color=RED, fontsize=12, weight="bold")
    savefig(path)
    return f"assets/{path.name}"


def asr_probe_visual() -> str:
    path = ASSET_DIR / "slide_13_asr_probe.png"
    boxes = [
        {"xywh": (0.7, 3.5, 2.0, 0.95), "text": "Audio", "color": BLUE},
        {"xywh": (3.25, 3.5, 2.0, 0.95), "text": "StableToken\nencoder", "color": TEAL},
        {"xywh": (5.8, 3.5, 2.0, 0.95), "text": "Quantized\nstates", "color": GOLD},
        {"xywh": (8.35, 3.5, 2.0, 0.95), "text": "Stock Whisper\ndecoder", "color": RED},
        {"xywh": (4.6, 1.65, 2.8, 0.95), "text": "WER/CER = 1.0\nempty punctuation output", "color": "#53656e"},
    ]
    return draw_boxes(path, "Zero-shot ASR probe fails", boxes, [(0, 1), (1, 2), (2, 3), (3, 4)])


def training_harness_visual() -> str:
    path = ASSET_DIR / "slide_14_training_harness.png"
    boxes = [
        {"xywh": (0.6, 3.8, 2.1, 0.95), "text": "Whisper\ncheckpoint", "color": BLUE},
        {"xywh": (3.2, 3.8, 2.1, 0.95), "text": "LFQ\nencoder", "color": TEAL},
        {"xywh": (5.8, 3.8, 2.1, 0.95), "text": "ASR CE +\nLFQ losses", "color": GOLD},
        {"xywh": (8.4, 3.8, 2.1, 0.95), "text": "Tokenizer\ncheckpoint", "color": RED},
        {"xywh": (2.0, 1.75, 2.4, 0.9), "text": "5 variants", "color": "#53656e"},
        {"xywh": (5.0, 1.75, 2.4, 0.9), "text": "1000 steps", "color": "#53656e"},
        {"xywh": (8.0, 1.75, 2.4, 0.9), "text": "2000 FR rows", "color": "#53656e"},
    ]
    return draw_boxes(path, "Reconstructed LFQ training path", boxes, [(0, 1), (1, 2), (2, 3)])


def open_work_visual() -> str:
    path = ASSET_DIR / "slide_17_open_work.png"
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.75)
    ax.text(0.6, 6.0, "What remains open", fontsize=22, weight="bold", color=INK)
    rows = [
        ("SER", "not executed", "Add emotion probe"),
        ("Speaker/prosody", "not executed", "Add preservation metrics"),
        ("SpeechLLM tasks", "not executed", "Adapter/LoRA first"),
        ("Tokenizer ASR", "zero-shot failed", "Train head/adapter"),
    ]
    y = 4.8
    for item, status, next_step in rows:
        ax.add_patch(plt.Rectangle((0.8, y - 0.45), 2.4, 0.65, facecolor=BLUE, edgecolor="none"))
        ax.add_patch(plt.Rectangle((3.45, y - 0.45), 2.4, 0.65, facecolor=RED, edgecolor="none"))
        ax.add_patch(plt.Rectangle((6.1, y - 0.45), 3.6, 0.65, facecolor=TEAL, edgecolor="none"))
        ax.text(2.0, y - 0.12, item, ha="center", va="center", color="white", weight="bold")
        ax.text(4.65, y - 0.12, status, ha="center", va="center", color="white", weight="bold")
        ax.text(7.9, y - 0.12, next_step, ha="center", va="center", color="white", weight="bold")
        y -= 1.0
    savefig(path)
    return f"assets/{path.name}"


def takeaways_visual() -> str:
    path = ASSET_DIR / "slide_18_takeaways.png"
    boxes = [
        {"xywh": (0.9, 2.5, 2.8, 1.6), "text": "Reproducible\n25 Hz\n750 tokens", "color": BLUE, "fontsize": 16},
        {"xywh": (4.6, 2.5, 2.8, 1.6), "text": "Hardest stressor\nBabble\nUED 0.6071", "color": RED, "fontsize": 16},
        {"xywh": (8.3, 2.5, 2.8, 1.6), "text": "Not true streaming\nOverlap mismatch\n0.3191", "color": GOLD, "fontsize": 16},
    ]
    return draw_boxes(path, "Three takeaways", boxes)


def build_assets() -> dict[str, str]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    assets = {
        "title": title_visual(),
        "why": why_evaluation_visual(),
        "pipeline": stabletoken_pipeline_visual(),
        "harness": harness_visual(),
        "sanity": metric_cards_visual(),
        "degradation": copy_asset(
            ROOT / "experiments/runs/degradation_full_100_20260517/plots/degradation_mean_ued.png",
            "slide_06_degradation_mean_ued.png",
        ),
        "codec": codec_channel_visual(),
        "streaming": streaming_windows_visual(),
        "chunking": copy_asset(
            ROOT / "experiments/reports/figures/figure_chunking_tradeoff.png",
            "slide_09_chunking_tradeoff.png",
        ),
        "distribution": copy_asset(
            ROOT / "experiments/reports/figures/figure_distribution_drift.png",
            "slide_10_distribution_drift.png",
        ),
        "zipf": copy_asset(
            ROOT / "experiments/runs/distribution_degradation_20260517/plots/distribution_zipf.png",
            "slide_11_distribution_zipf.png",
        ),
        "asr_mismatch": copy_asset(
            ROOT / "experiments/reports/figures/figure_token_asr_mismatch.png",
            "slide_12_token_asr_mismatch.png",
        ),
        "asr_probe": asr_probe_visual(),
        "training": training_harness_visual(),
        "lfq_matrix": copy_asset(
            ROOT / "experiments/reports/figures/figure_lfq_ablation_matrix.png",
            "slide_15_lfq_ablation_matrix.png",
        ),
        "latency": copy_asset(
            ROOT / "experiments/runs/latency_smoke_20260517/plots/latency_rtf.png",
            "slide_16_latency_rtf.png",
        ),
        "open_work": open_work_visual(),
        "takeaways": takeaways_visual(),
    }
    copy_asset(ROOT / "assets/stabletoken-logo.png", "stabletoken-logo.png")
    write_json(ASSET_DIR / "asset_manifest.json", assets)
    return assets


def metric(label: str, value: str) -> dict[str, str]:
    return {"label": label, "value": value}


def slides(assets: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "title": "StableToken: Robust Speech Token Evaluation",
            "eyebrow": "Technical interview deck",
            "image": assets["title"],
            "metrics": [metric("Presenter", "Changqing Fu"), metric("Affiliation", "IFM MBZUAI Paris"), metric("Date", "May 19, 2026")],
            "message": "A 30-minute review of reproducibility, robustness, pseudo-streaming, distribution drift, and downstream usefulness.",
        },
        {
            "title": "Why This Evaluation",
            "eyebrow": "Reviewer concern map",
            "image": assets["why"],
            "metrics": [metric("Baseline", "released checkpoint"), metric("Evidence", "10 experiment families")],
            "message": "Robustness, streaming behavior, and downstream usefulness are separate claims that need separate evidence.",
        },
        {
            "title": "StableToken At A Glance",
            "eyebrow": "Released tokenizer shape",
            "image": assets["pipeline"],
            "metrics": [metric("Clean baseline", "30s -> 750 tokens"), metric("Result shape", "8192 vocab, 325 bps")],
            "message": "The released tokenizer is fast and compact, but it is built around non-causal 30s chunks.",
        },
        {
            "title": "Experiment Harness",
            "eyebrow": "Audit-friendly runs",
            "image": assets["harness"],
            "metrics": [metric("Baseline", "fixed HF SHA"), metric("Artifacts", "CSV, JSON, PNG")],
            "message": "Every result is config-driven and tied to local splits, seeds, commands, and committed artifacts.",
        },
        {
            "title": "Reproducibility Sanity",
            "eyebrow": "Released checkpoint loads",
            "image": assets["sanity"],
            "metrics": [metric("Clean baseline", "25.0 Hz"), metric("Result", "750 tokens per 30s")],
            "message": "The released checkpoint matches the expected token rate, vocabulary size, and bitrate.",
        },
        {
            "title": "Hard Degradations",
            "eyebrow": "100-clip UED scale-up",
            "image": assets["degradation"],
            "metrics": [metric("Clean reference", "UED 0"), metric("Babble UED", "FR 0.6071, ZH 0.4170")],
            "message": "Speech-like interference is harder than Gaussian noise and should be a headline robustness test.",
        },
        {
            "title": "Codec And Channel Effects",
            "eyebrow": "Low-bitrate and telephone stressors",
            "image": assets["codec"],
            "metrics": [metric("AAC UED", "FR 0.2530, ZH 0.1371"), metric("Telephone UED", "FR 0.2538, ZH 0.1418")],
            "message": "Codec and channel effects are measurable, but they are not the dominant failure mode in this slice.",
        },
        {
            "title": "Pseudo-Streaming Problem",
            "eyebrow": "Window placement changes tokens",
            "image": assets["streaming"],
            "metrics": [metric("Baseline", "non-overlap = 0 mismatch"), metric("Best overlap", "0.3191 mismatch")],
            "message": "The same absolute audio can tokenize differently depending on which 30s window contains it.",
        },
        {
            "title": "Aggregation And Commit Latency",
            "eyebrow": "Post-hoc fixes are weak",
            "image": assets["chunking"],
            "metrics": [metric("1s latency", "0.7954 mismatch"), metric("15s latency", "0.6051 mismatch")],
            "message": "Confidence and hidden-state aggregation do not recover the offline non-overlap token stream.",
        },
        {
            "title": "Token Distribution Drift",
            "eyebrow": "KL clean to corruption",
            "image": assets["distribution"],
            "metrics": [metric("Gaussian KL", "FR 1.124, ZH 0.913"), metric("Babble KL", "FR 2.456, ZH 2.353")],
            "message": "Hard corruptions shift token usage more than additive Gaussian noise.",
        },
        {
            "title": "Token Usage And Zipf",
            "eyebrow": "No obvious collapse",
            "image": assets["zipf"],
            "metrics": [metric("FR clean unique", "4237 tokens"), metric("FR babble unique", "4665 tokens")],
            "message": "Code usage is broad, but analyses must be stratified by language and corruption.",
        },
        {
            "title": "Downstream ASR Mismatch",
            "eyebrow": "Tokenizer UED is not task WER",
            "image": assets["asr_mismatch"],
            "metrics": [metric("Clean FR WER", "0.1165"), metric("Babble FR WER", "0.2885")],
            "message": "Token robustness and direct Whisper-small ASR degradation are related but not interchangeable.",
        },
        {
            "title": "StableToken-Token ASR Probe",
            "eyebrow": "Zero-shot decoder replacement",
            "image": assets["asr_probe"],
            "metrics": [metric("Clean WER/CER", "1.0"), metric("Output", "empty or punctuation")],
            "message": "WER through StableToken needs a trained adapter or head, not a stock Whisper decoder.",
        },
        {
            "title": "Training Harness",
            "eyebrow": "Reconstructed LFQ path",
            "image": assets["training"],
            "metrics": [metric("Matrix", "5 matched variants"), metric("Budget", "1000 steps, 2000 FR rows")],
            "message": "The training runs are short diagnostics and should not be treated as paper-number reproduction.",
        },
        {
            "title": "LFQ Ablation Matrix",
            "eyebrow": "Short matched pilot",
            "image": assets["lfq_matrix"],
            "metrics": [metric("Best mean UED", "single_aug 0.6869"), metric("Voting row", "multi_aug_consensus 0.7970")],
            "message": "Augmentation helped modestly, while multi-branch Voting-LFQ hurt UED at this scale.",
        },
        {
            "title": "Efficiency",
            "eyebrow": "Latency smoke",
            "image": assets["latency"],
            "metrics": [metric("30s batch 1", "0.151s"), metric("RTF", "0.00504")],
            "message": "Token extraction is comfortably faster than realtime on the observed L4.",
        },
        {
            "title": "What Failed Or Remains Open",
            "eyebrow": "Next evidence gaps",
            "image": assets["open_work"],
            "metrics": [metric("Not executed", "SER, speaker, prosody"), metric("Next step", "trained ASR/SpeechLLM adapter")],
            "message": "The strongest next experiment is a tokenizer-through-task adapter evaluated on the fixed corruption suite.",
        },
        {
            "title": "Takeaways",
            "eyebrow": "Interview close",
            "image": assets["takeaways"],
            "metrics": [metric("Reproducible", "25 Hz"), metric("Hardest", "babble"), metric("Streaming", "not true streaming")],
            "message": "StableToken is inspectable and efficient, but robustness claims must separate tokenizer stability from downstream task impact.",
        },
    ]


def render_metrics(metrics: list[dict[str, str]]) -> str:
    return "\n".join(
        f'<div class="metric"><span>{html.escape(item["label"])}</span><strong>{html.escape(item["value"])}</strong></div>'
        for item in metrics
    )


def render_slide(slide: dict[str, Any], index: int, total: int) -> str:
    metrics_html = render_metrics(slide["metrics"])
    return f"""
      <section class="slide" id="slide-{index}">
        <div class="slide-head">
          <div>
            <p class="eyebrow">{html.escape(slide["eyebrow"])}</p>
            <h1>{html.escape(slide["title"])}</h1>
          </div>
          <div class="counter">{index:02d}/{total:02d}</div>
        </div>
        <div class="slide-body">
          <figure>
            <img src="{html.escape(slide["image"])}" alt="{html.escape(slide["title"])} visual">
          </figure>
          <aside>
            <div class="metrics">{metrics_html}</div>
            <p class="message">{html.escape(slide["message"])}</p>
          </aside>
        </div>
      </section>
    """


def render_html(deck_slides: list[dict[str, Any]]) -> str:
    slide_html = "\n".join(render_slide(slide, idx, len(deck_slides)) for idx, slide in enumerate(deck_slides, 1))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StableToken Interview Deck</title>
  <style>
    :root {{
      --bg: {BG};
      --ink: {INK};
      --muted: {MUTED};
      --blue: {BLUE};
      --teal: {TEAL};
      --red: {RED};
      --paper: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }}
    .deck {{
      width: 100vw;
      height: 100vh;
      position: relative;
    }}
    .slide {{
      display: none;
      width: 100vw;
      height: 100vh;
      padding: 4.2vh 4.8vw;
      background: var(--bg);
    }}
    .slide.active {{ display: flex; flex-direction: column; }}
    .slide-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 2rem;
      min-height: 14vh;
    }}
    .eyebrow {{
      margin: 0 0 0.6rem;
      color: var(--red);
      font-size: clamp(0.9rem, 1.25vw, 1.25rem);
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    h1 {{
      margin: 0;
      color: var(--ink);
      font-size: clamp(2rem, 4.4vw, 4.6rem);
      line-height: 1.02;
      letter-spacing: 0;
    }}
    .counter {{
      min-width: 5rem;
      padding: 0.65rem 0.8rem;
      border-top: 4px solid var(--blue);
      text-align: right;
      color: var(--muted);
      font-weight: 800;
      font-size: 1.05rem;
    }}
    .slide-body {{
      flex: 1;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(18rem, 24vw);
      gap: 3vw;
      align-items: stretch;
      min-height: 0;
    }}
    figure {{
      margin: 0;
      background: var(--paper);
      border: 1px solid #ded8cd;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 0;
      padding: 1rem;
      box-shadow: 0 14px 35px rgba(24, 32, 38, 0.09);
    }}
    figure img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: block;
    }}
    aside {{
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 1rem;
      min-height: 0;
    }}
    .metrics {{
      display: grid;
      gap: 0.85rem;
    }}
    .metric {{
      background: var(--paper);
      border-left: 6px solid var(--blue);
      border-radius: 8px;
      padding: 0.9rem 1rem;
      box-shadow: 0 8px 24px rgba(24, 32, 38, 0.07);
    }}
    .metric:nth-child(2n) {{ border-left-color: var(--teal); }}
    .metric:nth-child(3n) {{ border-left-color: var(--red); }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: clamp(0.8rem, 1vw, 0.95rem);
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
      margin-bottom: 0.35rem;
    }}
    .metric strong {{
      display: block;
      color: var(--ink);
      font-size: clamp(1.15rem, 1.75vw, 1.7rem);
      line-height: 1.1;
    }}
    .message {{
      margin: 0;
      background: #182026;
      color: white;
      border-radius: 8px;
      padding: 1rem 1.1rem;
      font-size: clamp(1rem, 1.35vw, 1.35rem);
      line-height: 1.35;
      font-weight: 700;
    }}
    .nav-hint {{
      position: fixed;
      left: 1rem;
      bottom: 0.7rem;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
    }}
    @media (max-aspect-ratio: 4/3) {{
      .slide {{ padding: 3vh 4vw; }}
      .slide-body {{ grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) auto; }}
      aside {{ display: grid; grid-template-columns: 1fr 1fr; align-items: stretch; }}
    }}
    @media print {{
      body {{ overflow: visible; background: white; }}
      .deck {{ width: auto; height: auto; }}
      .slide, .slide.active {{
        display: flex;
        width: 100vw;
        height: 100vh;
        page-break-after: always;
      }}
      .nav-hint {{ display: none; }}
    }}
  </style>
</head>
<body>
  <main class="deck">
{slide_html}
  </main>
  <div class="nav-hint">Use left/right arrows, space, Home, End. Print to PDF from the browser.</div>
  <script>
    const slides = Array.from(document.querySelectorAll('.slide'));
    let current = 0;
    function show(index) {{
      current = Math.max(0, Math.min(slides.length - 1, index));
      slides.forEach((slide, idx) => slide.classList.toggle('active', idx === current));
      window.location.hash = `slide-${{current + 1}}`;
    }}
    function fromHash() {{
      const match = window.location.hash.match(/slide-(\\d+)/);
      if (match) show(Number(match[1]) - 1);
      else show(0);
    }}
    window.addEventListener('keydown', event => {{
      if (['ArrowRight', 'PageDown', ' '].includes(event.key)) {{ event.preventDefault(); show(current + 1); }}
      if (['ArrowLeft', 'PageUp'].includes(event.key)) {{ event.preventDefault(); show(current - 1); }}
      if (event.key === 'Home') {{ event.preventDefault(); show(0); }}
      if (event.key === 'End') {{ event.preventDefault(); show(slides.length - 1); }}
    }});
    window.addEventListener('hashchange', fromHash);
    fromHash();
  </script>
</body>
</html>
"""


def main() -> None:
    setup_style()
    assets = build_assets()
    deck_slides = slides(assets)
    SLIDE_DIR.mkdir(parents=True, exist_ok=True)
    html_path = SLIDE_DIR / "stabletoken_interview_deck.html"
    html_path.write_text(render_html(deck_slides), encoding="utf-8")
    write_json(SLIDE_DIR / "slides_manifest.json", deck_slides)
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
