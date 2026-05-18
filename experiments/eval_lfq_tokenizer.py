#!/usr/bin/env python
"""Evaluate a trained LFQ/Voting-LFQ tokenizer checkpoint.

The trainer saves an encoder-only `tokenizer/` subdirectory so this evaluator can
measure token stability without needing to instantiate the Whisper decoder.
Outputs are intentionally plain CSV/JSON/JSONL artifacts for ledger use.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import WhisperFeatureExtractor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.train_lfq_tokenizer import augment_audio, load_audio, pick_audio_path, pick_text, set_seed
from src.model.modeling_whisper import WhisperLFQEncoder


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def read_manifest(path: Path, max_items: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_items is not None and len(rows) >= max_items:
                break
            row = json.loads(line)
            row.setdefault("id", f"{path.stem}:{idx}")
            rows.append(row)
    return rows


def fit_duration(wav: np.ndarray, seconds: float | None, sample_rate: int) -> np.ndarray:
    if seconds is None:
        return wav.astype(np.float32)
    n = max(1, int(round(seconds * sample_rate)))
    if wav.shape[0] >= n:
        return wav[:n].copy()
    out = np.zeros(n, dtype=np.float32)
    out[: wav.shape[0]] = wav
    return out


def tokenizer_dir_for_checkpoint(checkpoint: Path) -> Path:
    nested = checkpoint / "tokenizer"
    if (nested / "config.json").exists():
        return nested
    if (checkpoint / "config.json").exists():
        return checkpoint
    raise FileNotFoundError(
        f"Could not find a WhisperLFQEncoder checkpoint at {checkpoint} or {checkpoint / 'tokenizer'}"
    )


def load_tokenizer_checkpoint(checkpoint: Path, device: str):
    tokenizer_dir = tokenizer_dir_for_checkpoint(checkpoint)
    model = WhisperLFQEncoder.from_pretrained(str(tokenizer_dir)).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(tokenizer_dir))
    return model, feature_extractor, tokenizer_dir


def feature_stride_samples(model, feature_extractor) -> int:
    pooling = model.config.pooling_kernel_size or 1
    return int(model.conv1.stride[0] * model.conv2.stride[0] * pooling * feature_extractor.hop_length)


def valid_token_mask(features, model, token_shape: torch.Size) -> torch.Tensor:
    mask = features.attention_mask
    mask = mask[:, :: model.conv1.stride[0] * model.conv2.stride[0]]
    pooling = model.config.pooling_kernel_size or 1
    mask = mask[:, ::pooling]
    if mask.shape != token_shape:
        mask = mask[:, : token_shape[1]]
        if mask.shape != token_shape:
            raise RuntimeError(f"attention mask shape {mask.shape} does not match tokens {token_shape}")
    return mask.bool()


def tokenize_arrays(
    model,
    feature_extractor,
    arrays: list[np.ndarray],
    device: str,
    batch_size: int,
    sample_rate: int,
) -> list[list[int]]:
    out: list[list[int]] = []
    stride = feature_stride_samples(model, feature_extractor)
    for start in range(0, len(arrays), batch_size):
        batch = arrays[start : start + batch_size]
        features = feature_extractor(
            batch,
            sampling_rate=sample_rate,
            return_attention_mask=True,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=stride,
        ).to(device)
        with torch.inference_mode():
            tokens = model(**features, require_only_quantized_token=True)
        mask = valid_token_mask(features, model, tokens.shape)
        for i in range(tokens.shape[0]):
            out.append(tokens[i][mask[i]].detach().cpu().long().tolist())
    return out


def levenshtein(a: list[int], b: list[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def perturbation_name(spec: dict[str, Any]) -> str:
    if spec.get("name"):
        return str(spec["name"])
    if spec["type"] == "bit_crush":
        lo = spec.get("min_bit_depth")
        hi = spec.get("max_bit_depth")
        return f"bit_crush_{lo}_{hi}bit"
    lo = spec.get("min_snr_db")
    hi = spec.get("max_snr_db")
    return f"{spec['type']}_{lo}_{hi}db"


def summarize(rows: list[dict[str, Any]], histograms: dict[str, Counter[int]], vocab_size: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['language']}:{row['perturbation']}"].append(row)
        grouped[f"all:{row['perturbation']}"].append(row)

    groups = {}
    for name, vals in sorted(grouped.items()):
        groups[name] = {
            "items": len(vals),
            "mean_edit_distance": float(np.mean([v["edit_distance"] for v in vals])),
            "mean_ued": float(np.mean([v["ued"] for v in vals])),
            "mean_normalized_edit_distance": float(np.mean([v["normalized_edit_distance"] for v in vals])),
            "mean_clean_tokens": float(np.mean([v["clean_tokens"] for v in vals])),
            "mean_perturbed_tokens": float(np.mean([v["perturbed_tokens"] for v in vals])),
        }

    histogram_summary = {}
    for condition, counts in sorted(histograms.items()):
        total = sum(counts.values())
        histogram_summary[condition] = {
            "total_tokens": int(total),
            "unique_tokens": int(len(counts)),
            "unique_token_rate": float(len(counts) / vocab_size) if vocab_size else 0.0,
            "top_tokens": [int(token) for token, _ in counts.most_common(10)],
        }

    return {
        "items": len(rows),
        "groups": groups,
        "histograms": histogram_summary,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg: dict[str, Any] = load_yaml(Path(args.config)) if args.config else {}
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    if args.manifest:
        cfg["manifest"] = args.manifest
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.max_items is not None:
        cfg["max_items"] = args.max_items
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.device:
        cfg["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed

    if not cfg.get("checkpoint") or not cfg.get("manifest"):
        raise ValueError("Provide `checkpoint` and `manifest` in config or CLI arguments")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    device = select_device(str(cfg.get("device", "auto")))
    sample_rate = int(cfg.get("sample_rate", 16000))
    batch_size = int(cfg.get("batch_size", 1))
    checkpoint = Path(cfg["checkpoint"])
    output_dir = Path(cfg.get("output_dir") or REPO_ROOT / "experiments" / "runs" / f"lfq_eval_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, feature_extractor, tokenizer_dir = load_tokenizer_checkpoint(checkpoint, device)
    rows = read_manifest(Path(cfg["manifest"]), cfg.get("max_items"))
    audios = [
        fit_duration(load_audio(pick_audio_path(row), sample_rate), cfg.get("duration_seconds"), sample_rate)
        for row in rows
    ]
    clean_tokens = tokenize_arrays(model, feature_extractor, audios, device, batch_size, sample_rate)

    perturbations = cfg.get("perturbations") or [
        {"name": "gaussian_25db", "type": "gaussian", "min_snr_db": 25, "max_snr_db": 25}
    ]
    metric_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    histograms: dict[str, Counter[int]] = defaultdict(Counter)

    for row, tokens in zip(rows, clean_tokens):
        histograms["clean"].update(int(token) for token in tokens)

    for perturb in perturbations:
        name = perturbation_name(perturb)
        aug_cfg = {"enabled": True, "choices": [perturb]}
        perturbed = [augment_audio(wav, aug_cfg, sample_rate, rng) for wav in audios]
        perturbed_tokens = tokenize_arrays(model, feature_extractor, perturbed, device, batch_size, sample_rate)
        for row, clean, noisy in zip(rows, clean_tokens, perturbed_tokens):
            edit_distance = levenshtein(clean, noisy)
            metric_rows.append(
                {
                    "id": row.get("id", ""),
                    "language": row.get("language", ""),
                    "perturbation": name,
                    "clean_tokens": len(clean),
                    "perturbed_tokens": len(noisy),
                    "edit_distance": edit_distance,
                    "ued": edit_distance / max(1, len(clean)),
                    "normalized_edit_distance": edit_distance / max(1, len(clean), len(noisy)),
                    "reference": pick_text(row),
                }
            )
            sequence_rows.append(
                {
                    "id": row.get("id", ""),
                    "language": row.get("language", ""),
                    "perturbation": name,
                    "clean_tokens": clean,
                    "perturbed_tokens": noisy,
                }
            )
            histograms[name].update(int(token) for token in noisy)

    hist_rows = []
    for condition, counts in sorted(histograms.items()):
        for token, count in counts.most_common():
            hist_rows.append(
                {
                    "condition": condition,
                    "token": int(token),
                    "count": int(count),
                    "frequency": count / max(1, sum(counts.values())),
                }
            )

    write_csv(output_dir / "lfq_eval_item_metrics.csv", metric_rows)
    write_csv(output_dir / "token_usage_histogram.csv", hist_rows)
    with (output_dir / "token_sequences.jsonl").open("w", encoding="utf-8") as handle:
        for row in sequence_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(metric_rows, histograms, int(model.config.quantize_vocab_size or 0))
    summary.update(
        {
            "checkpoint": str(checkpoint),
            "tokenizer_dir": str(tokenizer_dir),
            "manifest": str(cfg["manifest"]),
            "output_dir": str(output_dir),
            "seed": seed,
            "device": device,
            "batch_size": batch_size,
        }
    )
    with (output_dir / "lfq_eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (output_dir / "command.txt").open("w", encoding="utf-8") as handle:
        handle.write(" ".join(sys.argv) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
