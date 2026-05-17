#!/usr/bin/env python
"""Config-driven ASR degradation evaluation.

This is a downstream smoke hook for the same corruptions used by the
StableToken tokenizer experiments. By default it evaluates a normal Whisper ASR
model. With `asr.use_stabletoken_encoder=true`, it replaces the Whisper encoder
with the released StableToken LFQ encoder so generation goes through the
discrete-token bottleneck before the Whisper decoder.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import torch
import yaml
from transformers import WhisperForConditionalGeneration, WhisperProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_experiment import (  # noqa: E402
    SAMPLE_RATE,
    corrupt_audio,
    ensure_checkpoint,
    feature_stride_samples,
    load_audio_item,
    load_items,
    load_tokenizer_model,
    write_csv,
    write_json,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_text(text: str, language: str) -> str:
    text = text.lower().strip()
    if language == "zh":
        text = re.sub(r"\s+", "", text)
        return "".join(ch for ch in text if ch not in string.punctuation and ch not in "，。！？；：“”‘’（）《》、")
    table = str.maketrans("", "", string.punctuation + "«»“”‘’…")
    text = text.translate(table)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_forced_decoder_ids(processor: WhisperProcessor, language: str) -> list[list[int]]:
    whisper_language = {"fr": "french", "zh": "chinese"}.get(language, language)
    try:
        return processor.get_decoder_prompt_ids(language=whisper_language, task="transcribe")
    except Exception:
        return []


def transcribe_batch(
    model,
    processor,
    arrays: list[np.ndarray],
    language: str,
    device: str,
    max_new_tokens: int,
    pad_to_multiple_of: int | None = None,
) -> list[str]:
    inputs = processor(
        arrays,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding="longest",
        return_attention_mask=True,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    input_features = inputs.input_features.to(device)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    forced_decoder_ids = make_forced_decoder_ids(processor, language)
    with torch.inference_mode():
        if pad_to_multiple_of is not None:
            encoder_outputs = model.model.encoder(
                input_features,
                attention_mask=attention_mask,
                return_dict=True,
            )
            pred_ids = model.generate(
                encoder_outputs=encoder_outputs,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=max_new_tokens,
            )
        else:
            pred_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=max_new_tokens,
            )
    return processor.batch_decode(pred_ids, skip_special_tokens=True)


def score_pair(reference: str, hypothesis: str, language: str) -> dict[str, float]:
    ref_norm = normalize_text(reference, language)
    hyp_norm = normalize_text(hypothesis, language)
    if not ref_norm:
        return {"wer": 0.0, "cer": 0.0}
    return {
        "wer": float(jiwer.wer(ref_norm, hyp_norm)),
        "cer": float(jiwer.cer(ref_norm, hyp_norm)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    run_id = args.run_id or f"{cfg.get('run_name', 'asr_degradation')}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = REPO_ROOT / cfg.get("output_root", "experiments/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    command = f"experiments/run_asr_degradation.py --config {args.config}"
    if args.run_id:
        command += f" --run-id {args.run_id}"
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    write_json(run_dir / "config.resolved.json", cfg)

    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    asr_cfg = cfg["asr"]
    model_id = asr_cfg.get("model_id", "openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    encoder_source = "native_whisper"
    pad_to_multiple_of = None
    if asr_cfg.get("use_stabletoken_encoder", False):
        checkpoint_dir = ensure_checkpoint(cfg.get("checkpoint", {}))
        stabletoken_encoder, stabletoken_feature_extractor = load_tokenizer_model(checkpoint_dir, device)
        model.model.encoder = stabletoken_encoder
        processor.feature_extractor = stabletoken_feature_extractor
        pad_to_multiple_of = feature_stride_samples(stabletoken_encoder, stabletoken_feature_extractor)
        encoder_source = f"stabletoken:{checkpoint_dir / 'tokenizer'}"
    model.eval()

    batch_size = int(cfg.get("batch_size", 4))
    max_new_tokens = int(asr_cfg.get("max_new_tokens", 128))
    items = load_items(cfg)
    loaded = [(item, load_audio_item(item)[0]) for item in items]
    distractors = [wav for _, wav in loaded]

    rows: list[dict[str, Any]] = []
    corruptions = cfg["degradation"]["corruptions"]
    start_time = time.perf_counter()
    for corruption in corruptions:
        corruption_name = corruption.get("name", corruption["type"])
        by_language: dict[str, list[tuple[Any, np.ndarray]]] = defaultdict(list)
        for item, wav in loaded:
            corrupted = corrupt_audio(wav, corruption, rng, distractors=distractors)
            by_language[item.language].append((item, corrupted))

        for language, pairs in sorted(by_language.items()):
            for offset in range(0, len(pairs), batch_size):
                batch = pairs[offset : offset + batch_size]
                preds = transcribe_batch(
                    model,
                    processor,
                    [wav for _, wav in batch],
                    language,
                    device,
                    max_new_tokens,
                    pad_to_multiple_of=pad_to_multiple_of,
                )
                for (item, wav), pred in zip(batch, preds):
                    scores = score_pair(item.reference, pred, language)
                    rows.append(
                        {
                            "item": item.id,
                            "source": item.source,
                            "language": language,
                            "duration_seconds": len(wav) / SAMPLE_RATE,
                            "corruption": corruption_name,
                            "corruption_type": corruption["type"],
                            "reference": item.reference,
                            "prediction": pred,
                            "wer": scores["wer"],
                            "cer": scores["cer"],
                        }
                    )

    summary_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["corruption"], row["language"])].append(row)
    for (corruption, language), vals in sorted(grouped.items()):
        summary_rows.append(
            {
                "corruption": corruption,
                "language": language,
                "items": len(vals),
                "mean_wer": float(np.mean([v["wer"] for v in vals])),
                "mean_cer": float(np.mean([v["cer"] for v in vals])),
            }
        )

    summary = {
        "asr": {
            "model_id": model_id,
            "encoder_source": encoder_source,
            "rows": len(rows),
            "summary": summary_rows,
        },
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    write_csv(run_dir / "asr_item_metrics.csv", rows)
    write_csv(run_dir / "asr_summary.csv", summary_rows)
    write_json(run_dir / "asr_metrics.json", {"summary": summary_rows, "rows": rows})
    write_json(run_dir / "summary.json", summary)
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
