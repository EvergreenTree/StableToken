#!/usr/bin/env python
"""Config-driven StableToken experiment runner.

The runner focuses on inference-first reviewer concerns:
- checkpoint sanity and token rate
- UED/nUED under controlled corruptions
- pseudo-streaming/chunking disagreement
- token distribution drift
- latency/RTF

It writes lightweight JSON/CSV/PNG artifacts that can be committed as an audit
trail. Large audio, downloaded datasets, and checkpoints remain ignored.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from huggingface_hub import snapshot_download
from matplotlib import pyplot as plt
from transformers import WhisperFeatureExtractor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.modeling_whisper import WhisperLFQEncoder

SAMPLE_RATE = 16_000


@dataclass
class AudioItem:
    id: str
    path: str | None
    audio: np.ndarray | None
    sample_rate: int
    language: str
    source: str
    reference: str = ""


def run_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def ensure_checkpoint(cfg: dict[str, Any]) -> Path:
    local_dir = REPO_ROOT / cfg.get("local_dir", "checkpoints/StableToken")
    tokenizer_config = local_dir / "tokenizer" / "config.json"
    tokenizer_weights = local_dir / "tokenizer" / "model.safetensors"
    if tokenizer_config.exists() and tokenizer_weights.exists():
        return local_dir

    repo_id = cfg.get("repo_id", "tencent/StableToken")
    revision = cfg.get("revision")
    allow_patterns = cfg.get("allow_patterns") or ["tokenizer/*"]
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )
    return local_dir


def load_tokenizer_model(checkpoint_root: Path, device: str):
    tokenizer_path = checkpoint_root / "tokenizer"
    model = WhisperLFQEncoder.from_pretrained(str(tokenizer_path)).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(tokenizer_path))
    return model, feature_extractor


def read_jsonl_items(source: dict[str, Any]) -> list[AudioItem]:
    path = Path(source["path"])
    max_items = source.get("max_items")
    language = source.get("language", "")
    items: list[AudioItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_items is not None and len(items) >= int(max_items):
                break
            row = json.loads(line)
            audio_path = row.get("audio") or row.get("path")
            ref = row.get("reference") or row.get("text") or row.get("sentence") or ""
            item_lang = source.get("language") or row.get("language") or language
            items.append(
                AudioItem(
                    id=f"{source['name']}:{idx}",
                    path=audio_path,
                    audio=None,
                    sample_rate=SAMPLE_RATE,
                    language=item_lang,
                    source=source["name"],
                    reference=ref,
                )
            )
    return items


def read_pickle_items(source: dict[str, Any]) -> list[AudioItem]:
    path = Path(source["path"])
    max_items = source.get("max_items")
    language = source.get("language", "")
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    items: list[AudioItem] = []
    for idx, row in enumerate(raw):
        if max_items is not None and len(items) >= int(max_items):
            break
        items.append(
            AudioItem(
                id=f"{source['name']}:{idx}",
                path=None,
                audio=np.asarray(row["audio"], dtype=np.float32),
                sample_rate=int(row.get("sr", SAMPLE_RATE)),
                language=source.get("language") or row.get("language", language),
                source=source["name"],
                reference=row.get("ref", ""),
            )
        )
    return items


def load_items(cfg: dict[str, Any]) -> list[AudioItem]:
    items: list[AudioItem] = []
    for source in cfg.get("data", {}).get("sources", []):
        stype = source.get("type")
        if stype == "jsonl":
            items.extend(read_jsonl_items(source))
        elif stype == "pickle":
            items.extend(read_pickle_items(source))
        else:
            raise ValueError(f"Unsupported data source type: {stype}")
    return items


def load_audio_item(item: AudioItem) -> tuple[np.ndarray, int]:
    if item.audio is not None:
        wav = np.asarray(item.audio, dtype=np.float32)
        sr = item.sample_rate
    else:
        if item.path is None:
            raise ValueError(f"Item {item.id} has no path or audio array")
        try:
            wav, sr = sf.read(item.path, dtype="float32", always_2d=False)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
        except Exception:
            tensor, sr = torchaudio.load(item.path)
            if tensor.ndim == 2:
                tensor = tensor.mean(dim=0)
            wav = tensor.cpu().numpy().astype(np.float32)
    if sr != SAMPLE_RATE:
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE).squeeze(0).numpy()
        sr = SAMPLE_RATE
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return wav.astype(np.float32), sr


def fit_duration(wav: np.ndarray, seconds: float) -> np.ndarray:
    n = max(1, int(round(seconds * SAMPLE_RATE)))
    if wav.shape[0] >= n:
        return wav[:n].copy()
    out = np.zeros(n, dtype=np.float32)
    out[: wav.shape[0]] = wav
    return out


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def scale_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if noise.shape[0] < clean.shape[0]:
        repeats = int(math.ceil(clean.shape[0] / max(1, noise.shape[0])))
        noise = np.tile(noise, repeats)
    noise = noise[: clean.shape[0]].astype(np.float32)
    noise = noise - float(np.mean(noise))
    target = rms(clean) / (10.0 ** (snr_db / 20.0))
    return noise * (target / (rms(noise) + 1e-9))


def colored_noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n).astype(np.float32)
    if kind == "gaussian":
        return white

    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    spectrum = np.fft.rfft(white)
    scale = np.ones_like(freqs)
    nonzero = freqs > 0
    if kind == "pink":
        scale[nonzero] = 1.0 / np.sqrt(freqs[nonzero])
    elif kind == "brown":
        scale[nonzero] = 1.0 / freqs[nonzero]
    else:
        raise ValueError(f"Unknown colored noise: {kind}")
    scale[~nonzero] = 0.0
    out = np.fft.irfft(spectrum * scale, n=n).astype(np.float32)
    out = out - float(np.mean(out))
    return out / (rms(out) + 1e-9)


def bit_crush(wav: np.ndarray, bit_depth: int) -> np.ndarray:
    levels = float(2 ** max(1, bit_depth - 1))
    return np.clip(np.round(wav * levels) / levels, -1.0, 1.0).astype(np.float32)


def random_dropout(wav: np.ndarray, drop_prob: float, max_drop_ms: float, rng: np.random.Generator) -> np.ndarray:
    out = wav.copy()
    frame = max(1, int(SAMPLE_RATE * max_drop_ms / 1000.0))
    pos = 0
    while pos < out.shape[0]:
        if rng.random() < drop_prob:
            length = int(rng.integers(1, frame + 1))
            out[pos : min(out.shape[0], pos + length)] = 0.0
        pos += frame
    return out


def telephone_bandpass(wav: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(wav).unsqueeze(0)
    x = torchaudio.functional.highpass_biquad(x, SAMPLE_RATE, 300.0)
    x = torchaudio.functional.lowpass_biquad(x, SAMPLE_RATE, 3400.0)
    return x.squeeze(0).numpy().astype(np.float32)


def eq_tilt(wav: np.ndarray, gain_db: float) -> np.ndarray:
    x = torch.from_numpy(wav).unsqueeze(0)
    low = torchaudio.functional.lowpass_biquad(x, SAMPLE_RATE, 1000.0)
    high = x - low
    high = high * (10.0 ** (gain_db / 20.0))
    out = low + high
    return torch.clamp(out, -1.0, 1.0).squeeze(0).numpy().astype(np.float32)


def simple_reverb(wav: np.ndarray, decay: float, delay_ms: float) -> np.ndarray:
    delay = max(1, int(SAMPLE_RATE * delay_ms / 1000.0))
    ir_len = min(SAMPLE_RATE, delay * 8)
    ir = np.zeros(ir_len, dtype=np.float32)
    ir[0] = 1.0
    amp = decay
    pos = delay
    while pos < ir_len:
        ir[pos] = amp
        amp *= decay
        pos += delay
    out = np.convolve(wav, ir, mode="full")[: wav.shape[0]]
    peak = max(1.0, float(np.max(np.abs(out))))
    return (out / peak).astype(np.float32)


def match_length(wav: np.ndarray, target_samples: int) -> np.ndarray:
    if wav.shape[0] > target_samples:
        return wav[:target_samples].astype(np.float32)
    if wav.shape[0] < target_samples:
        return np.pad(wav, (0, target_samples - wav.shape[0])).astype(np.float32)
    return wav.astype(np.float32)


def codec_roundtrip(wav: np.ndarray, fmt: str, bitrate: str) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("codec corruption requires ffmpeg on PATH")
    fmt = fmt.lower()
    suffix = ".m4a" if fmt == "aac" else f".{fmt}"
    with tempfile.TemporaryDirectory(prefix="stabletoken_codec_") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.wav"
        encoded_path = tmp_path / f"encoded{suffix}"
        decoded_path = tmp_path / "decoded.wav"
        sf.write(input_path, wav, SAMPLE_RATE)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-b:a",
                bitrate,
                str(encoded_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(encoded_path),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                str(decoded_path),
            ],
            check=True,
        )
        out, sr = sf.read(decoded_path, dtype="float32", always_2d=False)
    if out.ndim == 2:
        out = out.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav_t = torch.from_numpy(out).unsqueeze(0)
        out = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE).squeeze(0).numpy()
    return match_length(out, wav.shape[0])


def corrupt_audio(
    wav: np.ndarray,
    spec: dict[str, Any],
    rng: np.random.Generator,
    distractors: list[np.ndarray] | None = None,
) -> np.ndarray:
    ctype = spec.get("type", "clean")
    if ctype == "clean":
        return wav.copy()
    if ctype in {"gaussian", "pink", "brown"}:
        noise = colored_noise(ctype, wav.shape[0], rng)
        return np.clip(wav + scale_noise(wav, noise, float(spec["snr_db"])), -1.0, 1.0).astype(np.float32)
    if ctype == "bit_crush":
        return bit_crush(wav, int(spec["bit_depth"]))
    if ctype == "clipping":
        threshold = float(spec.get("threshold", 0.5))
        return np.clip(wav, -threshold, threshold).astype(np.float32)
    if ctype == "dropout":
        return random_dropout(
            wav,
            float(spec.get("drop_prob", 0.05)),
            float(spec.get("max_drop_ms", 120)),
            rng,
        )
    if ctype == "telephone_bandpass":
        return telephone_bandpass(wav)
    if ctype == "eq_tilt":
        return eq_tilt(wav, float(spec.get("gain_db", -8)))
    if ctype == "reverb":
        return simple_reverb(wav, float(spec.get("decay", 0.35)), float(spec.get("delay_ms", 45)))
    if ctype == "codec":
        return codec_roundtrip(wav, str(spec.get("format", "mp3")), str(spec.get("bitrate", "32k")))
    if ctype == "babble":
        if not distractors:
            raise ValueError("babble requires distractor audio")
        speakers = max(1, int(spec.get("speakers", 4)))
        selected = [random.choice(distractors) for _ in range(speakers)]
        max_len = max(x.shape[0] for x in selected)
        mix = np.zeros(max_len, dtype=np.float32)
        for other in selected:
            mix[: other.shape[0]] += other
        mix = mix / float(speakers)
        noise = scale_noise(wav, mix, float(spec.get("snr_db", 10)))
        return np.clip(wav + noise, -1.0, 1.0).astype(np.float32)
    if ctype == "competing_speech":
        if not distractors:
            raise ValueError("competing_speech requires distractor audio")
        other = random.choice(distractors)
        noise = scale_noise(wav, other, float(spec.get("snr_db", 16)))
        return np.clip(wav + noise, -1.0, 1.0).astype(np.float32)
    raise ValueError(f"Unsupported corruption: {ctype}")


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
) -> list[list[int]]:
    out: list[list[int]] = []
    stride = feature_stride_samples(model, feature_extractor)
    for start in range(0, len(arrays), batch_size):
        batch = arrays[start : start + batch_size]
        features = feature_extractor(
            batch,
            sampling_rate=SAMPLE_RATE,
            return_attention_mask=True,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=stride,
        )
        features = features.to(device)
        with torch.inference_mode():
            tokens = model(**features, require_only_quantized_token=True)
        mask = valid_token_mask(features, model, tokens.shape)
        for i in range(tokens.shape[0]):
            out.append(tokens[i][mask[i]].detach().cpu().long().tolist())
    return out


def tokenize_long_audio_policy(
    model,
    feature_extractor,
    wav: np.ndarray,
    device: str,
    batch_size: int,
    window_seconds: float,
    stride_seconds: float,
    max_windows: int | None = None,
) -> list[dict[str, Any]]:
    total_seconds = wav.shape[0] / SAMPLE_RATE
    starts: list[float] = []
    cur = 0.0
    while cur < total_seconds:
        starts.append(cur)
        if max_windows is not None and len(starts) >= max_windows:
            break
        if cur + window_seconds >= total_seconds:
            break
        cur += stride_seconds

    windows: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for start_sec in starts:
        start = int(round(start_sec * SAMPLE_RATE))
        end = min(wav.shape[0], start + int(round(window_seconds * SAMPLE_RATE)))
        segment = wav[start:end]
        windows.append(segment.astype(np.float32))
        metas.append({"start_seconds": start_sec, "end_seconds": end / SAMPLE_RATE})

    tokens_list = tokenize_arrays(model, feature_extractor, windows, device, batch_size)
    for meta, tokens in zip(metas, tokens_list):
        meta["tokens"] = tokens
    return metas


def token_bins(windows: list[dict[str, Any]], frame_rate: float, mode: str) -> dict[int, int]:
    bucket: dict[int, list[int]] = defaultdict(list)
    for win in windows:
        start_bin = int(round(float(win["start_seconds"]) * frame_rate))
        for idx, token in enumerate(win["tokens"]):
            bucket[start_bin + idx].append(int(token))
    reduced: dict[int, int] = {}
    for key, vals in bucket.items():
        if mode == "first":
            reduced[key] = vals[0]
        elif mode == "majority":
            reduced[key] = Counter(vals).most_common(1)[0][0]
        else:
            raise ValueError(mode)
    return reduced


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


def ued(a: list[int], b: list[int]) -> float:
    return levenshtein(a, b) / max(1, len(a))


def compare_bins(ref: dict[int, int], hyp: dict[int, int], selected: set[int] | None = None) -> dict[str, Any]:
    keys = sorted((set(ref) & set(hyp)) if selected is None else (set(ref) & set(hyp) & selected))
    ref_seq = [ref[k] for k in keys]
    hyp_seq = [hyp[k] for k in keys]
    mismatches = sum(int(x != y) for x, y in zip(ref_seq, hyp_seq))
    return {
        "tokens": len(keys),
        "mismatch_rate": mismatches / max(1, len(keys)),
        "ued": ued(ref_seq, hyp_seq),
        "edit_distance": levenshtein(ref_seq, hyp_seq),
    }


def entropy_from_counts(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        p = value / total
        entropy -= p * math.log2(p)
    return entropy


def kl_divergence(p_counts: Counter[int], q_counts: Counter[int], vocab_size: int, alpha: float = 1e-6) -> float:
    p_total = sum(p_counts.values()) + alpha * vocab_size
    q_total = sum(q_counts.values()) + alpha * vocab_size
    kl = 0.0
    for token in range(vocab_size):
        p = (p_counts.get(token, 0) + alpha) / p_total
        q = (q_counts.get(token, 0) + alpha) / q_total
        kl += p * math.log2(p / q)
    return kl


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_bar(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str, ylabel: str) -> None:
    if not rows:
        return
    labels = [str(row[x_key]) for row in rows]
    values = [float(row[y_key]) for row in rows]
    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    plt.bar(labels, values, color="#2f6f73")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def plot_zipf(path: Path, zipf_rows: list[dict[str, Any]]) -> None:
    if not zipf_rows:
        return
    groups = sorted({str(row["group"]) for row in zipf_rows})
    plt.figure(figsize=(7, 4.5))
    for group in groups:
        rows = [row for row in zipf_rows if row["group"] == group]
        ranks = [int(row["rank"]) for row in rows]
        freqs = [float(row["frequency"]) for row in rows]
        plt.loglog(ranks, freqs, marker=".", linewidth=1.2, label=group)
    plt.xlabel("Token rank")
    plt.ylabel("Frequency")
    plt.title("Token Zipf curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def run_sanity(cfg, run_dir: Path, model, feature_extractor, items, device: str, batch_size: int) -> dict[str, Any]:
    spec = cfg.get("sanity", {})
    expected_vocab = int(spec.get("expected_vocab_size", 8192))
    bits_per_token = math.log2(expected_vocab)
    target_seconds = float(spec.get("target_duration_seconds", 30.0))
    clean_arrays: list[np.ndarray] = []
    noisy_arrays: list[np.ndarray] = []
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    loaded = []
    for item in items:
        wav, _ = load_audio_item(item)
        wav = fit_duration(wav, target_seconds)
        loaded.append((item, wav))
        clean_arrays.append(wav)
        noisy_arrays.append(corrupt_audio(wav, spec.get("corruption", {"type": "gaussian", "snr_db": 25}), rng))

    if not loaded:
        silence = np.zeros(int(target_seconds * SAMPLE_RATE), dtype=np.float32)
        clean_arrays.append(silence)
        noisy_arrays.append(corrupt_audio(silence + 1e-5, spec.get("corruption", {"type": "gaussian", "snr_db": 25}), rng))

    clean_tokens = tokenize_arrays(model, feature_extractor, clean_arrays, device, batch_size)
    noisy_tokens = tokenize_arrays(model, feature_extractor, noisy_arrays, device, batch_size)
    rows = []
    for idx, clean in enumerate(clean_tokens):
        noise = noisy_tokens[idx]
        duration = len(clean_arrays[idx]) / SAMPLE_RATE
        local_ued = ued(clean, noise)
        rows.append(
            {
                "item": loaded[idx][0].id if idx < len(loaded) else "synthetic",
                "duration_seconds": duration,
                "clean_tokens": len(clean),
                "noisy_tokens": len(noise),
                "token_rate_hz": len(clean) / duration,
                "vocab_size": expected_vocab,
                "bits_per_token": bits_per_token,
                "bitrate_bps": bits_per_token * (len(clean) / duration),
                "ued": local_ued,
                "nued": local_ued / bits_per_token,
            }
        )
    summary = {
        "mean_token_rate_hz": float(np.mean([row["token_rate_hz"] for row in rows])),
        "mean_clean_tokens": float(np.mean([row["clean_tokens"] for row in rows])),
        "mean_ued": float(np.mean([row["ued"] for row in rows])),
        "mean_nued": float(np.mean([row["nued"] for row in rows])),
        "expected_vocab_size": expected_vocab,
        "model_vocab_size": int(model.config.quantize_vocab_size),
        "pooling_kernel_size": model.config.pooling_kernel_size,
        "quantize_position": model.config.quantize_position,
        "num_voters": model.config.num_voters,
        "num_clean_input": model.config.num_clean_input,
    }
    write_csv(run_dir / "sanity_metrics.csv", rows)
    write_json(run_dir / "sanity_metrics.json", {"summary": summary, "rows": rows})
    return summary


def make_long_audio(items: list[AudioItem], target_seconds: float) -> np.ndarray:
    chunks: list[np.ndarray] = []
    total = 0
    gap = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)
    for item in items:
        wav, _ = load_audio_item(item)
        chunks.extend([wav, gap])
        total += len(wav) + len(gap)
        if total / SAMPLE_RATE >= target_seconds:
            break
    if not chunks:
        return np.zeros(int(target_seconds * SAMPLE_RATE), dtype=np.float32)
    long_audio = np.concatenate(chunks).astype(np.float32)
    return fit_duration(long_audio, target_seconds)


def run_chunking(cfg, run_dir: Path, model, feature_extractor, items, device: str, batch_size: int) -> dict[str, Any]:
    spec = cfg.get("chunking", {})
    frame_rate = float(spec.get("frame_rate", 25.0))
    boundary_windows = [float(x) for x in spec.get("boundary_windows_seconds", [0.5, 1, 2, 5])]
    long_count = int(spec.get("long_audio_count", 1))
    target_seconds = float(spec.get("target_long_seconds", 75))
    policies = spec["policies"]
    all_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []

    for audio_idx in range(long_count):
        wav = make_long_audio(items[audio_idx:] + items[:audio_idx], target_seconds)
        policy_windows = {}
        for policy in policies:
            policy_windows[policy["name"]] = tokenize_long_audio_policy(
                model,
                feature_extractor,
                wav,
                device,
                batch_size,
                float(policy["window_seconds"]),
                float(policy["stride_seconds"]),
                policy.get("max_windows"),
            )

        ref_name = policies[0]["name"]
        ref_majority = token_bins(policy_windows[ref_name], frame_rate, "majority")
        cut_bins = {int(round(x * frame_rate)) for x in np.arange(30.0, target_seconds, 30.0)}
        for policy in policies[1:]:
            name = policy["name"]
            hyp_first = token_bins(policy_windows[name], frame_rate, "first")
            hyp_majority = token_bins(policy_windows[name], frame_rate, "majority")
            for mode, hyp in [("first", hyp_first), ("majority", hyp_majority)]:
                base = compare_bins(ref_majority, hyp)
                base.update({"audio_index": audio_idx, "policy": name, "aggregation": mode, "region": "overall"})
                all_rows.append(base)

                for seconds in boundary_windows:
                    radius = int(round(seconds * frame_rate))
                    selected = {
                        key
                        for key in ref_majority
                        if any(abs(key - cut) <= radius for cut in cut_bins)
                    }
                    row = compare_bins(ref_majority, hyp, selected)
                    row.update(
                        {
                            "audio_index": audio_idx,
                            "policy": name,
                            "aggregation": mode,
                            "region": f"boundary_pm_{seconds:g}s",
                        }
                    )
                    all_rows.append(row)

                thirds = {
                    "start_0_5s": lambda rel: rel < 5 * frame_rate,
                    "middle_5_25s": lambda rel: 5 * frame_rate <= rel < 25 * frame_rate,
                    "end_25_30s": lambda rel: rel >= 25 * frame_rate,
                }
                for region, pred in thirds.items():
                    selected = {key for key in ref_majority if pred(key % int(30 * frame_rate))}
                    row = compare_bins(ref_majority, hyp, selected)
                    row.update({"audio_index": audio_idx, "policy": name, "aggregation": mode, "region": region})
                    all_rows.append(row)

                bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 15)]
                common = sorted(set(ref_majority) & set(hyp))
                for lo, hi in bins:
                    selected = set()
                    for key in common:
                        dist = min(abs(key - cut) / frame_rate for cut in cut_bins) if cut_bins else 999
                        if lo <= dist < hi:
                            selected.add(key)
                    row = compare_bins(ref_majority, hyp, selected)
                    row.update(
                        {
                            "audio_index": audio_idx,
                            "policy": name,
                            "aggregation": mode,
                            "distance_bin": f"{lo:g}-{hi:g}s",
                        }
                    )
                    distance_rows.append(row)

    write_csv(run_dir / "chunking_metrics.csv", all_rows)
    write_csv(run_dir / "chunking_distance_metrics.csv", distance_rows)
    write_json(run_dir / "chunking_metrics.json", {"rows": all_rows, "distance_rows": distance_rows})
    plot_bar(
        run_dir / "plots" / "chunking_boundary_instability.png",
        [row for row in all_rows if row["aggregation"] == "majority" and row["region"] == "overall"],
        "policy",
        "mismatch_rate",
        "Chunking mismatch vs non-overlap",
        "Mismatch rate",
    )
    return {
        "rows": len(all_rows),
        "distance_rows": len(distance_rows),
        "mean_overall_mismatch": float(
            np.mean([row["mismatch_rate"] for row in all_rows if row["region"] == "overall"])
        )
        if all_rows
        else 0.0,
    }


def run_degradation(cfg, run_dir: Path, model, feature_extractor, items, device: str, batch_size: int) -> dict[str, Any]:
    spec = cfg.get("degradation", {})
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    expected_vocab = int(model.config.quantize_vocab_size or 8192)
    bits_per_token = math.log2(expected_vocab)
    loaded = [(item, load_audio_item(item)[0]) for item in items]
    distractors = [wav for _, wav in loaded]
    clean_tokens = tokenize_arrays(model, feature_extractor, [wav for _, wav in loaded], device, batch_size)
    rows: list[dict[str, Any]] = []
    for corruption in spec.get("corruptions", []):
        corrupted_arrays = []
        for _, wav in loaded:
            corrupted_arrays.append(corrupt_audio(wav, corruption, rng, distractors=distractors))
        corrupted_tokens = tokenize_arrays(model, feature_extractor, corrupted_arrays, device, batch_size)
        for (item, wav), clean, corrupt in zip(loaded, clean_tokens, corrupted_tokens):
            value = ued(clean, corrupt)
            rows.append(
                {
                    "item": item.id,
                    "source": item.source,
                    "language": item.language,
                    "duration_seconds": len(wav) / SAMPLE_RATE,
                    "corruption": corruption.get("name", corruption["type"]),
                    "corruption_type": corruption["type"],
                    "snr_db": corruption.get("snr_db", ""),
                    "intensity": corruption.get("bit_depth", corruption.get("threshold", "")),
                    "clean_tokens": len(clean),
                    "corrupt_tokens": len(corrupt),
                    "ued": value,
                    "nued": value / bits_per_token,
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
                "mean_ued": float(np.mean([v["ued"] for v in vals])),
                "mean_nued": float(np.mean([v["nued"] for v in vals])),
            }
        )
    write_csv(run_dir / "degradation_item_metrics.csv", rows)
    write_csv(run_dir / "degradation_summary.csv", summary_rows)
    write_json(run_dir / "degradation_metrics.json", {"summary": summary_rows, "rows": rows})
    plot_bar(
        run_dir / "plots" / "degradation_mean_ued.png",
        summary_rows,
        "corruption",
        "mean_ued",
        "Mean UED by corruption",
        "UED",
    )
    return {"rows": len(rows), "summary": summary_rows}


def run_distribution(cfg, run_dir: Path, model, feature_extractor, items, device: str, batch_size: int) -> dict[str, Any]:
    spec = cfg.get("distribution", {})
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    vocab_size = int(model.config.quantize_vocab_size or 8192)
    loaded = [(item, load_audio_item(item)[0]) for item in items]
    rows = []
    zipf_rows: list[dict[str, Any]] = []
    counts_by_group: dict[str, Counter[int]] = {}
    transition_counts_by_group: dict[str, Counter[tuple[int, int]]] = {}

    for corruption in spec.get("corruptions", [{"name": "clean", "type": "clean"}]):
        arrays = [corrupt_audio(wav, corruption, rng, distractors=[x for _, x in loaded]) for _, wav in loaded]
        tokens_list = tokenize_arrays(model, feature_extractor, arrays, device, batch_size)
        grouped_tokens: dict[str, list[list[int]]] = defaultdict(list)
        for (item, _), tokens in zip(loaded, tokens_list):
            grouped_tokens[f"{item.language}:{corruption.get('name', corruption['type'])}"].append(tokens)

        for group, seqs in grouped_tokens.items():
            counts: Counter[int] = Counter()
            transitions: Counter[tuple[int, int]] = Counter()
            for seq in seqs:
                counts.update(seq)
                transitions.update(zip(seq[:-1], seq[1:]))
            counts_by_group[group] = counts
            transition_counts_by_group[group] = transitions
            total = sum(counts.values())
            for rank, (token, count) in enumerate(counts.most_common(), 1):
                zipf_rows.append(
                    {
                        "group": group,
                        "rank": rank,
                        "token": token,
                        "count": count,
                        "frequency": count / max(1, total),
                    }
                )
            rows.append(
                {
                    "group": group,
                    "total_tokens": total,
                    "unique_tokens": len(counts),
                    "unique_token_rate": len(counts) / vocab_size,
                    "dead_token_rate": 1.0 - (len(counts) / vocab_size),
                    "entropy_bits": entropy_from_counts(counts),
                    "transition_entropy_bits": entropy_from_counts(transitions),
                    "top_tokens": " ".join(str(t) for t, _ in counts.most_common(10)),
                }
            )

    kl_rows = []
    groups = sorted(counts_by_group)
    for i, left in enumerate(groups):
        for right in groups:
            if left == right:
                continue
            kl_rows.append(
                {
                    "left": left,
                    "right": right,
                    "kl_bits": kl_divergence(counts_by_group[left], counts_by_group[right], vocab_size),
                }
            )

    write_csv(run_dir / "distribution_summary.csv", rows)
    write_csv(run_dir / "distribution_kl.csv", kl_rows)
    write_csv(run_dir / "distribution_zipf.csv", zipf_rows)
    write_json(run_dir / "distribution_metrics.json", {"summary": rows, "kl": kl_rows, "zipf": zipf_rows})
    plot_bar(
        run_dir / "plots" / "distribution_entropy.png",
        rows,
        "group",
        "entropy_bits",
        "Token entropy by group",
        "Entropy (bits)",
    )
    plot_zipf(run_dir / "plots" / "distribution_zipf.png", zipf_rows)
    return {"groups": len(rows), "kl_pairs": len(kl_rows)}


def run_latency(cfg, run_dir: Path, model, feature_extractor, device: str) -> dict[str, Any]:
    spec = cfg.get("latency", {})
    durations = [float(x) for x in spec.get("durations_seconds", [1, 10, 30])]
    batch_sizes = [int(x) for x in spec.get("batch_sizes", [1])]
    warmup = int(spec.get("warmup", 1))
    repeats = int(spec.get("repeats", 3))
    rows = []
    for duration in durations:
        wav = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
        wav[int(0.1 * SAMPLE_RATE) : int(0.5 * SAMPLE_RATE)] = 0.01
        for batch_size in batch_sizes:
            arrays = [wav.copy() for _ in range(batch_size)]
            for _ in range(warmup):
                tokenize_arrays(model, feature_extractor, arrays, device, batch_size)
            times = []
            mem_before = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            for _ in range(repeats):
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                tokens = tokenize_arrays(model, feature_extractor, arrays, device, batch_size)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            mem_peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
            mean_latency = float(np.mean(times))
            audio_seconds = duration * batch_size
            rows.append(
                {
                    "duration_seconds": duration,
                    "batch_size": batch_size,
                    "mean_latency_seconds": mean_latency,
                    "rtf": mean_latency / audio_seconds,
                    "throughput_audio_seconds_per_second": audio_seconds / mean_latency,
                    "tokens_per_item": len(tokens[0]) if tokens else 0,
                    "peak_memory_mb": mem_peak / (1024 * 1024),
                    "memory_before_mb": mem_before / (1024 * 1024),
                }
            )
    write_csv(run_dir / "latency_metrics.csv", rows)
    write_json(run_dir / "latency_metrics.json", {"rows": rows})
    plot_bar(
        run_dir / "plots" / "latency_rtf.png",
        [{**row, "label": f"{row['duration_seconds']:g}s_b{row['batch_size']}"} for row in rows],
        "label",
        "rtf",
        "Latency RTF",
        "RTF",
    )
    return {"rows": len(rows), "mean_rtf": float(np.mean([row["rtf"] for row in rows])) if rows else 0.0}


def write_run_scaffold(run_dir: Path, cfg: dict[str, Any], config_path: Path, checkpoint_dir: Path) -> None:
    metadata = {
        "timestamp_utc": timestamp(),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_cmd(["git", "status", "--short"]),
        "command": " ".join(sys.argv),
        "config_path": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint": cfg.get("checkpoint", {}),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    write_json(run_dir / "run_meta.json", metadata)
    write_json(run_dir / "config.resolved.json", cfg)
    (run_dir / "command.txt").write_text(metadata["command"] + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-id", default=None, help="Optional output run id.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    set_seed(int(cfg.get("seed", 42)))
    device = select_device(str(cfg.get("device", "auto")))
    output_root = REPO_ROOT / cfg.get("output_root", "experiments/runs")
    run_name = args.run_id or f"{cfg.get('run_name', config_path.stem)}_{timestamp()}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = ensure_checkpoint(cfg.get("checkpoint", {}))
    write_run_scaffold(run_dir, cfg, config_path, checkpoint_dir)

    model, feature_extractor = load_tokenizer_model(checkpoint_dir, device)
    items = load_items(cfg) if cfg.get("data") else []
    batch_size = int(cfg.get("batch_size", 8))
    summaries = {}

    t0 = time.perf_counter()
    for task in cfg.get("tasks", []):
        if task == "sanity":
            summaries[task] = run_sanity(cfg, run_dir, model, feature_extractor, items, device, batch_size)
        elif task == "chunking":
            summaries[task] = run_chunking(cfg, run_dir, model, feature_extractor, items, device, batch_size)
        elif task == "degradation":
            summaries[task] = run_degradation(cfg, run_dir, model, feature_extractor, items, device, batch_size)
        elif task == "distribution":
            summaries[task] = run_distribution(cfg, run_dir, model, feature_extractor, items, device, batch_size)
        elif task == "latency":
            summaries[task] = run_latency(cfg, run_dir, model, feature_extractor, device)
        else:
            raise ValueError(f"Unsupported task: {task}")
    summaries["elapsed_seconds"] = time.perf_counter() - t0
    write_json(run_dir / "summary.json", summaries)
    print(json.dumps({"run_dir": str(run_dir), "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
