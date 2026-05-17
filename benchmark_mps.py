import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Must be set before importing torch. It allows unsupported MPS ops to fall
# back to CPU instead of aborting the benchmark.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "third_party" / "Matcha-TTS"))

import torch
import torchaudio
from transformers import WhisperFeatureExtractor

from src.model.modeling_whisper import WhisperLFQEncoder
from src.utils.flow_inference import AudioDecoder
from src.utils.utils import extract_speech_token, speech_token_to_wav


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark StableToken inference on CPU, CUDA, or MPS.")
    parser.add_argument("--model_path", default="checkpoints/StableToken")
    parser.add_argument("--audio_path", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, etc.")
    parser.add_argument("--duration", type=float, default=5.0, help="Generated benchmark WAV duration in seconds.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--flow_steps", type=int, default=None, help="Override decoder flow steps.")
    parser.add_argument("--decoder_autocast", action="store_true", help="Force decoder autocast.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def resolve_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device):
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def timed(device, fn):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    return time.perf_counter() - start, result


def make_benchmark_audio(path, duration, sample_rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(duration * sample_rate)
    t = torch.arange(frames, dtype=torch.float32) / sample_rate
    waveform = (
        0.16 * torch.sin(2 * math.pi * 220 * t)
        + 0.08 * torch.sin(2 * math.pi * 440 * t)
        + 0.04 * torch.sin(2 * math.pi * 880 * t)
    ).unsqueeze(0)
    torchaudio.save(str(path), waveform, sample_rate)
    return path


def percentile(values, pct):
    values = sorted(values)
    index = (len(values) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def summarize(values):
    return {
        "min_s": min(values),
        "median_s": percentile(values, 0.5),
        "max_s": max(values),
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model_path = Path(args.model_path)
    audio_path = Path(args.audio_path) if args.audio_path else ROOT_DIR / "benchmarks" / f"tone_{args.duration:g}s.wav"
    if args.audio_path is None:
        make_benchmark_audio(audio_path, args.duration)

    print(f"device={device}")
    print(f"model_path={model_path}")
    print(f"audio_path={audio_path}")
    print(f"torch={torch.__version__}")
    print(f"mps_available={torch.backends.mps.is_available()}")

    tokenizer_path = model_path / "tokenizer"
    decoder_path = model_path / "decoder"

    load_s, loaded = timed(
        device,
        lambda: (
            WhisperLFQEncoder.from_pretrained(str(tokenizer_path)).eval().to(device),
            WhisperFeatureExtractor.from_pretrained(str(tokenizer_path)),
            AudioDecoder(
                config_path=str(decoder_path / "config.yaml"),
                flow_ckpt_path=str(decoder_path / "flow.pt"),
                hift_ckpt_path=str(decoder_path / "hift.pt"),
                device=device,
                flow_steps=args.flow_steps,
            ),
        ),
    )
    tokenizer, feature_extractor, audio_decoder = loaded

    token_runs = []
    tokens = None
    for i in range(args.warmup + args.repeats):
        elapsed, result = timed(
            device,
            lambda: extract_speech_token(
                tokenizer,
                feature_extractor,
                [str(audio_path)],
                batch_size=args.batch_size,
                device=device,
            ),
        )
        tokens = result[0]
        if i >= args.warmup:
            token_runs.append(elapsed)

    decode_runs = []
    output_rate = None
    output_samples = None
    for i in range(args.warmup + args.repeats):
        elapsed, result = timed(
            device,
            lambda: speech_token_to_wav(
                audio_decoder,
                tokens,
                autocast_enabled=True if args.decoder_autocast else None,
            ),
        )
        speech, output_rate = result
        output_samples = speech.shape[-1]
        if i >= args.warmup:
            decode_runs.append(elapsed)

    input_samples, input_rate = torchaudio.info(str(audio_path)).num_frames, torchaudio.info(str(audio_path)).sample_rate
    audio_seconds = input_samples / input_rate
    output_seconds = output_samples / output_rate
    results = {
        "device": device,
        "torch": torch.__version__,
        "audio_path": str(audio_path),
        "input_seconds": audio_seconds,
        "tokens": len(tokens),
        "output_seconds": output_seconds,
        "decoder_autocast": args.decoder_autocast or torch.device(device).type == "cuda",
        "flow_steps": args.flow_steps or getattr(audio_decoder.flow, "inference_steps", None),
        "load_s": load_s,
        "tokenization": summarize(token_runs),
        "reconstruction": summarize(decode_runs),
        "tokenization_realtime_factor": summarize([run / audio_seconds for run in token_runs]),
        "reconstruction_realtime_factor": summarize([run / output_seconds for run in decode_runs]),
    }

    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
