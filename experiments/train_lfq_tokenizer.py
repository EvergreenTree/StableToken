#!/usr/bin/env python
"""Whisper + Voting-LFQ tokenizer training harness.

This harness recreates the published StableToken training shape closely enough
to run controlled ablation pilots:
- initialize from a Hugging Face Whisper checkpoint
- replace the encoder with the repo's `WhisperLFQEncoder`
- train ASR CE plus quantizer losses exposed by the encoder
- optionally pass a waveform-augmented noisy view into the LFQ branches

The released StableToken repo does not include its tokenizer trainer, so this is
kept explicit and config-driven rather than hidden behind notebook state.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.whisper.modeling_whisper import shift_tokens_right

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.configuration_whisper import WhisperVQConfig
from src.model.modeling_whisper import QuantizedBaseModelOutput, WhisperLFQEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def pick_text(row: dict[str, Any]) -> str:
    text = row.get("reference") or row.get("text") or row.get("sentence") or ""
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    return text


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    try:
        out, sr = sf.read(path, dtype="float32", always_2d=False)
        if out.ndim == 2:
            out = out.mean(axis=1)
    except Exception:
        wav, sr = torchaudio.load(path)
        if wav.ndim == 2:
            wav = wav.mean(dim=0)
        out = wav.cpu().numpy().astype(np.float32)
    if sr != sample_rate:
        wav_t = torch.from_numpy(out).unsqueeze(0)
        out = torchaudio.functional.resample(wav_t, sr, sample_rate).squeeze(0).numpy()
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def colored_noise(kind: str, n: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n).astype(np.float32)
    if kind == "gaussian":
        return white
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.fft.rfft(white)
    scale = np.ones_like(freqs)
    nonzero = freqs > 0
    if kind == "pink":
        scale[nonzero] = 1.0 / np.sqrt(freqs[nonzero])
    elif kind == "brown":
        scale[nonzero] = 1.0 / freqs[nonzero]
    else:
        raise ValueError(kind)
    scale[~nonzero] = 0.0
    out = np.fft.irfft(spectrum * scale, n=n).astype(np.float32)
    return out / (rms(out) + 1e-9)


def add_noise(clean: np.ndarray, kind: str, snr_db: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    noise = colored_noise(kind, clean.shape[0], sample_rate, rng)
    target = rms(clean) / (10.0 ** (snr_db / 20.0))
    noise = noise * (target / (rms(noise) + 1e-9))
    return np.clip(clean + noise, -1.0, 1.0).astype(np.float32)


def bit_crush(wav: np.ndarray, bit_depth: int) -> np.ndarray:
    levels = float(2 ** max(1, bit_depth - 1))
    return np.clip(np.round(wav * levels) / levels, -1.0, 1.0).astype(np.float32)


def augment_audio(wav: np.ndarray, aug_cfg: dict[str, Any], sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    if not aug_cfg.get("enabled", False):
        return wav.copy()
    choices = aug_cfg.get("choices") or [{"type": "gaussian", "min_snr_db": 20, "max_snr_db": 30}]
    choice = dict(random.choice(choices))
    ctype = choice["type"]
    if ctype in {"gaussian", "pink", "brown"}:
        snr = float(rng.uniform(float(choice["min_snr_db"]), float(choice["max_snr_db"])))
        return add_noise(wav, ctype, snr, sample_rate, rng)
    if ctype == "bit_crush":
        bit_depth = int(rng.integers(int(choice["min_bit_depth"]), int(choice["max_bit_depth"]) + 1))
        return bit_crush(wav, bit_depth)
    raise ValueError(f"Unsupported augmentation type: {ctype}")


class WhisperLFQForConditionalGeneration(WhisperForConditionalGeneration):
    """Whisper generation head with LFQ encoder loss added to CE."""

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        decoder_position_ids=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        noise_input_features=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                labels,
                self.config.pad_token_id,
                self.config.decoder_start_token_id,
            )

        if encoder_outputs is None:
            input_features = self.model._mask_input_features(input_features, attention_mask=attention_mask)
            encoder_outputs = self.model.encoder(
                input_features,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                noise_input_features=noise_input_features,
            )
        elif not isinstance(encoder_outputs, QuantizedBaseModelOutput):
            encoder_outputs = QuantizedBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
        lm_logits = self.proj_out(decoder_outputs.last_hidden_state)

        ce_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            ce_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        quantized_loss = encoder_outputs.quantized_loss
        if quantized_loss is None:
            quantized_loss = lm_logits.new_tensor(0.0)
        loss = ce_loss + quantized_loss if ce_loss is not None else quantized_loss

        if not return_dict:
            return (loss, lm_logits)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def build_vq_config(base_config, variant: dict[str, Any], train_cfg: dict[str, Any]) -> WhisperVQConfig:
    data = base_config.to_dict()
    data.update(
        {
            "quantize_position": int(variant["quantize_position"]),
            "pooling_position": int(variant.get("pooling_position", variant["quantize_position"])),
            "pooling_kernel_size": variant.get("pooling_kernel_size", 2),
            "pooling_type": variant.get("pooling_type", "avg"),
            "quantize_vocab_size": int(variant.get("quantize_vocab_size", 8192)),
            "num_voters": int(variant.get("num_voters", 5)),
            "num_clean_input": int(variant.get("num_clean_input", 3)),
            "quantize_commit_coefficient": float(train_cfg.get("commitment_loss_weight", 0.25)),
            "consensus_loss_weight": float(variant.get("consensus_loss_weight", train_cfg.get("consensus_loss_weight", 0.25))),
            "codebook_entropy_loss_weight": float(train_cfg.get("codebook_entropy_loss_weight", 1.0)),
            "sample_minimization_weight": 1.0,
            "batch_maximization_weight": 1.0,
        }
    )
    if data["quantize_position"] > data["encoder_layers"]:
        raise ValueError(
            f"quantize_position={data['quantize_position']} exceeds encoder_layers={data['encoder_layers']}"
        )
    return WhisperVQConfig(**data)


def build_model(base_model: str, variant: dict[str, Any], train_cfg: dict[str, Any]) -> WhisperLFQForConditionalGeneration:
    model = WhisperLFQForConditionalGeneration.from_pretrained(base_model)
    vq_config = build_vq_config(model.config, variant, train_cfg)
    lfq_encoder = WhisperLFQEncoder(vq_config)
    missing, unexpected = lfq_encoder.load_state_dict(model.model.encoder.state_dict(), strict=False)
    model.model.encoder = lfq_encoder
    model.config = vq_config
    model.model.config = vq_config
    model.config.use_cache = False
    print("LFQ encoder loaded", {"missing": len(missing), "unexpected": len(unexpected)})
    return model


@dataclass
class LFQDataCollator:
    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizer
    sample_rate: int
    augmentation: dict[str, Any]
    seed: int

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audios = [load_audio(f["audio"], self.sample_rate) for f in features]
        noisy = [augment_audio(w, self.augmentation, self.sample_rate, self.rng) for w in audios]
        texts = [pick_text(f) for f in features]

        clean_inputs = self.feature_extractor(
            audios,
            sampling_rate=self.sample_rate,
            return_attention_mask=True,
            return_tensors="pt",
            padding="longest",
        )
        noise_inputs = self.feature_extractor(
            noisy,
            sampling_rate=self.sample_rate,
            return_attention_mask=True,
            return_tensors="pt",
            padding="longest",
        )
        labels_batch = self.tokenizer(texts, padding=True, return_tensors="pt")
        labels = labels_batch.input_ids.masked_fill(labels_batch.attention_mask.ne(1), -100)
        if labels.shape[1] > 0 and (labels[:, 0] == self.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        return {
            "input_features": clean_inputs.input_features,
            "attention_mask": clean_inputs.attention_mask,
            "noise_input_features": noise_inputs.input_features,
            "labels": labels,
        }


def load_json_dataset(path: str, max_items: int | None):
    ds = load_dataset("json", data_files=path, split="train")
    if max_items is not None and max_items < len(ds):
        ds = ds.select(range(max_items))
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default=None, help="Variant name to run. Defaults to top-level `variant`.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without downloading or training.")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    set_seed(int(cfg.get("seed", 42)))
    train_cfg = dict(cfg["training"])
    data_cfg = cfg["data"]

    variant = cfg.get("variant")
    if args.variant:
        variants = []
        for group in cfg.get("matrix", {}).values():
            variants.extend(group)
        matches = [v for v in variants if v["name"] == args.variant]
        if not matches:
            raise ValueError(f"Unknown variant: {args.variant}")
        variant = matches[0]
    if variant is None:
        raise ValueError("Config must define top-level `variant` or pass --variant")

    output_root = REPO_ROOT / cfg.get("output_root", "experiments/training_runs")
    output_dir = output_root / f"{cfg.get('run_name', 'train')}_{variant['name']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": cfg, "variant": variant}, handle, indent=2)

    if args.dry_run:
        print(json.dumps({"status": "dry_run_ok", "output_dir": str(output_dir), "variant": variant}, indent=2))
        return

    language = data_cfg.get("language", "en")
    base_model = train_cfg["base_model"]
    feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
    tokenizer = WhisperTokenizer.from_pretrained(base_model, language=language, task="transcribe")
    model = build_model(base_model, variant, train_cfg)

    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    aug_cfg = dict(train_cfg.get("augmentation", {}))
    if "augmentation_enabled" in variant:
        aug_cfg["enabled"] = bool(variant["augmentation_enabled"])

    train_ds = load_json_dataset(data_cfg["train_file"], data_cfg.get("max_train_items"))
    eval_ds = None
    if data_cfg.get("eval_file"):
        eval_ds = load_json_dataset(data_cfg["eval_file"], data_cfg.get("max_eval_items"))

    collator = LFQDataCollator(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        augmentation=aug_cfg,
        seed=int(cfg.get("seed", 42)),
    )

    args_train = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("grad_accum", 1)),
        max_steps=int(train_cfg.get("max_steps", -1)),
        learning_rate=float(train_cfg.get("learning_rate", 1.5e-5)),
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(train_cfg.get("grad_clip", 1.0)),
        bf16=bool(train_cfg.get("bf16", torch.cuda.is_available())),
        logging_steps=int(train_cfg.get("logging_steps", 25)),
        save_strategy=str(train_cfg.get("save_strategy", "steps" if train_cfg.get("save_model", True) else "no")),
        save_steps=int(train_cfg.get("save_steps", 1000)),
        save_total_limit=2,
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=int(cfg.get("seed", 42)),
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )
    train_output = trainer.train()
    metrics = {key: float(value) if isinstance(value, (int, float)) else value for key, value in train_output.metrics.items()}
    result = {
        "status": "trained",
        "output_dir": str(output_dir),
        "variant": variant,
        "base_model": base_model,
        "metrics": metrics,
        "saved_model": bool(train_cfg.get("save_model", True)),
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if train_cfg.get("save_model", True):
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        feature_extractor.save_pretrained(str(output_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
