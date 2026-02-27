import argparse
import json
import logging
import os
import torch
from datasets import load_dataset
from jiwer import wer
from tqdm import tqdm
from transformers import WhisperFeatureExtractor
from src.model.modeling_whisper import WhisperLFQEncoder
from src.utils.utils import extract_speech_token


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UED using StableToken")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for inference (auto, cpu, cuda, cuda:0, etc.)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model directory (including the \
              tokenizer subdirectory and the decoder subdirectory)"
    )
    parser.add_argument(
        "--parquet_files",
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to the parquet file(s) containing clean & noisy audio data"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./UED_results/ued_results.json",
        help="Path to the output JSON file to save UED results"
    )

    return parser.parse_args()


def load_stabletoken_model(model_path: str, device: str):
    """Load StableToken model and feature extractor from the specified path."""
    tokenizer_path = os.path.join(model_path, "tokenizer")
    tokenizer = WhisperLFQEncoder.from_pretrained(tokenizer_path).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)
    return tokenizer, feature_extractor


def calculate_ued(tokens_clean_list, tokens_noise_list):
    total_ued = 0.0
    total_tokens = 0

    for tokens_clean, tokens_noise in zip(tokens_clean_list, tokens_noise_list):
        clean_str = ' '.join(str(x) for x in tokens_clean)
        noise_str = ' '.join(str(x) for x in tokens_noise)

        ued = wer(clean_str, noise_str)
        total_ued += ued * len(tokens_clean)
        total_tokens += len(tokens_clean)

    ued = total_ued / total_tokens if total_tokens > 0 else 1.0
    return ued


def evaluate_ued(
    tokenizer: WhisperLFQEncoder,
    feature_extractor: WhisperFeatureExtractor,
    parquet_file: str,
    device: str = 'cuda',
):
    # Load dataset
    dataset = load_dataset('parquet', data_files=parquet_file)['train']

    tokens_clean_list = {'en': [], 'zh': []}
    tokens_noise_list = {'en': [], 'zh': []}

    for idx in tqdm(range(len(dataset)), desc="Extracting speech tokens"):
        for lang in ('en', 'zh'):
            clean_audio = dataset[idx][f'audio_{lang}_clean']['array']
            clean_sr    = dataset[idx][f'audio_{lang}_clean']['sampling_rate']
            noise_audio = dataset[idx][f'audio_{lang}_noise']['array']
            noise_sr    = dataset[idx][f'audio_{lang}_noise']['sampling_rate']

            # Extract tokens for clean audio
            tokens_clean = extract_speech_token(
                model=tokenizer,
                feature_extractor=feature_extractor,
                audios=[(clean_audio, clean_sr)],
                device=device,
            )[0]
            tokens_clean_list[lang].append(tokens_clean)

            # Extract tokens for noisy audio
            tokens_noise = extract_speech_token(
                model=tokenizer,
                feature_extractor=feature_extractor,
                audios=[(noise_audio, noise_sr)],
                device=device,
            )[0]
            tokens_noise_list[lang].append(tokens_noise)

    ued_en = calculate_ued(tokens_clean_list['en'], tokens_noise_list['en'])
    ued_zh = calculate_ued(tokens_clean_list['zh'], tokens_noise_list['zh'])

    logging.info(f"UED Results for {os.path.basename(parquet_file)}:")
    logging.info(f"  UED (English): {ued_en:.4f}")
    logging.info(f"  UED (Chinese): {ued_zh:.4f}")
    logging.info(f"  Average UED: {(ued_en + ued_zh) / 2:.4f}")

    ued = {"ued_en": ued_en, "ued_zh": ued_zh, "ued_avg": (ued_en + ued_zh) / 2}
    return ued


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logging.info(f"Using device: {device}")

    # Load StableToken model and feature extractor
    tokenizer, feature_extractor = load_stabletoken_model(args.model_path, device)

    # Evaluate UED for each parquet file
    all_ued_results = {}
    for parquet_file in args.parquet_files:
        logging.info(f"Evaluating UED for {parquet_file}...")

        ued_results = evaluate_ued(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            parquet_file=parquet_file,
            device=device,
        )
        all_ued_results[parquet_file] = ued_results

    # Save UED results to output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_ued_results, f, indent=4, ensure_ascii=False)
    logging.info(f"UED results saved to {args.output_file}")
