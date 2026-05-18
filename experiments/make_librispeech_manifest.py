#!/usr/bin/env python
"""Create a simple StableToken JSONL manifest from LibriSpeech-style folders.

Expected layout is the standard LibriSpeech pattern with `*.trans.txt` files
beside audio files. Output rows use:
`audio_path`, `text`, `language`, `speaker_id`, `chapter_id`, and `utterance_id`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def audio_for_utterance(root: Path, utt_id: str) -> Path | None:
    for suffix in (".flac", ".wav", ".mp3", ".ogg"):
        matches = list(root.rglob(f"{utt_id}{suffix}"))
        if matches:
            return matches[0]
    return None


def iter_rows(root: Path, language: str):
    for trans_path in sorted(root.rglob("*.trans.txt")):
        audio_root = trans_path.parent
        with trans_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                utt_id, text = line.split(" ", 1)
                audio_path = audio_for_utterance(audio_root, utt_id)
                if audio_path is None:
                    continue
                parts = utt_id.split("-")
                yield {
                    "audio_path": str(audio_path.resolve()),
                    "text": text,
                    "language": language,
                    "speaker_id": parts[0] if len(parts) > 0 else "",
                    "chapter_id": parts[1] if len(parts) > 1 else "",
                    "utterance_id": utt_id,
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="LibriSpeech split root, e.g. LibriSpeech/train-clean-100")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in iter_rows(Path(args.root), args.language):
            if args.max_items is not None and count >= args.max_items:
                break
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"output": str(output), "items": count}, indent=2))


if __name__ == "__main__":
    main()
