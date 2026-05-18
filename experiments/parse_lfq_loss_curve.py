#!/usr/bin/env python
"""Parse Trainer loss dictionaries from an LFQ training log."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path


def parse_loss_rows(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")

    json_rows = []
    for line in text.splitlines():
        if "LFQ_LOG " not in line:
            continue
        payload = line.split("LFQ_LOG ", 1)[1].strip()
        try:
            row = json.loads(payload)
        except Exception:
            continue
        if "loss" in row:
            json_rows.append(row)
    if json_rows:
        return json_rows

    rows = []
    for match in re.finditer(r"\{[^{}]*'loss'[^{}]*\}", text):
        try:
            row = ast.literal_eval(match.group(0))
        except Exception:
            continue
        if "loss" in row:
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], logging_steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["log_index", "approx_step", "loss", "grad_norm", "learning_rate", "epoch"],
        )
        writer.writeheader()
        for idx, row in enumerate(rows, 1):
            step = row.get("step") or idx * logging_steps
            writer.writerow(
                {
                    "log_index": idx,
                    "approx_step": step,
                    "loss": row.get("loss"),
                    "grad_norm": row.get("grad_norm"),
                    "learning_rate": row.get("learning_rate"),
                    "epoch": row.get("epoch"),
                }
            )


def write_plot(path: Path, rows: list[dict], logging_steps: int, title: str) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [row.get("step") or idx * logging_steps for idx, row in enumerate(rows, 1)]
    ys = [float(row["loss"]) for row in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o", linewidth=1.5)
    plt.xlabel("approx step")
    plt.ylabel("training loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--png", required=True)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--title", default="LFQ training loss")
    args = parser.parse_args()

    rows = parse_loss_rows(Path(args.log))
    write_csv(Path(args.csv), rows, args.logging_steps)
    write_plot(Path(args.png), rows, args.logging_steps, args.title)
    print({"loss_points": len(rows), "csv": args.csv, "png": args.png})


if __name__ == "__main__":
    main()
