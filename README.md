# StableToken Experiment Ledger

This fork is being used to stress-test StableToken against reviewer concerns about
robustness, efficiency, pseudo-streaming behavior, token distribution, and
SpeechLLM usefulness.

The original project README has been preserved at [README.upstream.md](README.upstream.md).

## Current Ground Truth

| Item | Value |
|---|---|
| Repo branch | `main` |
| Starting commit | `af20e58` |
| Commit identity | `EvergreenTree <627430923@qq.com>` |
| Python environment | `/data/venv/bin/python` |
| GPU observed | NVIDIA L4, 23 GB |
| Released checkpoint | `tencent/StableToken`, target HF SHA `dd19e26b6e7a2ffcc8523d15b80238175c69a2b6` |
| Upstream code status | Inference and UED scripts are released; tokenizer training scripts are not released upstream. |

## Experiment Ledger

| Date UTC | Step | Commit | Config / Command | Dataset / Split | Metrics | Lessons | Notes |
|---|---|---:|---|---|---|---|---|
| 2026-05-17 | Ledger initialized | pending | `mv README.md README.upstream.md` plus new ledger README | n/a | n/a | Keep the run history in the front page; preserve upstream docs separately. | First implementation step before adding runners or launching experiments. |

## Planned Experiment Families

| Priority | Family | Status | Outputs |
|---:|---|---|---|
| 1 | Reproducibility and baseline sanity | Planned | Token rate, 30s token count, vocab size, UED/nUED, exact command log |
| 2 | Chunking and pseudo-streaming stability | Planned | Boundary disagreement CSV/JSON and instability plots |
| 3 | Degradation suite | Planned | UED/nUED by corruption, SNR/intensity, and split |
| 4 | Token distribution and drift | Planned | Entropy, dead-token rate, Zipf curves, KL divergence |
| 5 | Latency and efficiency | Planned | RTF, throughput, memory, batch/duration table |
| 6 | Training ablations | Planned | Configs and smoke harness first; full runs only if compute is feasible |
| 7 | SpeechLLM usefulness | Planned | Adapter/LoRA task configs after tokenizer-level results land |

## Reproducibility Rules

- Every experiment must be config-driven and record the exact command.
- Every run must log checkpoint ID, git commit, seed, dataset split, corruption type,
  SNR/intensity, and output paths.
- Results should be written as CSV/JSON plus plots where useful.
- Large generated artifacts, downloaded datasets, checkpoints, and audio stay out of git.
- Commit after each coherent step so the ledger and code evolve together.

## Local Data Candidates

| Source | Path | Use |
|---|---|---|
| FLEURS-fr Qwen export | `/data/speech2text/Qwen3-ASR/finetuning/data/fleurs-fr` | French clean speech, local manifests, quick sanity and distribution tests |
| CV21-zh Qwen export | `/data/speech2text/Qwen3-ASR/finetuning/data/cv21-zh` | Chinese clean speech, local manifests, language distribution tests |
| ASR benchmark pickles | `/data/speech2text/asr_bench/test_fleurs_fr.pkl`, `/data/speech2text/asr_bench/test_cv21_zh.pkl` | 500-clip local slices for broader inference evaluation |

## Quick Commands

```bash
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/sanity.yaml
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/chunking.yaml
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/degradation.yaml
```

## Report Targets

The final report will live at `experiments/reports/final_report.md` and include:

- What was run and what failed or was infeasible.
- Key robustness, streaming, distribution, and latency tables.
- Plots for boundary instability, degradation sensitivity, and token usage.
- Interpretation and recommended next runs.
