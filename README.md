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
| 2026-05-17 | Ledger initialized | `0d50926` | `mv README.md README.upstream.md` plus new ledger README | n/a | n/a | Keep the run history in the front page; preserve upstream docs separately. | First implementation step before adding runners or launching experiments. |
| 2026-05-17 | Config-driven inference runner added | `7407108` | `experiments/run_experiment.py` plus `experiments/configs/*.yaml` | Local FLEURS-fr and CV21-zh slices discovered under `/data/speech2text` | Supports sanity, chunking, degradation, distribution, and latency tasks | Inference-first plan is feasible without training data scale changes. | Outputs are CSV/JSON/PNG/TXT under `experiments/runs/`; large checkpoints remain ignored. |
| 2026-05-17 | LFQ ablation harness scaffolded | `9cbbf5d` | `/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_smoke.yaml --dry-run` | Config-only smoke using local JSONL manifests | Dry run status: `dry_run_ok` | Training-heavy reviewer ablations can be launched later with matched seeds/steps; upstream does not ship full tokenizer training scripts. | Added layer/voter/noise/consensus knobs without launching expensive training. |
| 2026-05-17 | Reproducibility sanity smoke run | `e5529bd` | `/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/sanity.yaml --run-id sanity_smoke_20260517` | 2 FLEURS-fr + 2 CV21-zh local dev clips, each 30s crop | Mean token rate `25.0 Hz`; mean clean tokens `750`; vocab `8192`; bits/token `13`; bitrate `325 bps`; mean UED `0.1553`; mean nUED `0.01195` | Released checkpoint loads and basic extraction matches the paper-facing token-rate/vocab claims; audio loading needed `soundfile` before `torchaudio.load` because local TorchCodec/FFmpeg is incomplete. | Run artifacts: `experiments/runs/sanity_smoke_20260517/`. TorchAO optional extension warnings are non-fatal. |
| 2026-05-17 | Real-world degradation smoke run | `2f43fc4` | `/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/degradation.yaml --run-id degradation_smoke_20260517` | 5 FLEURS-fr + 5 CV21-zh clips over 10 corruptions | 100 rows. Highest mean UED: FR reverb `0.4222`, FR competing speech `0.3603`, FR Gaussian 25 dB `0.2588`, ZH competing speech `0.2324`, ZH reverb `0.2248`. Lowest: FR clipping `0.0`, ZH bitcrush `0.0305`. | Reverb and competing speech are the most valuable immediate stressors; language/source differences are large enough that later runs should keep per-language reporting. | Run artifacts: `experiments/runs/degradation_smoke_20260517/`. |
| 2026-05-17 | Pseudo-streaming chunking smoke run | `7e150f0` | `/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/chunking.yaml --run-id chunking_smoke_20260517` | 1 stitched FLEURS-fr long sample, 75s target | 48 region rows; 30 distance rows; mean overall mismatch `0.4109`. Overall mismatch by first-token aggregation: stride 15s `0.3067`, stride 5s `0.4491`, stride 1s `0.3555`. Boundary ±0.5s mismatch: stride 15s `0.44`, stride 5s `0.48`, stride 1s `0.40`. | Whisper-style 30s window placement materially changes tokens for the same absolute time span; naive overlap majority can worsen disagreement and needs a smarter confidence/alignment rule. | Run artifacts and boundary plot: `experiments/runs/chunking_smoke_20260517/`. |
| 2026-05-17 | Token distribution smoke run | `8e751c6` | `/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/distribution.yaml --run-id distribution_smoke_20260517` | 20 FLEURS-fr + 20 CV21-zh clips, clean and Gaussian 25 dB | Entropy bits: FR clean `10.959`, FR noisy `10.992`, ZH clean `10.324`, ZH noisy `10.304`. Dead-token rate: FR `0.687/0.682`, ZH `0.794/0.796`. KL clean->noisy: FR `2.134`, ZH `1.304`; cross-language KL about `15-16`. | Language/source distribution dominates Gaussian token drift in this smoke; future multilingual claims should report token usage separately by language and noise. | Run artifacts and entropy plot: `experiments/runs/distribution_smoke_20260517/`. |
| 2026-05-17 | Latency smoke run | pending | `/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/latency.yaml --run-id latency_smoke_20260517` | Synthetic zero audio, durations 1s/10s/30s, batch sizes 1/4 | 6 rows; mean RTF `0.00588`. 30s batch 1: `0.151s`, RTF `0.00504`, 750 tokens. 30s batch 4: `0.586s`, RTF `0.00488`, peak CUDA memory `1638 MB`. | Token extraction is comfortably faster than realtime on the L4 for smoke settings; 1s rows show fixed overhead and should not be used alone for throughput claims. | Run artifacts and RTF plot: `experiments/runs/latency_smoke_20260517/`. |

## Planned Experiment Families

| Priority | Family | Status | Outputs |
|---:|---|---|---|
| 1 | Reproducibility and baseline sanity | Smoke complete | `experiments/runs/sanity_smoke_20260517/` |
| 2 | Chunking and pseudo-streaming stability | Smoke complete | `experiments/runs/chunking_smoke_20260517/` |
| 3 | Degradation suite | Smoke complete | `experiments/runs/degradation_smoke_20260517/` |
| 4 | Token distribution and drift | Smoke complete | `experiments/runs/distribution_smoke_20260517/` |
| 5 | Latency and efficiency | Smoke complete | `experiments/runs/latency_smoke_20260517/` |
| 6 | Training ablations | Harness dry-run complete | Configs and smoke harness first; full runs only if compute is feasible |
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
