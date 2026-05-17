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
| 2026-05-17 | Latency smoke run | `6a394c4` | `/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/latency.yaml --run-id latency_smoke_20260517` | Synthetic zero audio, durations 1s/10s/30s, batch sizes 1/4 | 6 rows; mean RTF `0.00588`. 30s batch 1: `0.151s`, RTF `0.00504`, 750 tokens. 30s batch 4: `0.586s`, RTF `0.00488`, peak CUDA memory `1638 MB`. | Token extraction is comfortably faster than realtime on the L4 for smoke settings; 1s rows show fixed overhead and should not be used alone for throughput claims. | Run artifacts and RTF plot: `experiments/runs/latency_smoke_20260517/`. |
| 2026-05-17 | Full degradation and Zipf support added | `f122ad6` | `experiments/configs/degradation_full.yaml`; `/data/venv/bin/python -m py_compile experiments/run_experiment.py` | n/a | Adds config-driven MP3/Opus/AAC codec roundtrip, babble mixing, and distribution Zipf CSV/plot output. | Codec corruptions require `ffmpeg`; this host does not currently have `ffmpeg` on PATH, so compression rows are prepared but not executed here. | Next distribution run should include `distribution_zipf.csv` and `plots/distribution_zipf.png`. |
| 2026-05-17 | Token distribution Zipf smoke run | `9124249` | `/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/distribution.yaml --run-id distribution_zipf_smoke_20260517` | 20 FLEURS-fr + 20 CV21-zh clips, clean and Gaussian 25 dB | Same summary as distribution smoke plus `distribution_zipf.csv`. Top FR-clean token `1124` frequency `0.00508`; top ZH-clean token `3234` frequency `0.00990`. | Token usage is broad but sparse in the 8192-code vocabulary; Zipf plots are now available for reviewer-facing collapse/drift checks. | Run artifacts and Zipf plot: `experiments/runs/distribution_zipf_smoke_20260517/`. |
| 2026-05-17 | Trim bulky Zipf JSON | `ca156b8` | `jq 'del(.zipf) + {zipf_csv: "distribution_zipf.csv"}' .../distribution_metrics.json` | n/a | Removed expanded Zipf rows from JSON while keeping CSV and plot. | CSV is the right artifact for full rank tables; JSON should stay summary-sized. | Current artifacts keep `distribution_zipf.csv` as the source of truth. |
| 2026-05-17 | Final inference-first report | `a10c838` | `experiments/reports/final_report.md` | All smoke runs above | Report includes what ran, failed/infeasible items, key tables, plots, interpretation, and recommended next runs. | Inference-first smoke tests are enough to identify the next compute priorities: chunking aggregation, reverb/competing speech, codec degradations, and minimal L8/L16/L24 training. | Final report path: `experiments/reports/final_report.md`. |
| 2026-05-17 | Codec degradation support enabled | `905fd14` | `/data/venv/bin/pip install imageio-ffmpeg==0.6.0`; codec roundtrip smoke in Python | Synthetic 1s sine wave | MP3 32k roundtrip produced 16000 float32 samples; ffmpeg resolved to `imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`. | Full degradation can now run codec rows even when system `ffmpeg` is absent. | Added `imageio-ffmpeg==0.6.0` to `requirements.txt`. |
| 2026-05-17 | Full degradation run with codecs and babble | `6437b2c` | `/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/degradation_full.yaml --run-id degradation_full_20260517` | 50 FLEURS-fr + 50 CV21-zh clips over 13 corruptions | 1300 rows, elapsed `126.9s`. Highest mean UED: FR babble `0.6382`, FR competing speech `0.4579`, ZH babble `0.4134`, FR reverb `0.3639`, ZH competing speech `0.2882`. Codec mean UED: MP3 `0.2392/0.1193`, Opus `0.2063/0.1025`, AAC `0.2547/0.1394` for FR/ZH. | Multi-speaker babble is now the hardest real-world degradation; codecs matter but are not the dominant failure mode at these bitrates. | Run artifacts and plot: `experiments/runs/degradation_full_20260517/`. |
| 2026-05-17 | Scaled chunking config and center aggregation | `6f63389` | `/data/venv/bin/python -m py_compile experiments/run_experiment.py` | n/a | Adds `center` aggregation and `experiments/configs/chunking_scaled.yaml` for 10 stitched 75s FLEURS-fr audios. | Center aggregation tests the hypothesis that tokens farthest from 30s window edges are more stable than first or majority overlap votes. | Run command below executed in the next row. |
| 2026-05-17 | Scaled chunking stability run | `327848e` | `/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/chunking_scaled.yaml --run-id chunking_scaled_20260517` | 10 stitched FLEURS-fr long samples, 75s each | 720 region rows; 450 distance rows; mean overall mismatch `0.4406`. Best overall: stride 15s first/majority `0.3215`, stride 1s first `0.3629`, stride 5s first `0.4566`. Center aggregation worsened boundary mismatch: stride 15s ±0.5s `0.8440`, stride 5s ±0.5s `0.8440`, stride 1s ±0.5s `0.8080`. | Chunk instability persists at 10-audio scale; central-window selection does not reproduce non-overlap tokens and is a poor default for this target. | Run artifacts and plot: `experiments/runs/chunking_scaled_20260517/`. |
| 2026-05-17 | Training harness smoke fix and real two-step run | `9af9b18` | `/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_smoke.yaml` | 8 FLEURS-fr train rows, `openai/whisper-tiny`, variant `smoke_l2_n3` | Succeeded for 2 steps. Losses `9.0804`, `8.0062`; train loss `8.5433`; runtime `1.97s`; LFQ encoder missing keys `10`, unexpected `0`. | Minimal training is feasible after adding default `encoder_causal_convolution=false`; large checkpoint artifacts must stay ignored. | Lightweight result: `experiments/training_runs/training_smoke_smoke_l2_n3/result.json`. |
| 2026-05-17 | Quantizer-layer pilot config prepared | `f75afa8` | `/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_layer_pilot.yaml --variant {l8_n5_c3,l16_n5_c3,l24_n5_c3} --dry-run` | Config only | Dry-run status `dry_run_ok` for L8, L16, and L24 on `openai/whisper-medium`; `save_model=false` keeps actual pilots lightweight. | The exact reviewer-requested layer positions are launchable as matched one-step feasibility pilots before any expensive ablation. | Config: `experiments/configs/training_layer_pilot.yaml`. |
| 2026-05-17 | Quantizer-layer one-step pilot runs | `ab88249` | `/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_layer_pilot.yaml --variant {l8_n5_c3,l16_n5_c3,l24_n5_c3}` | 6 FLEURS-fr train rows, `openai/whisper-medium`, 1 step each | L8 loss `78.1307`, runtime `0.800s`; L16 loss `98.0062`, runtime `0.791s`; L24 loss `76.4871`, runtime `0.818s`; all `saved_model=false`. | L8/L16/L24 are feasible in this environment, but one-step losses are only harness checks, not scientific evidence about optimal layer placement. | Results: `experiments/training_runs/training_layer_pilot_l*_n5_c3/result.json`. |
| 2026-05-17 | Architecture-vs-augmentation factorial pilot config prepared | `8c41bf1` | `/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_factorial_pilot.yaml --variant <variant> --dry-run` | Config only | Dry-run status `dry_run_ok` for `single_no_aug`, `single_aug`, `multi_no_aug`, `multi_aug`, and `multi_aug_consensus`. | Reviewer concern is represented as a matched config matrix: single vs multi branch, augmentation on/off, and consensus loss on/off. | Config: `experiments/configs/training_factorial_pilot.yaml`. |
| 2026-05-17 | Architecture-vs-augmentation one-step pilot runs | `6137874` | `/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_factorial_pilot.yaml --variant <variant>` | 6 FLEURS-fr train rows, `openai/whisper-tiny`, 1 step each | Losses: single no aug `9.6039`, single aug `9.6039`, multi no aug `10.0473`, multi aug `10.1541`, multi aug + consensus `10.2009`; all `saved_model=false`. | The factorial matrix runs end to end; one-step losses show overhead/feasibility only and should not be interpreted as robust ablation outcomes. | Results: `experiments/training_runs/training_factorial_pilot_*/result.json`. |
| 2026-05-17 | Voter count and clean:noisy pilot config prepared | `1254f34` | `/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_voter_pilot.yaml --variant <variant> --dry-run` | Config only | Dry-run status `dry_run_ok` for N=5 ratios `5:0`, `4:1`, `3:2`, `2:3`, `1:4`, `0:5` and voter counts N=`1,3,5,7`. | Reviewer concern about whether `3:2` and N=5 are actually optimal is represented as a matched pilot matrix. | Config: `experiments/configs/training_voter_pilot.yaml`. |
| 2026-05-17 | Voter count and clean:noisy one-step pilot runs | pending | `/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_voter_pilot.yaml --variant <variant>` | 6 FLEURS-fr train rows, `openai/whisper-tiny`, 1 step each | N=5 ratio losses: 5:0 `10.0948`, 4:1 `10.1706`, 3:2 `10.2009`, 2:3 `10.2554`, 1:4 `10.2594`, 0:5 `10.1938`. Voter-count losses: N1 `9.6039`, N3 `8.8181`, N5 `10.2009`, N7 `8.9456`. | The full voter/ratio pilot matrix runs, including N=7 and 0-clean edge cases; one-step losses are feasibility signals only. | Results: `experiments/training_runs/training_voter_pilot_*/result.json`. |

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

The final report lives at [experiments/reports/final_report.md](experiments/reports/final_report.md) and includes:

- What was run and what failed or was infeasible.
- Key robustness, streaming, distribution, and latency tables.
- Plots for boundary instability, degradation sensitivity, and token usage.
- Interpretation and recommended next runs.
