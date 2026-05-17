# StableToken Inference-First Experiment Report

Date UTC: 2026-05-17

This report covers the first feasible experiment pass on the local `main` branch. It prioritizes inference-only checks before training-heavy ablations, following the reviewer concern order: reproducibility, pseudo-streaming stability, real-world degradation, token distribution, and efficiency.

## Setup

| Item | Value |
|---|---|
| Paper target | StableToken, ICLR 2026, arXiv `2509.22220` |
| Released checkpoint | `tencent/StableToken`, revision `dd19e26b6e7a2ffcc8523d15b80238175c69a2b6` |
| Python | `/data/venv/bin/python` |
| Torch | `2.11.0+cu128` |
| GPU | NVIDIA L4 |
| Starting repo commit | `af20e58` |
| Experiment identity | `EvergreenTree <627430923@qq.com>` |
| Local data | FLEURS-fr and CV21-zh eval slices under `/data/speech2text/Qwen3-ASR/finetuning/eval_slices/` |

## What Ran

| Family | Commit | Run Directory | Status |
|---|---:|---|---|
| Ledger + upstream README preservation | `0d50926` | `README.md`, `README.upstream.md` | Complete |
| Config-driven inference runner | `7407108` | `experiments/run_experiment.py` | Complete |
| Training ablation harness scaffold | `9cbbf5d` | `experiments/train_lfq_tokenizer.py` | Dry-run complete |
| Sanity/reproducibility smoke | `e5529bd` | `experiments/runs/sanity_smoke_20260517/` | Complete |
| Hard degradation smoke | `2f43fc4` | `experiments/runs/degradation_smoke_20260517/` | Complete |
| Chunking/pseudo-streaming smoke | `7e150f0` | `experiments/runs/chunking_smoke_20260517/` | Complete |
| Distribution drift smoke | `8e751c6` | `experiments/runs/distribution_smoke_20260517/` | Complete |
| Latency smoke | `6a394c4` | `experiments/runs/latency_smoke_20260517/` | Complete |
| Full degradation config + Zipf support | `f122ad6` | `experiments/configs/degradation_full.yaml` | Complete |
| Distribution Zipf smoke | `9124249`, trimmed by `ca156b8` | `experiments/runs/distribution_zipf_smoke_20260517/` | Complete |

## Exact Commands

```bash
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_smoke.yaml --dry-run
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/sanity.yaml --run-id sanity_smoke_20260517
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/degradation.yaml --run-id degradation_smoke_20260517
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/chunking.yaml --run-id chunking_smoke_20260517
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/distribution.yaml --run-id distribution_smoke_20260517
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/latency.yaml --run-id latency_smoke_20260517
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/distribution.yaml --run-id distribution_zipf_smoke_20260517
```

## Key Results

### Reproducibility Sanity

| Metric | Value |
|---|---:|
| Mean token rate | `25.0 Hz` |
| Mean tokens for 30s crop | `750` |
| Vocab size | `8192` |
| Bits/token | `13` |
| Bitrate | `325 bps` |
| Mean UED, Gaussian smoke | `0.1553` |
| Mean nUED | `0.01195` |
| Quantizer layer | `16` |
| Voters / clean inputs | `5 / 3` |

Result: the released checkpoint loads and matches the expected 25 Hz, 30s-to-750-token, 8192-vocab behavior.

### Degradation Suite

| Corruption | FR Mean UED | ZH Mean UED | Lesson |
|---|---:|---:|---|
| Reverb small room | `0.4222` | `0.2248` | Hardest smoke corruption; scale this first. |
| Competing speech 16 dB | `0.3603` | `0.2324` | Important because competing speech may be semantic content. |
| Gaussian 25 dB | `0.2588` | `0.1134` | Additive noise remains useful but not sufficient. |
| Pink 22 dB | `0.2352` | `0.0980` | Colored noise should stay in the suite. |
| Telephone bandpass | `0.2252` | `0.1267` | Channel effects are non-trivial. |
| Dropout 5 pct | `0.1570` | `0.2053` | Packet-loss behavior differs by source/language. |
| Bitcrush 10-bit | `0.2007` | `0.0305` | Highly source dependent. |
| Clipping 0.50 | `0.0000` | `0.0506` | Mild clipping was mostly harmless in this slice. |

Plot: [`degradation_mean_ued.png`](../runs/degradation_smoke_20260517/plots/degradation_mean_ued.png)

### Chunking / Pseudo-Streaming

| Policy | Aggregation | Overall Mismatch | Boundary ±0.5s Mismatch | Boundary ±1s Mismatch |
|---|---|---:|---:|---:|
| 30s window, 15s stride | first | `0.3067` | `0.4400` | `0.3922` |
| 30s window, 15s stride | majority | `0.3067` | `0.4400` | `0.3922` |
| 30s window, 5s stride | first | `0.4491` | `0.4800` | `0.4706` |
| 30s window, 5s stride | majority | `0.4896` | `0.6800` | `0.6569` |
| 30s window, 1s stride | first | `0.3555` | `0.4000` | `0.3333` |
| 30s window, 1s stride | majority | `0.5577` | `0.5200` | `0.6471` |

Plot: [`chunking_boundary_instability.png`](../runs/chunking_smoke_20260517/plots/chunking_boundary_instability.png)

Result: window placement materially changes tokens for the same absolute time span. Naive token majority over overlapping windows is not a fix; it often worsens mismatch and should be replaced with confidence, hidden-state, or alignment-aware aggregation.

### Token Distribution

| Group | Entropy Bits | Transition Entropy Bits | Unique Tokens | Dead-Token Rate |
|---|---:|---:|---:|---:|
| FR clean | `10.959` | `11.917` | `2562` | `0.687` |
| FR Gaussian 25 dB | `10.992` | `11.921` | `2606` | `0.682` |
| ZH clean | `10.324` | `10.922` | `1689` | `0.794` |
| ZH Gaussian 25 dB | `10.304` | `10.920` | `1671` | `0.796` |

| KL Pair | KL Bits |
|---|---:|
| FR clean -> FR noisy | `2.134` |
| FR noisy -> FR clean | `2.308` |
| ZH clean -> ZH noisy | `1.304` |
| ZH noisy -> ZH clean | `1.228` |
| FR/ZH cross-language | roughly `15-16` |

Plots: [`distribution_entropy.png`](../runs/distribution_zipf_smoke_20260517/plots/distribution_entropy.png), [`distribution_zipf.png`](../runs/distribution_zipf_smoke_20260517/plots/distribution_zipf.png)

Result: language/source distribution dominates Gaussian token drift in this slice. Future claims should report token usage separately by language/domain/noise condition.

### Latency / Efficiency

| Duration | Batch | Mean Latency | RTF | Audio Seconds / Second | Tokens / Item | Peak CUDA MB |
|---:|---:|---:|---:|---:|---:|---:|
| 1s | 1 | `0.011s` | `0.0112` | `89.7` | `25` | `1299` |
| 1s | 4 | `0.0207s` | `0.00518` | `193.1` | `25` | `1305` |
| 10s | 1 | `0.0455s` | `0.00455` | `219.6` | `250` | `1315` |
| 10s | 4 | `0.179s` | `0.00448` | `223.4` | `250` | `1428` |
| 30s | 1 | `0.151s` | `0.00504` | `198.2` | `750` | `1395` |
| 30s | 4 | `0.586s` | `0.00488` | `204.9` | `750` | `1638` |

Plot: [`latency_rtf.png`](../runs/latency_smoke_20260517/plots/latency_rtf.png)

Result: extraction is comfortably faster than realtime on the L4 for these smoke settings. The 1s rows are dominated by fixed overhead and should not be used alone for throughput claims.

## What Failed Or Was Infeasible

| Item | Status | Reason / Fix |
|---|---|---|
| Initial audio loading through `torchaudio.load` | Fixed | Local TorchCodec/FFmpeg libraries were incomplete. The runner and training harness now try `soundfile` first and use `torchaudio` as fallback. |
| MP3/Opus/AAC compression run | Prepared, not executed | `ffmpeg` is not on PATH in this host. `codec` corruptions are config-driven in `degradation_full.yaml` and will run once ffmpeg is installed. |
| Full training ablations L8/L12/L16/L20/L24 | Not executed | Upstream does not ship full tokenizer training scripts; local harness was scaffolded and dry-run only. Full runs need matched data scale, steps, and more compute time. |
| Architecture-vs-augmentation factorial | Config/harness path only | Needs real training budget and matched seeds/steps. |
| ASR WER, SER, speaker/prosody, TTS metrics | Not executed | No stable downstream evaluation pipeline was wired in this pass. |
| SpeechLLM QA/translation/summarization/instruction tasks | Not executed | Needs an adapter/LoRA or full SpeechLLM pipeline; recommended after tokenizer-level issues are scaled. |

## Interpretation

The sanity run supports the basic reproducibility claims for released inference: 25 Hz, 750 tokens for 30 seconds, and 8192 vocabulary.

The most important negative finding is chunking instability. StableToken is robust to some perturbations, but the Whisper-style non-causal 30s window can still produce substantially different token sequences for the same absolute time span under different streaming policies. This should be addressed before strong streaming claims.

The degradation suite says the next robustness budget should go to reverb and competing speech, not just additive noise. Competing speech is especially valuable because the tokenizer may encode it as real semantic content rather than discard it.

The distribution results show broad but sparse code usage, with meaningful language/domain drift. This is not collapse, but it does mean multilingual and noisy/clean analyses need stratified reporting.

## Recommended Next Runs

1. Install `ffmpeg`, then run:

```bash
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/degradation_full.yaml --run-id degradation_full_20260517
```

2. Scale `degradation.yaml` to 100-500 clips per language and add more languages. Keep per-language metrics, because the smoke slice shows large source effects.

3. Rerun chunking on at least 10 long audios and test aggregation beyond token majority: central-window selection, confidence from hidden-state distance, and hidden-state averaging before quantization.

4. Launch the minimum quantizer-placement training ablation first: L8, L16, L24, fixed data, fixed steps, fixed seeds. Add L12/L20 only after the minimum run validates the harness.

5. Run the architecture-vs-augmentation factorial only after the layer ablation: single-branch/no-aug, single-branch/aug, multi-branch/no-aug, multi-branch/aug, multi-branch/aug+consensus.

6. Add downstream ASR WER and SER on the same clean/noisy/degradation splits before investing in full SpeechLLM experiments.

7. For SpeechLLM usefulness, start with a small adapter/LoRA controlled experiment on spoken QA or noisy spoken dialogue understanding, using identical data and steps across tokenizer variants.
