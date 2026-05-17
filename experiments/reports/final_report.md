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
| Codec degradation support | `905fd14` | `requirements.txt`, `experiments/run_experiment.py` | Complete |
| Full degradation with codecs and babble | `6437b2c` | `experiments/runs/degradation_full_20260517/` | Complete |
| Scaled chunking aggregation config | `6f63389` | `experiments/configs/chunking_scaled.yaml` | Complete |
| Scaled chunking stability run | `327848e` | `experiments/runs/chunking_scaled_20260517/` | Complete |
| Training harness two-step smoke | `9af9b18` | `experiments/training_runs/training_smoke_smoke_l2_n3/result.json` | Complete |
| Quantizer-layer pilot config | `f75afa8` | `experiments/configs/training_layer_pilot.yaml` | Complete |
| Quantizer-layer one-step pilots | `ab88249` | `experiments/training_runs/training_layer_pilot_l*_n5_c3/result.json` | Complete |
| Factorial pilot config | `8c41bf1` | `experiments/configs/training_factorial_pilot.yaml` | Complete |
| Factorial one-step pilots | pending | `experiments/training_runs/training_factorial_pilot_*/result.json` | Complete |

## Exact Commands

```bash
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_smoke.yaml --dry-run
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/sanity.yaml --run-id sanity_smoke_20260517
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/degradation.yaml --run-id degradation_smoke_20260517
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/chunking.yaml --run-id chunking_smoke_20260517
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/distribution.yaml --run-id distribution_smoke_20260517
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/latency.yaml --run-id latency_smoke_20260517
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/distribution.yaml --run-id distribution_zipf_smoke_20260517
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/degradation_full.yaml --run-id degradation_full_20260517
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/chunking_scaled.yaml --run-id chunking_scaled_20260517
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_smoke.yaml
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_layer_pilot.yaml --variant l8_n5_c3
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_layer_pilot.yaml --variant l16_n5_c3
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_layer_pilot.yaml --variant l24_n5_c3
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_factorial_pilot.yaml --variant single_no_aug
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_factorial_pilot.yaml --variant single_aug
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_factorial_pilot.yaml --variant multi_no_aug
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_factorial_pilot.yaml --variant multi_aug
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_factorial_pilot.yaml --variant multi_aug_consensus
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

### Full Degradation With Codecs

| Corruption | FR Mean UED | ZH Mean UED | Lesson |
|---|---:|---:|---|
| Babble 4 speakers 10 dB | `0.6382` | `0.4134` | Hardest condition; multi-speaker interference should be a headline robustness test. |
| Competing speech 16 dB | `0.4579` | `0.2882` | Still a top semantic interference risk. |
| Reverb small room | `0.3639` | `0.2315` | Remains stronger than most additive/channel corruptions. |
| Gaussian 25 dB | `0.2607` | `0.1305` | Useful baseline, but not the hardest condition. |
| AAC 32k | `0.2547` | `0.1394` | Codec degradation is measurable at low bitrate. |
| Telephone bandpass | `0.2518` | `0.1334` | Similar scale to low-bitrate codecs. |
| MP3 32k | `0.2392` | `0.1193` | Lower than AAC in this run. |
| Opus 24k | `0.2063` | `0.1025` | Lowest codec UED here despite lower nominal bitrate. |
| Brown 16 dB | `0.1050` | `0.0485` | Milder than expected on this slice. |
| Clipping 0.50 | `0.0000` | `0.0695` | Mild clipping is not a useful stressor for FR in this slice. |

Full run size: 50 FLEURS-fr + 50 CV21-zh clips x 13 corruptions = 1300 rows.

Plot: [`degradation_mean_ued.png`](../runs/degradation_full_20260517/plots/degradation_mean_ued.png)

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

### Scaled Chunking / Aggregation

| Policy | Aggregation | Overall Mismatch | Boundary ±0.5s | Boundary ±1s | End 25-30s |
|---|---|---:|---:|---:|---:|
| 30s window, 15s stride | first | `0.3215` | `0.4520` | `0.4049` | `0.0000` |
| 30s window, 15s stride | majority | `0.3215` | `0.4520` | `0.4049` | `0.0000` |
| 30s window, 15s stride | center | `0.4049` | `0.8440` | `0.8127` | `0.8672` |
| 30s window, 5s stride | first | `0.4566` | `0.4920` | `0.4627` | `0.0000` |
| 30s window, 5s stride | majority | `0.4623` | `0.6360` | `0.5608` | `0.3364` |
| 30s window, 5s stride | center | `0.5509` | `0.8440` | `0.8127` | `0.8652` |
| 30s window, 1s stride | first | `0.3629` | `0.4520` | `0.4294` | `0.0000` |
| 30s window, 1s stride | majority | `0.5445` | `0.7320` | `0.7275` | `0.7024` |
| 30s window, 1s stride | center | `0.5399` | `0.8080` | `0.7961` | `0.8672` |

Scaled run size: 10 stitched 75s FLEURS-fr long samples. The run produced 720 region rows and 450 distance rows.

Plot: [`chunking_boundary_instability.png`](../runs/chunking_scaled_20260517/plots/chunking_boundary_instability.png)

Result: the chunking instability persists beyond the single-audio smoke. The new `center` aggregation is a useful negative result: selecting the token farthest from an overlapping window edge does not reproduce the non-overlap tokenization target and is especially bad near reference chunk ends.

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
| MP3/Opus/AAC compression run | Fixed and executed | Added `imageio-ffmpeg==0.6.0` fallback because system `ffmpeg` is not on PATH. Full codec rows are in `degradation_full_20260517`. |
| Full training ablations L8/L12/L16/L20/L24 | Pilot only | L8/L16/L24 each ran one matched step. Full scientific runs still need matched data scale, seeds, steps, and downstream evaluation. |
| Architecture-vs-augmentation factorial | Pilot only | All five variants ran one matched step. Full conclusions need real training budget, matched seeds/steps, and downstream evaluation. |
| ASR WER, SER, speaker/prosody, TTS metrics | Not executed | No stable downstream evaluation pipeline was wired in this pass. |
| SpeechLLM QA/translation/summarization/instruction tasks | Not executed | Needs an adapter/LoRA or full SpeechLLM pipeline; recommended after tokenizer-level issues are scaled. |

## Interpretation

The sanity run supports the basic reproducibility claims for released inference: 25 Hz, 750 tokens for 30 seconds, and 8192 vocabulary.

The most important negative finding is chunking instability. StableToken is robust to some perturbations, but the Whisper-style non-causal 30s window can still produce substantially different token sequences for the same absolute time span under different streaming policies. The scaled run confirms this beyond a single example, and simple first/majority/center aggregation does not solve it.

The degradation suite says the next robustness budget should go to babble, competing speech, and reverb, not just additive noise. Competing speech is especially valuable because the tokenizer may encode it as real semantic content rather than discard it.

The distribution results show broad but sparse code usage, with meaningful language/domain drift. This is not collapse, but it does mean multilingual and noisy/clean analyses need stratified reporting.

The training harness can now run real steps locally. The two-step `openai/whisper-tiny` smoke is not a scientific ablation, but it validates that the LFQ encoder, noisy branch, tokenizer, collator, and Trainer loop are wired well enough to justify small matched ablation pilots.

## Training Smoke

| Metric | Value |
|---|---:|
| Variant | `smoke_l2_n3` |
| Base model | `openai/whisper-tiny` |
| Max steps | `2` |
| Step losses | `9.0804`, `8.0062` |
| Train loss | `8.5433` |
| Runtime | `1.97s` |
| Train samples/second | `1.013` |
| LFQ encoder missing/unexpected keys | `10 / 0` |

Result file: [`result.json`](../training_runs/training_smoke_smoke_l2_n3/result.json)

The first actual training run failed once because `WhisperVQConfig` lacked `encoder_causal_convolution`; this was fixed by adding a default `false`, matching the non-causal chunk behavior used by the released tokenizer.

## Quantizer-Layer Pilot

These are one-step feasibility pilots, not scientific ablations. They validate that the requested L8/L16/L24 layer positions can run under matched config on `openai/whisper-medium` with `save_model=false`.

| Variant | Quantizer Layer | Steps | Train Loss | Runtime | Samples/s | Saved Model |
|---|---:|---:|---:|---:|---:|---|
| `l8_n5_c3` | 8 | 1 | `78.1307` | `0.800s` | `1.249` | false |
| `l16_n5_c3` | 16 | 1 | `98.0062` | `0.791s` | `1.264` | false |
| `l24_n5_c3` | 24 | 1 | `76.4871` | `0.818s` | `1.223` | false |

Result files: [`l8`](../training_runs/training_layer_pilot_l8_n5_c3/result.json), [`l16`](../training_runs/training_layer_pilot_l16_n5_c3/result.json), [`l24`](../training_runs/training_layer_pilot_l24_n5_c3/result.json)

Interpretation: the exact requested layer positions are feasible locally. The next valid comparison needs fixed seeds, fixed data, more steps, and downstream UED/ASR/SER evaluation; one-step losses should not be used to rank L8/L16/L24.

## Factorial Pilot

These are one-step feasibility pilots for the architecture-vs-augmentation concern. They validate that the matched matrix is runnable; they do not isolate real performance differences yet.

| Variant | Branches | Augmentation | Consensus | Train Loss | Runtime | Saved Model |
|---|---:|---|---:|---:|---:|---|
| `single_no_aug` | 1 | false | `0.0` | `9.6039` | `0.682s` | false |
| `single_aug` | 1 | true | `0.0` | `9.6039` | `0.689s` | false |
| `multi_no_aug` | 5 | false | `0.0` | `10.0473` | `0.776s` | false |
| `multi_aug` | 5 | true | `0.0` | `10.1541` | `0.725s` | false |
| `multi_aug_consensus` | 5 | true | `0.25` | `10.2009` | `0.715s` | false |

Result files live under `experiments/training_runs/training_factorial_pilot_*/result.json`.

Interpretation: the exact factorial variants now run end to end. Since each run is a single step on a tiny slice, the next meaningful version needs enough steps to produce checkpoints, then the same UED/degradation/downstream evaluation used for inference.

## Recommended Next Runs

1. Scale `degradation_full.yaml` to 100-500 clips per language and add more languages. Keep per-language metrics, because the smoke slice shows large source effects.

2. Test stronger chunk aggregation beyond first/majority/center: confidence from hidden-state distance, hidden-state averaging before quantization, and training-time chunk-position augmentation.

3. Extend the quantizer-placement pilot from one step to a small matched run: L8, L16, L24, fixed data, fixed steps, fixed seeds, then evaluate the resulting checkpoints on UED/degradation before adding L12/L20.

4. Extend the architecture-vs-augmentation factorial from one-step pilots to matched short runs that save checkpoints, then evaluate UED/nUED, token entropy, ASR WER, and SER.

5. Add downstream ASR WER and SER on the same clean/noisy/degradation splits before investing in full SpeechLLM experiments.

6. For SpeechLLM usefulness, start with a small adapter/LoRA controlled experiment on spoken QA or noisy spoken dialogue understanding, using identical data and steps across tokenizer variants.
