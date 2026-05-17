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

## Reviewer-Facing Claims

| Claim / Concern | Experiment | Result | Interpretation |
|---|---|---|---|
| Released inference is reproducible. | Load `tencent/StableToken` and tokenize clean/noisy 30s crops. | `25.0 Hz`, `750` tokens per 30s, vocab `8192`, bitrate `325 bps`. | Basic released-checkpoint behavior matches the paper-facing claims. |
| Whisper-style 30s chunks are not true streaming. | Compare the same absolute audio under non-overlap, 15s, 5s, and 1s stride windows. | Best non-overlap match among tested overlap policies was 15s stride first/majority mismatch `0.3191`; stride-1s edge+voter mismatch to non-overlap was `0.5733`. | Window placement materially changes tokens. Use fixed 30s chunks for reproducibility or define a separate high-overlap consensus tokenizer with explicit latency. |
| Inference-only aggregation is not enough. | Test first/majority/center, edge/voter/entropy weights, and pre-LFQ hidden averaging. | Hidden edge+voter mismatch to non-overlap was `0.6096`; confidence weighting also worsened the non-overlap target. | The next streaming fix should be training-time chunk-position augmentation or a formally defined high-overlap serving policy, not naive post-hoc aggregation. |
| Hard real-world degradations matter more than Gaussian alone. | 100-clip UED degradation run over babble, competing speech, reverb, codecs, channel effects, and colored noise. | Babble was hardest in both FR and ZH; competing speech and reverb followed. | Robustness claims should include speech-like interference and reverb, not only additive SNR tests. |
| Token UED and downstream ASR are related but not interchangeable. | Join 100-clip token UED with Whisper-small WER/CER on the same corruptions. | Babble is high on both axes; competing speech is high token UED but modest direct Whisper-small FR WER. | Report tokenizer robustness separately from downstream task impact. |
| WER through StableToken needs a trained adapter/head. | Replace Whisper-large-v3 encoder with released StableToken encoder and decode with stock Whisper decoder. | WER/CER were `1.0` for every row in the 16-row smoke; outputs were empty or punctuation. | Zero-shot decoder replacement is not a usable StableToken-token ASR evaluation. |
| Token distribution drift is language- and corruption-dependent. | Distribution/Zipf/KL under clean, Gaussian, reverb, babble, and competing speech for FR/ZH. | Clean→babble KL was `2.456` bits FR and `2.353` bits ZH; Gaussian was lower at `1.124`/`0.913`. | Token-collapse and drift reports should be stratified by language and hard corruption type. |

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
| Advanced chunking stability config | `3d0ce9a` | `experiments/configs/chunking_advanced.yaml` | Complete |
| Advanced chunking stability run | `00f951b` | `experiments/runs/chunking_advanced_20260517/` | Complete |
| Training harness two-step smoke | `9af9b18` | `experiments/training_runs/training_smoke_smoke_l2_n3/result.json` | Complete |
| Quantizer-layer pilot config | `f75afa8` | `experiments/configs/training_layer_pilot.yaml` | Complete |
| Quantizer-layer one-step pilots | `ab88249` | `experiments/training_runs/training_layer_pilot_l*_n5_c3/result.json` | Complete |
| Factorial pilot config | `8c41bf1` | `experiments/configs/training_factorial_pilot.yaml` | Complete |
| Factorial one-step pilots | `6137874` | `experiments/training_runs/training_factorial_pilot_*/result.json` | Complete |
| Voter pilot config | `1254f34` | `experiments/configs/training_voter_pilot.yaml` | Complete |
| Voter one-step pilots | `514a554` | `experiments/training_runs/training_voter_pilot_*/result.json` | Complete |
| 100-clip degradation config | `1dc0119` | `experiments/configs/degradation_full_100.yaml` | Complete |
| 100-clip degradation run | `2098fb9` | `experiments/runs/degradation_full_100_20260517/` | Complete |
| ASR degradation runner | `e39aaa6`, `ed03e6f` | `experiments/run_asr_degradation.py` | Complete |
| ASR degradation smoke | `112024a` | `experiments/runs/asr_degradation_smoke_20260517/` | Complete |
| Whisper-small ASR config | `bc1f017` | `experiments/configs/asr_degradation_small.yaml` | Complete |
| Whisper-small ASR smoke | `3aadf75` | `experiments/runs/asr_degradation_small_20260517/` | Complete |
| Whisper-small ASR 100-clip config | `2743038` | `experiments/configs/asr_degradation_small_100.yaml` | Complete |
| Whisper-small ASR 100-clip run | `c1a0a81` | `experiments/runs/asr_degradation_small_100_20260517/` | Complete |
| Token-vs-ASR comparison | `cb3c99e` | `experiments/analysis/token_asr_degradation_comparison.csv` | Complete |
| StableToken-encoder ASR probe | `8f10d9e`, `8f480b0`, `141eedd` | `experiments/configs/asr_stabletoken_smoke.yaml` | Complete |
| StableToken-encoder ASR smoke | `f2b5462` | `experiments/runs/asr_stabletoken_smoke_20260517/` | Negative result complete |
| Degradation distribution config | `16620a1` | `experiments/configs/distribution_degradation.yaml` | Complete |
| Degradation distribution run | `16ea989` | `experiments/runs/distribution_degradation_20260517/` | Complete |

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
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/chunking_advanced.yaml --run-id chunking_advanced_20260517
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_smoke.yaml
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_layer_pilot.yaml --variant l8_n5_c3
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_layer_pilot.yaml --variant l16_n5_c3
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_layer_pilot.yaml --variant l24_n5_c3
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_factorial_pilot.yaml --variant single_no_aug
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_factorial_pilot.yaml --variant single_aug
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_factorial_pilot.yaml --variant multi_no_aug
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_factorial_pilot.yaml --variant multi_aug
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_factorial_pilot.yaml --variant multi_aug_consensus
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_voter_pilot.yaml --variant n5_clean5_noisy0
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_voter_pilot.yaml --variant n5_clean4_noisy1
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_voter_pilot.yaml --variant n5_clean3_noisy2
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_voter_pilot.yaml --variant n5_clean2_noisy3
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_voter_pilot.yaml --variant n5_clean1_noisy4
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_voter_pilot.yaml --variant n5_clean0_noisy5
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_voter_pilot.yaml --variant n1_clean1
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_voter_pilot.yaml --variant n3_clean2
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_voter_pilot.yaml --variant n5_clean3
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/training_voter_pilot.yaml --variant n7_clean4
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/degradation_full_100.yaml --run-id degradation_full_100_20260517
/data/venv/bin/python experiments/run_asr_degradation.py --config experiments/configs/asr_degradation_smoke.yaml --run-id asr_degradation_smoke_20260517
/data/venv/bin/python experiments/run_asr_degradation.py --config experiments/configs/asr_degradation_small.yaml --run-id asr_degradation_small_20260517
/data/venv/bin/python experiments/run_asr_degradation.py --config experiments/configs/asr_degradation_small_100.yaml --run-id asr_degradation_small_100_20260517
/data/venv/bin/python experiments/run_asr_degradation.py --config experiments/configs/asr_stabletoken_smoke.yaml --run-id asr_stabletoken_smoke_20260517
/data/venv/bin/python experiments/run_experiment.py --config experiments/configs/distribution_degradation.yaml --run-id distribution_degradation_20260517
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

### 100-Clip Degradation Scale-Up

| Corruption | FR Mean UED | ZH Mean UED | Lesson |
|---|---:|---:|---|
| Babble 4 speakers 10 dB | `0.6071` | `0.4170` | Stable hardest condition at larger sample count. |
| Competing speech 16 dB | `0.4817` | `0.2750` | Strong semantic interference risk. |
| Reverb small room | `0.3661` | `0.2399` | Still clearly harder than most codec/channel rows. |
| Gaussian 25 dB | `0.2635` | `0.1308` | Additive noise remains a useful but incomplete baseline. |
| Telephone bandpass | `0.2538` | `0.1418` | Similar scale to AAC. |
| AAC 32k | `0.2530` | `0.1371` | Highest codec UED in the 100-clip run. |
| MP3 32k | `0.2458` | `0.1125` | Codec degradation is consistent but not dominant. |
| Opus 24k | `0.2063` | `0.1037` | Lowest codec UED here. |
| Brown 16 dB | `0.0948` | `0.0493` | Mild for this slice. |
| Clipping 0.50 | `0.0000` | `0.0703` | Still not a useful FR stressor at this threshold. |

Scale-up run size: 100 FLEURS-fr + 100 CV21-zh clips x 13 corruptions = 2600 rows.

Plot: [`degradation_mean_ued.png`](../runs/degradation_full_100_20260517/plots/degradation_mean_ued.png)

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

### Advanced Chunking / Window Placement

This pass adds overlap consistency by absolute token index, distance-from-window-edge curves, comparison against both non-overlap and high-overlap references, hidden-state averaging before LFQ, distance/voter/entropy confidence weights, and commit-latency curves.

| Policy | Aggregation | Reference | Mean Mismatch |
|---|---|---|---:|
| 30s window, 15s stride | first | non-overlap majority | `0.3191` |
| 30s window, 15s stride | edge + voter weighted | non-overlap majority | `0.4012` |
| 30s window, 5s stride | first | non-overlap majority | `0.4535` |
| 30s window, 1s stride | first | non-overlap majority | `0.4878` |
| 30s window, 1s stride | edge + voter weighted | non-overlap majority | `0.5733` |
| 30s window, 1s stride | hidden edge + voter | non-overlap majority | `0.6096` |
| 30s window, 1s stride | edge + voter weighted | high-overlap edge-voter | `0.0000` |
| 30s window, 5s stride | edge + voter weighted | high-overlap edge-voter | `0.3282` |
| 30s window, 15s stride | edge + voter weighted | high-overlap edge-voter | `0.4204` |

| Policy | Mean Overlap Disagreement |
|---|---:|
| 30s window, 15s stride | `0.3883` |
| 30s window, 5s stride | `0.4881` |
| 30s window, 1s stride | `0.5624` |

| Commit Latency | Best vs High-Overlap Consensus | Coverage | Best vs Non-Overlap | Coverage |
|---:|---:|---:|---:|---:|
| 1s | `0.7748` | `0.6133` | `0.7954` | `0.6133` |
| 3s | `0.7214` | `0.6400` | `0.7542` | `0.6400` |
| 5s | `0.7035` | `0.6667` | `0.7264` | `0.6667` |
| 10s | `0.4453` | `0.7333` | `0.6349` | `0.7333` |
| 15s | `0.2524` | `0.8000` | `0.6051` | `0.8000` |

Plots: [`advanced_reference_comparison.png`](../runs/chunking_advanced_20260517/plots/advanced_reference_comparison.png), [`advanced_edge_distance.png`](../runs/chunking_advanced_20260517/plots/advanced_edge_distance.png), [`advanced_commit_latency.png`](../runs/chunking_advanced_20260517/plots/advanced_commit_latency.png), [`advanced_overlap_disagreement.png`](../runs/chunking_advanced_20260517/plots/advanced_overlap_disagreement.png)

Result: the best policy depends on the serving contract. If the goal is to reproduce the offline non-overlap tokenizer, the least bad tested option is still 15s-stride first/majority at `0.3191` mismatch; hidden-state aggregation and confidence weighting make that target worse. If the goal is a new high-overlap streaming policy, stride-1s edge+voter weighting is internally stable by construction, and 15s commit latency gets mismatch down to `0.2524` versus that high-overlap consensus with `0.8000` coverage. Recommendation: do not describe StableToken as true streaming under the released 30s non-causal tokenizer; serve either fixed 30s non-overlap chunks for reproducibility or explicitly define a separate high-overlap consensus tokenizer with documented latency and coverage.

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

### Distribution Drift Under Hard Degradations

| Group | Entropy Bits | Transition Entropy Bits | Unique Tokens | Dead-Token Rate |
|---|---:|---:|---:|---:|
| FR clean | `11.461` | `13.058` | `4237` | `0.483` |
| FR reverb | `11.506` | `13.075` | `4321` | `0.473` |
| FR babble | `11.749` | `13.300` | `4665` | `0.431` |
| FR competing speech | `11.724` | `13.312` | `4651` | `0.432` |
| ZH clean | `11.023` | `12.123` | `3163` | `0.614` |
| ZH reverb | `11.009` | `12.122` | `3125` | `0.619` |
| ZH babble | `11.093` | `12.229` | `3253` | `0.603` |
| ZH competing speech | `11.096` | `12.204` | `3242` | `0.604` |

| Clean -> Corruption KL | FR KL Bits | ZH KL Bits |
|---|---:|---:|
| Gaussian 25 dB | `1.124` | `0.913` |
| Reverb small room | `1.605` | `1.726` |
| Babble 4 speakers 10 dB | `2.456` | `2.353` |
| Competing speech 16 dB | `1.603` | `1.574` |

Cross-language KL `FR -> ZH` decreased from `13.018` on clean speech to `10.371` under babble and `11.210` under competing speech, suggesting speech-like interference pushes both languages toward a more shared noisy token distribution.

Plots: [`distribution_entropy.png`](../runs/distribution_degradation_20260517/plots/distribution_entropy.png), [`distribution_zipf.png`](../runs/distribution_degradation_20260517/plots/distribution_zipf.png)

Result: Gaussian noise understates distribution drift. Babble is the largest clean-to-corruption drift in both languages and also increases unique-token usage, especially for French. Future token-collapse/drift claims should include hard speech-like corruptions, not only additive noise.

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

### ASR Degradation Smoke

This downstream smoke uses `openai/whisper-tiny` directly on corrupted audio. It is not a StableToken-token SpeechLLM evaluation, but it provides a lightweight WER/CER reference for the same degradation suite.

| Corruption | FR WER | FR CER | ZH CER | Note |
|---|---:|---:|---:|---|
| Clean | `0.4201` | `0.2120` | `0.6072` | Whisper-tiny is a weak baseline on these slices. |
| Gaussian 25 dB | `0.5121` | `0.2657` | `0.6077` | Small ASR degradation for zh CER in this tiny slice. |
| Reverb small room | `0.4743` | `0.2262` | `0.6151` | Mild-to-moderate ASR degradation. |
| Babble 4 speakers 10 dB | `0.7260` | `0.3811` | `0.7510` | Strong downstream hit. |
| Competing speech 16 dB | `1.0545` | `0.7200` | `0.7702` | Worst ASR condition in this smoke. |
| Telephone bandpass | `0.4796` | `0.2191` | `0.6042` | Comparable to clean/reverb for tiny model here. |
| AAC 32k | `0.5311` | `0.2448` | `0.6206` | Codec degradation is measurable but not worst. |

Run size: 8 FLEURS-fr + 8 CV21-zh clips x 7 conditions = 112 rows.

Artifacts: [`asr_summary.csv`](../runs/asr_degradation_smoke_20260517/asr_summary.csv), [`asr_item_metrics.csv`](../runs/asr_degradation_smoke_20260517/asr_item_metrics.csv)

For Chinese, WER is not useful without word segmentation; use CER.

### Whisper-Small ASR Smoke

The same ASR degradation config was rerun with `openai/whisper-small` to check whether the strongest downstream conclusion survives a better ASR model.

| Corruption | FR WER | FR CER | ZH CER | Note |
|---|---:|---:|---:|---|
| Clean | `0.1331` | `0.0562` | `0.4920` | Much stronger French baseline than Whisper-tiny. |
| Gaussian 25 dB | `0.1713` | `0.0904` | `0.4220` | Moderate FR degradation. |
| Reverb small room | `0.1808` | `0.0834` | `0.3848` | Moderate FR degradation; zh smoke is noisy. |
| Babble 4 speakers 10 dB | `0.2522` | `0.1217` | `0.4375` | Largest FR WER hit for the stronger ASR model. |
| Competing speech 16 dB | `0.1659` | `0.0826` | `0.4868` | Less severe than babble for FR in this tiny slice. |
| Telephone bandpass | `0.1067` | `0.0406` | `0.4794` | Not harmful for FR here. |
| AAC 32k | `0.1271` | `0.0485` | `0.5039` | Similar to clean FR WER. |

Artifacts: [`asr_summary.csv`](../runs/asr_degradation_small_20260517/asr_summary.csv), [`asr_item_metrics.csv`](../runs/asr_degradation_small_20260517/asr_item_metrics.csv)

### Whisper-Small ASR 100-Clip Scale-Up

| Corruption | FR WER | FR CER | ZH CER | Note |
|---|---:|---:|---:|---|
| Clean | `0.1165` | `0.0468` | `0.3760` | Stronger, more stable ASR baseline. |
| Gaussian 25 dB | `0.1637` | `0.0692` | `0.3312` | Moderate FR degradation. |
| Reverb small room | `0.1428` | `0.0636` | `0.3626` | Mild-to-moderate FR degradation. |
| Babble 4 speakers 10 dB | `0.2885` | `0.1540` | `0.5059` | Largest downstream ASR hit in both FR WER and ZH CER. |
| Competing speech 16 dB | `0.1409` | `0.0636` | `0.3873` | Much less severe for Whisper-small than token UED suggested. |
| Telephone bandpass | `0.1162` | `0.0480` | `0.3880` | Near-clean FR WER. |
| AAC 32k | `0.1217` | `0.0488` | `0.3807` | Near-clean FR WER. |

Scale-up run size: 100 FLEURS-fr + 100 CV21-zh clips x 7 conditions = 1400 rows.

Artifacts: [`asr_summary.csv`](../runs/asr_degradation_small_100_20260517/asr_summary.csv), [`asr_item_metrics.csv`](../runs/asr_degradation_small_100_20260517/asr_item_metrics.csv)

### Token-vs-ASR Comparison

The shared 100-clip token and Whisper-small ASR runs were joined for the overlapping corruptions.

| Corruption | Lang | Token UED | ASR WER | ASR CER |
|---|---|---:|---:|---:|
| Babble 4 speakers 10 dB | fr | `0.6071` | `0.2885` | `0.1540` |
| Babble 4 speakers 10 dB | zh | `0.4170` | `0.9700` | `0.5059` |
| Competing speech 16 dB | fr | `0.4817` | `0.1409` | `0.0636` |
| Competing speech 16 dB | zh | `0.2750` | `0.9600` | `0.3873` |
| Reverb small room | fr | `0.3661` | `0.1428` | `0.0636` |
| Reverb small room | zh | `0.2399` | `0.9100` | `0.3626` |
| Gaussian 25 dB | fr | `0.2635` | `0.1637` | `0.0692` |
| Gaussian 25 dB | zh | `0.1308` | `0.9000` | `0.3312` |
| AAC 32k | fr | `0.2530` | `0.1217` | `0.0488` |
| AAC 32k | zh | `0.1371` | `0.9300` | `0.3807` |
| Telephone bandpass | fr | `0.2538` | `0.1162` | `0.0480` |
| Telephone bandpass | zh | `0.1418` | `0.9100` | `0.3880` |

Artifacts: [`token_asr_degradation_comparison.csv`](../analysis/token_asr_degradation_comparison.csv), [`token_asr_degradation_comparison.json`](../analysis/token_asr_degradation_comparison.json)

Interpretation: token UED is a useful robustness diagnostic, but it is not a direct proxy for downstream ASR impact. Babble is high on both axes. Competing speech is high token UED but modest French Whisper-small WER, suggesting either the tokenizer changes are not always ASR-relevant or Whisper-small ignores some competing content.

### StableToken-Encoder ASR Probe

To test whether WER can be measured through the released tokenizer without a new training run, the ASR runner was extended to replace `openai/whisper-large-v3`'s encoder with the released StableToken LFQ encoder, precompute the quantized encoder outputs, and decode with the stock Whisper decoder.

| Corruption | FR WER | FR CER | ZH CER | Note |
|---|---:|---:|---:|---|
| Clean | `1.0000` | `1.0000` | `1.0000` | Predictions were empty or punctuation. |
| Reverb small room | `1.0000` | `1.0000` | `1.0000` | Same failure mode. |
| Babble 4 speakers 10 dB | `1.0000` | `1.0000` | `1.0000` | Same failure mode. |
| Competing speech 16 dB | `1.0000` | `1.0000` | `1.0000` | Same failure mode. |

Run size: 2 FLEURS-fr + 2 CV21-zh clips x 4 conditions = 16 rows.

Artifacts: [`asr_summary.csv`](../runs/asr_stabletoken_smoke_20260517/asr_summary.csv), [`asr_item_metrics.csv`](../runs/asr_stabletoken_smoke_20260517/asr_item_metrics.csv)

Result: zero-shot decoder replacement is not a usable StableToken-token ASR evaluation. This is a useful negative result: reviewer-grade downstream WER through StableToken needs a trained adapter/head or LoRA, not just a stock Whisper decoder attached to the released tokenizer.

## What Failed Or Was Infeasible

| Item | Status | Reason / Fix |
|---|---|---|
| Initial audio loading through `torchaudio.load` | Fixed | Local TorchCodec/FFmpeg libraries were incomplete. The runner and training harness now try `soundfile` first and use `torchaudio` as fallback. |
| MP3/Opus/AAC compression run | Fixed and executed | Added `imageio-ffmpeg==0.6.0` fallback because system `ffmpeg` is not on PATH. Full codec rows are in `degradation_full_20260517`. |
| StableToken-encoder ASR integration | Fixed for smoke | Whisper generation padded features in a way that broke the custom encoder mask; the runner now pads to the StableToken stride and precomputes encoder outputs before decoder generation. |
| Full training ablations L8/L12/L16/L20/L24 | Pilot only | L8/L16/L24 each ran one matched step. Full scientific runs still need matched data scale, seeds, steps, and downstream evaluation. |
| Architecture-vs-augmentation factorial | Pilot only | All five variants ran one matched step. Full conclusions need real training budget, matched seeds/steps, and downstream evaluation. |
| ASR WER/CER | Smoke complete | Direct Whisper downstream ASR and zero-shot StableToken-encoder ASR are wired. A trained StableToken-token ASR adapter/head remains future work. |
| Hidden-state chunk aggregation | Executed negative result | Pre-LFQ hidden averaging was accessible and ran, but it worsened mismatch against the offline non-overlap reference and added substantial runtime. |
| SER, speaker/prosody, TTS metrics | Not executed | No stable downstream evaluation pipeline was wired in this pass. |
| SpeechLLM QA/translation/summarization/instruction tasks | Not executed | Needs an adapter/LoRA or full SpeechLLM pipeline; recommended after tokenizer-level issues are scaled. |

## Interpretation

The sanity run supports the basic reproducibility claims for released inference: 25 Hz, 750 tokens for 30 seconds, and 8192 vocabulary.

The most important negative finding is chunking instability. StableToken is robust to some perturbations, but the Whisper-style non-causal 30s window can still produce substantially different token sequences for the same absolute time span under different streaming policies. The scaled and advanced runs confirm this beyond a single example. Simple first/majority/center aggregation does not solve it, and neither do hidden-state averaging or voter/entropy confidence weighting when the target is the released offline non-overlap tokenizer.

The degradation suite says the next robustness budget should go to babble, competing speech, and reverb, not just additive noise. The 100-clip token run confirms this ranking. The 100-clip Whisper-small ASR run agrees that babble is the strongest downstream ASR condition, but competing speech is much less severe for direct Whisper ASR than for token UED, which is an important mismatch to investigate. The StableToken-encoder ASR probe shows that downstream WER through the tokenizer cannot be obtained by simply attaching the stock Whisper decoder; it needs a trained adapter/head.

The distribution results show broad but sparse code usage, with meaningful language/domain drift. This is not collapse, but it does mean multilingual and noisy/clean analyses need stratified reporting. The degradation distribution run strengthens that point: babble and competing speech change token usage more than Gaussian noise and partly compress cross-language differences into a shared noisy-token regime.

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

## Voter / Clean-Noisy Pilot

These are one-step feasibility pilots for voter count and clean:noisy branch ratio. They validate that all requested N=5 ratios and N in `{1,3,5,7}` run under matched config.

| Variant | Voters | Clean Inputs | Train Loss | Runtime | Saved Model |
|---|---:|---:|---:|---:|---|
| `n5_clean5_noisy0` | 5 | 5 | `10.0948` | `0.763s` | false |
| `n5_clean4_noisy1` | 5 | 4 | `10.1706` | `0.709s` | false |
| `n5_clean3_noisy2` | 5 | 3 | `10.2009` | `0.701s` | false |
| `n5_clean2_noisy3` | 5 | 2 | `10.2554` | `0.704s` | false |
| `n5_clean1_noisy4` | 5 | 1 | `10.2594` | `0.700s` | false |
| `n5_clean0_noisy5` | 5 | 0 | `10.1938` | `0.822s` | false |
| `n1_clean1` | 1 | 1 | `9.6039` | `0.701s` | false |
| `n3_clean2` | 3 | 2 | `8.8181` | `0.694s` | false |
| `n5_clean3` | 5 | 3 | `10.2009` | `0.708s` | false |
| `n7_clean4` | 7 | 4 | `8.9456` | `0.707s` | false |

Result files live under `experiments/training_runs/training_voter_pilot_*/result.json`.

Interpretation: the voter matrix is runnable, including the 0-clean and N=7 cases. One-step losses are not evidence that N=3 or N=7 is better; the next useful run needs enough steps to produce comparable checkpoints, plus token entropy/dead-token/runtime and downstream robustness evaluation.

## Recommended Next Runs

1. Build the StableToken-token ASR head or adapter so WER/CER is measured through the tokenizer itself, then rerun the fixed corruption suite: babble, competing speech, reverb, Gaussian, telephone bandpass, and AAC.

2. For pseudo-streaming, either keep fixed non-overlap 30s chunks for reproducibility or define a separate high-overlap consensus tokenizer. The next chunking experiment should test training-time chunk-position augmentation, because inference-only aggregation did not recover the offline token stream.

3. Extend the quantizer-placement pilot from one step to a small matched run: L8, L16, L24, fixed data, fixed steps, fixed seeds, then evaluate the resulting checkpoints on UED/degradation before adding L12/L20.

4. Extend the architecture-vs-augmentation factorial from one-step pilots to matched short runs that save checkpoints, then evaluate UED/nUED, token entropy, ASR WER, and SER.

5. Extend the voter/ratio pilot from one-step checks to matched short runs, then report robustness vs clean performance, token entropy/dead tokens, runtime, and downstream ASR/SER.

6. Add SER and speaker/prosody probes on the same clean/noisy/degradation splits before investing in full SpeechLLM experiments.

7. For SpeechLLM usefulness, start with a small adapter/LoRA controlled experiment on spoken QA or noisy spoken dialogue understanding, using identical data and steps across tokenizer variants.
