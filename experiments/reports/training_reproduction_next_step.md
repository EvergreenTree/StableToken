# StableToken Training Reproduction Next Step

Date: 2026-05-18

## What Is Implemented

This step turns the previous training scaffold into a runnable LFQ/Voting-LFQ tokenizer training path.

- `experiments/train_lfq_tokenizer.py` now trains a Whisper encoder-decoder with `WhisperLFQEncoder` inserted at a configurable encoder layer.
- LFQ uses 13-bit binary codes by default, giving an 8192-entry tokenizer vocabulary.
- Average pooling with kernel size 2 is applied after Whisper's 50 Hz encoder stride, producing 25 Hz token streams.
- `VotingLFQ` supports N branches, including N=1/3/5, sign binarization with STE, averaged branch quantized bits during training, and majority-voted signs for token IDs.
- The loss is ASR cross entropy plus configurable commitment, codebook entropy, and optional branch-consensus losses.
- Augmentation supports Gaussian, pink, brown, bit-crush, and optional real-world noise manifests/directories.
- Training saves a reloadable encoder-only `tokenizer/` checkpoint plus `lfq_tokenizer_metadata.json`.
- `experiments/eval_lfq_tokenizer.py` evaluates trained checkpoints with clean tokens, perturbed tokens, UED, normalized edit distance, and token usage histograms.
- Five medium-based ablation configs are available under `experiments/configs/lfq_ablation_*.yaml`.
- `experiments/make_librispeech_manifest.py` creates JSONL manifests from LibriSpeech-style folders.

Smoke verification used:

```bash
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/lfq_reproduction_smoke.yaml
/data/venv/bin/python experiments/eval_lfq_tokenizer.py --config experiments/configs/lfq_eval_smoke.yaml
```

Observed smoke training result: `openai/whisper-tiny`, 2 steps, 4 train rows, train loss `8.5837`, saved encoder tokenizer `true`.

Observed smoke eval result: 2 dev clips, 750 clean tokens per clip, Gaussian 25 dB mean UED `0.5740`, pink 20 dB mean UED `0.6760`, bit-crush 10-bit mean UED `0.3400`.

## Full Medium Pilot Result

After the smoke implementation landed, the first full medium pilot was launched with:

```bash
/data/venv/bin/python experiments/train_lfq_tokenizer.py --config experiments/configs/lfq_ablation_multi_aug_consensus.yaml --run-name lfq_full_multi_aug_consensus_20260518
```

The run used `openai/whisper-medium`, 2000 FLEURS-fr training rows, 1000 optimizer steps, N=5 voters, noisy branches, and consensus loss.

Observed training result:

- Runtime: `2162.1s`
- Train loss: `-2.7280`
- Final logged loss at step 1000: `-4.7764`
- Final grad norm: `12.52`
- Saved encoder tokenizer: `true`

The negative total loss is possible because the codebook entropy term is an auxiliary optimization objective that can be negative and can dominate the positive CE/commitment components. It should not be read like a plain ASR CE curve.

Smoke evaluation on the same two FLEURS-fr dev clips showed:

- Clean tokens: 750 per clip
- Clean unique token rate: `0.1447`
- Gaussian 25 dB mean UED: `0.9020`
- Pink 20 dB mean UED: `0.9000`
- Bit-crush 10-bit mean UED: `0.6653`

Interpretation: the medium pilot uses the 8192-code vocabulary much more broadly than the tiny smoke run, but it does not yet produce stable perturbation-invariant tokens on this tiny evaluation. That is a useful failure signal: the next run should compare all five ablations on a larger held-out manifest before treating consensus/noise augmentation as helpful.

## Five-Way Ablation Matrix Result

The matched matrix was then run for all five planned variants with `openai/whisper-medium`, 2000 FLEURS-fr training rows, 1000 steps, seed `42`, and a shared 50-clip FLEURS-fr eval manifest.

| Variant | Voters | Augmentation | Consensus | Final logged loss | Clean unique token rate | Gaussian 25 dB UED | Pink 20 dB UED | Bit-crush 10-bit UED | Mean all UED |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| single_clean | 1 | no | 0.00 | `-4.6840` | `0.7476` | `0.8508` | `0.8457` | `0.4452` | `0.7139` |
| single_aug | 1 | yes | 0.00 | `-4.6965` | `0.7285` | `0.8333` | `0.8326` | `0.3947` | `0.6869` |
| multi_clean_no_consensus | 5 | no | 0.00 | `-4.7713` | `0.7982` | `0.9161` | `0.9135` | `0.5943` | `0.8080` |
| multi_aug_no_consensus | 5 | yes | 0.00 | `-4.7802` | `0.8185` | `0.9126` | `0.9085` | `0.5719` | `0.7977` |
| multi_aug_consensus | 5 | yes | 0.25 | `-4.7764` | `0.7946` | `0.9116` | `0.9038` | `0.5756` | `0.7970` |

Findings:

- All five models trained and saved reloadable encoder tokenizers.
- All five learned broad code usage on the 50-clip eval, with clean unique token rates from `0.7285` to `0.8185` of the 8192-code vocabulary.
- The best perturbation stability in this matrix is `single_aug`, not a multi-branch model. It improves mean all-UED by about `0.0271` absolute over `single_clean`.
- Multi-branch voting broadens code usage, but worsens UED substantially at this training scale. `multi_clean_no_consensus` is the worst row with mean all-UED `0.8080`.
- Augmentation helps inside both single-branch and multi-branch settings, but only modestly.
- Consensus loss slightly improves multi-branch noisy UED versus multi-branch augmentation without consensus (`0.7970` vs `0.7977` mean all-UED), but the effect is tiny and does not recover the single-branch baseline.

Interpretation: the pipeline can now run the intended ablations, but this 1000-step French-only pilot does not support the paper-style intuition that Voting-LFQ + noise + consensus automatically improves token stability. At this scale, the simpler single-branch augmented LFQ is the strongest variant by UED, while multi-branch voting appears undertrained or poorly calibrated for perturbation-invariant token extraction.

Primary artifacts:

- `experiments/analysis/lfq_ablation_matrix_summary.csv`
- `experiments/analysis/lfq_ablation_matrix_summary.json`
- `experiments/runs/lfq_matrix_*_eval_20260518/`
- `experiments/training_runs/lfq_full_*_20260518_*/`

## What Remains Non-Faithful To The Paper

This is not a paper-number reproduction.

- The initial smoke run uses `openai/whisper-tiny`, not `openai/whisper-medium`, and only runs two optimizer steps.
- One `openai/whisper-medium` ablation has now run for 1000 steps, but the matched five-way ablation matrix has not been run.
- The data is local FLEURS-fr smoke data, not the original paper's full tokenizer training corpus.
- The current implementation trains with Hugging Face Whisper ASR CE as the reconstruction/semantic objective; it does not reproduce any undisclosed upstream trainer details.
- The evaluator reports tokenizer stability and usage, not downstream SpeechLLM quality.
- ASR WER through StableToken tokens still needs a trained adapter/head; zero-shot decoder replacement was already shown to be invalid in the earlier ledger.

## Data And Compute Needed Next

The next scientifically useful scale-up should keep the five ablation configs fixed except for data size and steps.

- Data: at least tens to hundreds of hours of multilingual, transcribed 16 kHz speech in the JSONL manifest format.
- Compute: `openai/whisper-medium` with gradient checkpointing, batch size 1, grad accumulation 8 or larger, and enough GPU hours to run matched 1k/5k/10k step pilots.
- Evaluation: a held-out manifest shared across all five variants, with fixed perturbation seeds and the same UED/token histogram protocol.
- Tracking: preserve `resolved_config.json`, `result.json`, `lfq_tokenizer_metadata.json`, eval CSV/JSON artifacts, and the exact git commit for every run.

## Blockers And Assumptions

- Upstream does not publish the full tokenizer trainer, so the loss weighting and training schedule are an informed reconstruction.
- Larger training will create heavyweight encoder checkpoints that should remain out of git.
- Real-world noise augmentation is implemented, but no real-noise manifest is checked into the repo.
- The current configs use local `/data/speech2text/...` smoke manifests; external users should generate manifests with `experiments/make_librispeech_manifest.py` or provide equivalent JSONL files.
- No StableToken-token ASR or SpeechLLM adapter is trained yet, so downstream usefulness remains future work.
