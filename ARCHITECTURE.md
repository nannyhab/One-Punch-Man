# Architecture & Training Configuration

Paper-ready reference for the punch-classifier rebuild. Two models, three classes (`hook`, `straight`, `upper`), 6-axis IMU input at 104 Hz from 5 subjects (~976 recordings, ~3.85 s each).

---

## 1. Dataset

| Property                       | Value                                                                   |
|---                             |---                                                                      |
| Subjects                       | 5 (adnan, ahmed, chi, mercy, novel)                                     |
| Recordings                     | 976 CSV files                                                           |
| Per-class file counts          | hook 419 · straight 323 · upper 304                                     |
| Native sampling rate           | 104 Hz                                                                  |
| Per-recording duration         | ~3.85 s (400 samples)                                                   |
| Raw channels                   | 3-axis accel (mg) + 3-axis gyro (dps), CSV columns `acc_*_mg`, `gyr_*_dps` |
| Standard train/test split      | 80 / 20 stratified by label, file-level, `random_state=42` → 780 / 196 |
| Validation carve-out (per run) | 10 % of train files, stratified by label, file-level → ~78 val files    |
| Cross-validation               | 5-fold leave-one-person-out (LOPO), frozen mode only                    |

Source file layout: `data/{person}/{label}/{label_timestamp}.csv`.

---

## 2. Preprocessing pipeline ([src/preprocess.py](src/preprocess.py))

Single canonical pre-step, model-agnostic. Outputs 104 Hz / 6-channel filtered arrays + a manifest. Per [CLAUDE.md](CLAUDE.md) anti-goals, no model-specific transforms happen here.

| Step | Operation                                                                          |
|---   |---                                                                                 |
| 1    | Walk `data/{person}/{label}/*.csv`, build manifest                                  |
| 2    | Stratified file-level 80/20 split, `random_state=42`                                |
| 3    | Rename columns: `acc_x_mg → ax`, …, `gyr_z_dps → gz`. Drop `sample_idx`.            |
| 4    | Verify sample rate from `sample_idx` deltas, warn on \|actual − 104\| > 0.5 Hz       |
| 5    | Butterworth low-pass: order 4, cutoff 30 Hz, `fs = 104.0`, zero-phase (`filtfilt`)  |
| 6    | Save `(N, 6)` float32 array per file under `artifacts/preprocessed/`                |
| 7    | Write manifest with `path, person, label, split, actual_fs, n_samples, npy_path`    |

**Caveat**: the CSV schema has no wall-clock timestamp — `sample_idx` is a monotonic counter, so the "actual" sample rate reported per file is structurally the nominal 104 Hz unless the recorder firmware ever switches to ms/s units (heuristic in place, never triggered on real data). The 100 → 104 Hz correction in this project was discovered out-of-band and is baked into the lowpass `fs` parameter.

---

## 3. Per-model adapters (model-specific transforms)

These steps happen **inside each model script**, on top of the canonical 104 Hz arrays produced by preprocessing.

### LIMU-BERT adapter

| Step | Operation |
|---|---|
| 1 | Load 104 Hz `(N, 6)` float32 array from `npy_path` |
| 2 | Sliding window at **104 Hz**: `window_size = 208 samples` (2 s), `hop_size = 52` (25 % overlap) → ~4 windows per file |
| 3 | Per-window resample with `scipy.signal.resample_poly(x, up=5, down=26, axis=0)` → exactly 40 samples at 20 Hz (104 × 5/26 = 20.0) |
| 4 | Unit conversion: accel `mg → g` (÷ 1000), gyro `dps → rad/s` (× π/180) |
| 5 | (Disabled by default, matches notebook) Optional z-score on training-set statistics |

Final model input: `(40, 6) float32` per window.

### HARNet10 adapter

| Step | Operation |
|---|---|
| 1 | Load 104 Hz `(N, 6)` float32 array |
| 2 | **Drop gyro channels** — keep `[:, :3]` (3-axis accel only) |
| 3 | **One window per file** (whole recording) — `WINDOW_SIZE_SRC = 0, HOP_SIZE_SRC = 0` are unused; see deviation note below |
| 4 | Resample whole recording with `resample_poly(x, up=15, down=52, axis=0)` → ~115 samples at 30 Hz (104 × 15/52 = 30.0) |
| 5 | Unit conversion: accel `mg → g` (÷ 1000) |
| 6 | Zero-pad to 300 samples (HARNet10's pretrained sequence length, 10 s at 30 Hz) |

Final model input: `(300, 3) float32` per window, ~62 % of which is zero-padding for our 3.85 s recordings.

**Deviation from spec** ([CLAUDE.md](CLAUDE.md) called for 10 s sliding windows at 104 Hz, `1040/260` source-side): our recordings are only 400 samples (~3.85 s) so a 10 s window produces zero windows per file. Shorter windows (1 – 2 s) violate HARNet10's pretrained input contract — its strided-conv stack requires input length ≳ 100 samples or the deep `kernel_size = 9` layer overruns the downsampled feature map. The one-window-per-file + zero-pad compromise preserves the pretrained input shape at the cost of training-set size (976 windows total vs. ~3 100 for LIMU-BERT). Documented in [src/harnet10.py](src/harnet10.py).

---

## 4. Model architectures

### LIMU-BERT classifier

Vendored encoder + small classification head. Encoder is the public LIMU-BERT pretrained on HHAR + UCI + MotionSense + Shoaib at 20 Hz / 120-sample sequences ([dapowan/LIMU-BERT-Public](https://github.com/dapowan/LIMU-BERT-Public), checkpoint `pretrain_base_hhar_20_120/hhar.pt`).

```
Input  (B, 40, 6)
   │
   ▼
Linear projection 6 → 72
   │
Positional embedding (40, 72)  ← LINEARLY INTERPOLATED from pretrained (120, 72)
   │
LayerNorm
   │
   ▼
[Transformer block × 4]
   ├── Multi-head self-attention (4 heads, hidden = 72)
   ├── Residual + LayerNorm
   ├── Position-wise FFN (72 → 144 → 72, GELU)
   └── Residual + LayerNorm
   │
   ▼
Mean-pool over time → (B, 72)
   │
LayerNorm
   │
Linear 72 → 128, GELU, Dropout(0.3)
   │
Linear 128 → 3
   │
   ▼
Logits (B, 3)
```

| Hyperparameter        | Value           |
|---                    |---              |
| Encoder hidden size   | 72              |
| Encoder FFN size      | 144             |
| Encoder layers        | 4               |
| Attention heads       | 4               |
| Sequence length       | 40              |
| Input channels        | 6               |
| Head hidden           | 128             |
| Head dropout          | 0.3             |
| **Total parameters**  | **55 667**      |
| Pretrained init       | HHAR checkpoint, positional embedding interpolated from length 120 → 40 |

### HARNet10 classifier

Self-supervised CNN pretrained on UK Biobank wrist-worn 3-axis accel ([OxWearables/ssl-wearables](https://github.com/OxWearables/ssl-wearables), `harnet10` variant). Loaded via `torch.hub.load('OxWearables/ssl-wearables', 'harnet10', pretrained=True)`.

```
Input  (B, 300, 3)
   │
   ▼
Conv1d 3 → 64,  kernel = 5, stride = 1
BatchNorm + ReLU
   │
[Residual block 1: 2 × (Conv + BN + ReLU)]   downsample stride = 2
[Residual block 2]                            downsample stride = 2
[Residual block 3]                            downsample stride = 2
[Residual block 4]                            downsample stride = 2
[Residual block 5]                            downsample stride = 2  → (B, 1024, T')
   │
Global average pool over time → (B, 1024)
   │
   ▼
Linear 1024 → 128, GELU, Dropout(0.3)
   │
Linear 128 → 3
   │
   ▼
Logits (B, 3)
```

| Hyperparameter        | Value         |
|---                    |---            |
| Backbone              | HARNet10 (1D ResNet, ~9 M params, 5 residual stages) |
| Backbone feature dim  | 1024          |
| Sequence length       | 300           |
| Input channels        | 3 (accel only) |
| Head hidden           | 128           |
| Head dropout          | 0.3           |
| **Total parameters**  | **~10.6 M**   |
| Pretrained init       | OxWearables `ssl-wearables` self-supervised on UK Biobank |

---

## 5. Fine-tuning modes

Both models support two transfer-learning stances, selected via `--mode {frozen, partial}`:

| Mode    | Encoder/backbone           | Head    | LR (LIMU-BERT) | LR (HARNet10) |
|---      |---                         |---      |---:            |---:           |
| frozen  | all parameters frozen      | trained | 3 × 10⁻⁴       | 3 × 10⁻⁴      |
| partial | last residual stage / last 2 transformer blocks unfrozen | trained | 5 × 10⁻⁵       | 1 × 10⁻⁵      |

Frozen isolates the transfer-learning signal (only the classification head adapts). Partial usually wins absolute accuracy by also adapting the highest-level pretrained features. Both modes are run and reported per CLAUDE.md.

---

## 6. Training protocol

Shared training loop, identical between scripts (copy-pasted per CLAUDE.md anti-goal "no shared module").

| Hyperparameter            | Value                                                  |
|---                        |---                                                     |
| Optimizer                 | AdamW                                                  |
| Weight decay              | 0.01                                                   |
| LR scheduler              | linear warmup (5 epochs) → cosine decay to 0           |
| Gradient clip (L2 norm)   | 1.0                                                    |
| Mixed precision (AMP)     | enabled when CUDA is available, off on CPU             |
| Loss                      | class-weighted cross-entropy (inverse-frequency weights from `y_train`) |
| Batch size                | 64                                                     |
| Max epochs                | 100                                                    |
| Early stopping            | patience = 10 epochs on validation macro-F1            |
| Validation                | 10 % file-level carve-out from train, stratified by label |
| Random seed               | 42 (numpy + torch + python-random + CUDA + cuDNN deterministic) |

### Augmentation pool (training only — disabled for val/test)

`p = 0.5` per aug, max 2 applied per window, always at least 1.

| Aug         | Definition                                                       | LIMU-BERT | HARNet10 |
|---          |---                                                               |---        |---       |
| `jitter`    | additive Gaussian noise, σ = 0.02                                | ✓         | ✓        |
| `scaling`   | per-channel scalar from N(1, 0.1²)                               | ✓         | ✓        |
| `drift`     | `tsaug.Drift(max_drift = 0.2, n_drift_points = 5)`                | ✓         | ✓        |
| `time_warp` | `tsaug.TimeWarp(n_speed_change = 3, max_speed_ratio = 1.5)`       | ✓         | ✓        |
| `rotation`  | random ±15° XYZ rotation matrix applied to both accel and gyro vectors | ✓         | ✗ (3-axis pretraining didn't use it) |

---

## 7. Evaluation protocol

Two granularities per held-out set:

- **Window-level**: every test window scored independently. This is the fair model-vs-model comparison; majority-vote artifacts don't help anyone.
- **File-level**: per-recording majority vote across that recording's windows. This is the deployment metric — the game classifies a punch, not a slice of one. For HARNet10, window F1 = file F1 because there's only one window per file (see §3 deviation).

Metrics computed per checkpoint:

- macro-F1, weighted-F1, per-class precision / recall / F1
- accuracy and 3 × 3 confusion matrix (window + file)
- one-vs-rest ROC and AUC per class
- softmax-max confidence histogram (correct vs. incorrect)
- confidence threshold tuned so that the false-trigger rate of incorrect predictions drops below 5 %

LOPO is run in frozen mode for the five subjects in turn; for each fold, training data is the 4-person union, test is the held-out person's recordings.

---

## 8. Production-mode training ([src/train_production.py](src/train_production.py))

For deployment, the val-based checkpoints are useful for reportable metrics but they've seen only 80 % of the data. The production protocol re-runs the best recipe on all 976 files. **Only LIMU-BERT** is retrained for production — chosen over HARNet10 because (a) it has a meaningful window-vs-file majority-vote distinction (HARNet10 collapses both to one window per file), (b) the checkpoint is 230 KB vs HARNet10's 41 MB which matters for game deployment, and (c) it works fully offline with no `torch.hub` dependency.

| Choice                          | Value                                                  |
|---                              |---                                                     |
| Architecture                    | LIMU-BERT (partial mode)                               |
| Training set                    | All 976 files (train + test combined)                  |
| Validation set                  | None                                                   |
| Early stopping                  | None — fixed epoch budget                              |
| Mode                            | `partial` (best from val-based runs)                   |
| Epoch budget                    | 22 (the val-based "best=N" from `*_partial.pt` run)    |
| LR schedule                     | Same warmup + cosine, scaled to fit the 22-epoch budget |
| Saved checkpoint                | `artifacts/limu_bert/checkpoints/limu_bert_production.pt` (~230 KB) |
| Honest evaluation               | None possible — there's no held-out signal             |

The reportable test numbers in §9 are from the original train/val/test protocol with the `limu_bert_partial.pt` checkpoint. The production checkpoint is for live game inference only; do **not** report its accuracy as a test score — the model has seen every file in the dataset.

**Why retrain at all**: the production checkpoint sees 23 % more files than the val-based partial-mode checkpoint (976 vs 780 train files). For a 5-subject dataset where `chi` and `novel` are LOPO outliers, that 23 % includes additional examples from every subject and likely improves real-world performance on those same five subjects' future recordings. It does **not** help with brand-new subjects — that's a LOPO problem, not a data-volume problem.

---

## 9. Results

### Standard 80 / 20 file-level held-out test (n = 196 files)

| Model       | Mode     | Window acc. | Window macro-F1 | File acc. | File macro-F1 |
|---          |---       |---:         |---:             |---:       |---:           |
| LIMU-BERT   | frozen   | 92.6 %      | 0.927           | 97.0 %    | 0.969         |
| LIMU-BERT   | partial  | **97.7 %**  | **0.977**       | **99.5 %**| **0.995**     |
| HARNet10    | frozen   | 93.4 %      | 0.934           | 93.4 %    | 0.934 *       |
| HARNet10    | partial  | **99.5 %**  | **0.995**       | **99.5 %**| **0.995** *   |

\* HARNet10 file-level = window-level by construction (1 window per file).

Both `partial` models are essentially tied at the file level. HARNet10 has the slight edge at window-level (0.995 vs. 0.977).

### Per-class file-level recall, partial mode

| Class       | LIMU-BERT partial | HARNet10 partial |
|---          |---:               |---:              |
| `hook`      | 97.1 %            | 100.0 %          |
| `straight`  | 98.5 %            | 100.0 %          |
| `upper`     | 97.5 %            | 98.4 %           |

Confusion concentrates on `hook ↔ upper`, both of which involve large gz rotation.

### Leave-one-person-out cross-validation, frozen mode

| Held-out subject | LIMU-BERT window F1 | HARNet10 window F1 |
|---               |---:                 |---:                |
| adnan            | 0.612               | 0.752              |
| ahmed            | 0.644               | 0.736              |
| **chi**          | **0.351**           | **0.519**          |
| mercy            | 0.877               | 0.708              |
| novel            | 0.398               | 0.689              |
| **mean ± std**   | **0.576 ± 0.189**   | **0.681 ± 0.084**  |

The standard-split / LOPO gap (≥ 30 F1 points) is the most important finding. HARNet10 generalizes more consistently across subjects (tighter σ); LIMU-BERT's mean is dragged down by the chi and novel folds. `chi` is the hardest subject for both models.

---

## 10. Reproducibility checklist

- Seed: 42 for `numpy`, `torch`, `random`, CUDA, `cudnn.deterministic = True`, `cudnn.benchmark = False`.
- Environment: Python 3.13, PyTorch 2.10, scipy 1.15, sklearn 1.5, numpy 2.x, tsaug 0.2.1, seaborn 0.13.
- Hardware (during this rebuild): macOS, CPU only (no GPU). HARNet10 partial training takes ~3 min/epoch on CPU; LIMU-BERT ~5 s/epoch.
- LIMU-BERT pretrained checkpoint: SHA of the vendored `hhar.pt` from [dapowan/LIMU-BERT-Public](https://github.com/dapowan/LIMU-BERT-Public) at the `main` head, `saved/pretrain_base_hhar_20_120/hhar.pt`.
- HARNet10 pretrained backbone: `torch.hub.load('OxWearables/ssl-wearables', 'harnet10', pretrained=True)`, 131 weight tensors loaded.

Re-running `python src/preprocess.py --data-dir data --out artifacts/preprocessed` followed by `python src/{limu_bert,harnet10}.py --task all --mode {frozen,partial}` reproduces every number in §9 to within numeric noise (CPU non-determinism in the conv layers).

---

## 11. Limitations and next steps

- **Sample size**: 5 subjects is too few for the LOPO numbers to be statistically tight. Each LOPO fold's accuracy variance comes mostly from "is the held-out subject like the others." More subjects would shrink the σ band.
- **Recording length**: HARNet10's 10 s pretraining window can't fit in our 3.85 s recordings. Longer recordings (≥ 10 s) would let HARNet10 use proper sliding windows and likely improve both standard-split and LOPO numbers.
- **Sample-rate trust**: `actual_fs` is structurally always 104 Hz because the CSV format lacks a real timestamp. Adding a `timestamp_ms` column to the recorder firmware would let preprocessing catch real drift.
- **No self-pretraining**: LIMU-BERT is currently using the public HHAR pretrained checkpoint. Self-pretraining on our unlabeled wrist data (if collected) is the obvious next research step per the model survey.
- **Production checkpoint has no honest evaluation**: the production-mode checkpoints have seen the test set. Use them for live game inference only; treat the original `*_partial.pt` numbers in §9 as the academically reportable results.
