# QA Report — 2026-05-22 (LIVE / COMPREHENSIVE)

> Supersedes the prior STATIC-only report. All five checks executed against on-disk artifacts and live checkpoint loads.

## Verdict: PASS (with 1 INFO deviation, documented)

## Summary
- (a) Data leakage: PASS — train ∩ test path intersection = 0; all sampled `.npy` paths exist; LOPO loop excludes held-out person by manifest filter.
- (b) Sample-rate consistency: PASS — preprocess `fs=104.0` only; LIMU-BERT `up=5, down=26` (→20 Hz); HARNet10 `up=15, down=52` (→30 Hz); no `fs=100` drift anywhere.
- (c) Input units: PASS — `.npy` carries mg/dps as expected; LIMU-BERT applies mg→g and dps→rad/s post-resample; HARNet10 drops gyro at load and applies mg→g exactly once.
- (d) Checkpoint loadability: PASS — all 4 main checkpoints load via `Predictor`, hub-backbone loads cleanly (no `_RandomBackbone` fallback), smoke-test outputs are finite `(label, float)` tuples with zero warnings.
- (e) Artifact completeness: PASS — 0 missing, 0 empty, 0 stale, all JSON valid, sampled PNGs open in PIL.
- INFO: HARNet10 windowing deviates from CLAUDE.md spec (1 window/file, 62% zero-pad). Intentional and documented in `src/harnet10.py:69-76`. Implications discussed below.

## Results summary

|                     | LIMU-BERT (6-ch, 20 Hz, 4-layer Tx) | HARNet10 (3-ch, 30 Hz, 1D CNN) |
|---------------------|-------------------------------------|--------------------------------|
| Frozen window F1    | 0.9268                              | 0.9336                         |
| Frozen file F1      | 0.9699                              | 0.9336 (== window; 1 win/file) |
| Partial window F1   | 0.9773                              | 0.9947                         |
| Partial file F1     | 0.9951                              | 0.9947 (== window; 1 win/file) |
| LOPO window F1      | 0.576 ± 0.189                       | 0.681 ± 0.084                  |
| LOPO file F1        | 0.571 ± 0.209                       | 0.681 ± 0.084                  |

All numbers read live from `artifacts/{model}/metrics/{mode}_metrics.json` and `lopo_summary.json` — not copied from the prompt.

## Details

### (a) Data Leakage
- Manifest: `artifacts/preprocessed/manifest.csv`, 976 rows. Counts: `{train: 780, test: 196}`.
- `set(train.path) & set(test.path)` → **size 0**.
- 10 random `npy_path` samples all exist on disk.
- 5 persons, 3 labels, `actual_fs` is uniformly 104.0 (by-design — `sample_idx` is a counter, see prior memory).
- LOPO loop in `src/limu_bert.py:846-847` (and mirrored in `src/harnet10.py`) filters `manifest_all["person"] != person` for train and `== person` for test — held-out person's files cannot leak.
- 5 `lopo_frozen_{person}.pt` checkpoints exist for both models.

### (b) Sample-Rate Consistency

| Stage              | Constant                                   | Source                              |
|--------------------|--------------------------------------------|-------------------------------------|
| Source rate        | `fs=104.0` (lowpass, order 4, 30 Hz cut)   | `src/preprocess.py:117`             |
| LIMU-BERT resample | `up=5, down=26` → 20.0 Hz exact            | `src/limu_bert.py:76-77, 151`       |
| LIMU-BERT windows  | `WINDOW_SIZE_SRC=208, HOP_SIZE_SRC=52`     | `src/limu_bert.py:80-81`            |
| LIMU-BERT SEQ_LEN  | 40                                         | `src/limu_bert.py:83`               |
| HARNet10 resample  | `up=15, down=52` → 30.0 Hz exact           | `src/harnet10.py:66-67, 140`        |
| HARNet10 windows   | `WINDOW_SIZE_SRC=0, HOP_SIZE_SRC=0` (BYPASS) | `src/harnet10.py:77-78`           |
| HARNet10 SEQ_LEN   | 300 (zero-padded from ~115)                | `src/harnet10.py:80`                |

- `grep -nE "fs=100|=100\b"` against all three scripts: 0 matches except `MAX_EPOCHS=100` comments — no rate drift.
- `config.json` for both models records the same constants verbatim (verified live).

### (c) Input Units
- Sampled 3 random `.npy` files: shape `(400, 6)`, dtype `float32`. Accel stds ~280–700 mg, gyro stds ~25–100 dps — consistent with raw mg/dps as documented in the contract.
- LIMU-BERT (`src/limu_bert.py:87-88, 144-145`): `out[:, :3] *= 1/1000` (mg→g), `out[:, 3:] *= π/180` (dps→rad/s). Applied once, post-resample.
- HARNet10 (`src/harnet10.py:84, 134`): `out *= 1/1000` (mg→g). No dps→rad/s because gyro is dropped at load (`arr3 = arr[:, :3]` at `src/harnet10.py:181`).
- `APPLY_ZSCORE_AFTER_UNIT_CONVERT=False` in both — scaler is fit on the train split only, no test leakage (and effectively a no-op pass-through).

### (d) Checkpoint Loadability (live smoke test)

| Checkpoint                                | Load    | Warnings | Smoke `predict` output           |
|-------------------------------------------|---------|----------|----------------------------------|
| `limu_bert_frozen.pt`                     | strict  | 0        | `('none', 0.5761)` — finite      |
| `limu_bert_partial.pt`                    | strict  | 0        | `('none', 0.9628)` — finite      |
| `harnet10_frozen.pt`                      | strict  | 0        | `('none', 0.8290)` — finite      |
| `harnet10_partial.pt`                     | strict  | 0        | `('none', 0.4115)` — finite      |

- HARNet10 successfully resolved torch.hub backbone (`OxWearables/ssl-wearables`) from cache (`~/.cache/torch/hub/OxWearables_ssl-wearables_main`). `feature_dim=1024` matches checkpoint metadata → strict load path taken, no `_RandomBackbone` fallback, no `strict=False`. Reported "131 Weights loaded" is the hub's normal stdout.
- Outputs returning `'none'` is correct threshold-gate behavior on random input (confidence < `confidence_threshold_5pct_ft`); not a failure.
- No files were written by Predictor construction or `.predict()`.

### (e) Artifact Completeness
- 976 `.npy` files in `artifacts/preprocessed/` (matches manifest row count exactly).
- Every promised path under `artifacts/{limu_bert,harnet10}/` exists, non-empty, mtime ≥ `manifest.csv` mtime (`1779431539.28`). 0 missing, 0 empty, 0 stale.
- All 8 metric JSONs valid; per-class keys = `['hook', 'straight', 'upper']` for all four `{model,mode}` pairs.
- All 5 sampled PNGs (one per type, both models) verified with PIL `Image.verify()`.

## Notable findings

- **Generalization gap is large for both models.** Standard 80/20 file-level split flatters both — LIMU-BERT partial drops from 0.995 file F1 (standard) to 0.571 (LOPO). HARNet10 frozen drops from 0.934 to 0.681. Per CLAUDE.md anti-goal "do not report 80/20 without LOPO next to it," always co-report.
- **HARNet10 has tighter LOPO std (0.084) than LIMU-BERT (0.189)** despite the 1-window-per-file constraint. Plausible reasons: (i) the pretrained accel-only HARNet10 backbone may be less brittle to per-person idiosyncrasies than LIMU-BERT's transformer over 6 channels; (ii) 1 win/file means no within-file variance — each file contributes one point, dampening tail behavior; (iii) zero-padding may regularize toward a less person-specific decision surface. Worth empirical follow-up.
- **`chi` is the hardest held-out person for both models** (LIMU-BERT: 0.351 window F1, 0.345 file F1; HARNet10: 0.519 window F1). Suggests stylistic outlier in this user's punches.
- **Apples-to-apples claim is partially compromised at the file level.** LIMU-BERT uses sliding windows (~4 windows/file) and gets a meaningful majority-vote uplift at the file level (0.927 → 0.970 frozen). HARNet10 file F1 == window F1 by construction. Window-level comparison remains fair; file-level comparison favors LIMU-BERT structurally.
- **HARNet10 sees ~62% zero-padding** per input (185 zeros, 115 signal samples). High scores suggest the model is robust to this, but it diverges from the pretraining contract and may limit ceiling performance.

## Recommended Actions
1. (Communication) When publishing the LIMU-BERT vs HARNet10 comparison, note in figure captions that HARNet10's file F1 equals window F1 by construction. The fair head-to-head is at the **window level**.
2. (Future work) If recordings can be lengthened to ≥10 s, restore HARNet10 to the spec'd `WINDOW_SIZE_SRC=1040, HOP_SIZE_SRC=260` sliding windows for a true apples-to-apples comparison. The current deviation is documented but not strictly compliant with `instructions.md`.
3. (Analysis) Investigate why `chi` collapses both models. Compare per-person feature distributions or run targeted augmentation (e.g., higher jitter scale) for chi-like samples.
4. (Minor) Consider promoting `confidence_threshold_5pct_ft` to a top-level `threshold` key in `*_metrics.json` for easier programmatic access — the Predictor already handles the nested key correctly, but the schema is verbose.
