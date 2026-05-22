---
name: project-pipeline-contract
description: Canonical fs/window/unit-conversion/split rules for the punch classifier (preprocess + limu_bert + harnet10) — the spec QA enforces.
metadata:
  type: project
---

Canonical contract for the punch-classifier rebuild (CLAUDE.md `instructions.md`).

**Why:** This rebuild exists because of a prior 100-vs-104 Hz bug. Every sample-rate constant and resample ratio is load-bearing; any drift back to 100 Hz / `q=5` / `fs=100` in `src/` is a FAIL.

**How to apply:** When running check (b) sample-rate consistency, compare findings to this table verbatim. When running (c) input units, confirm both conversions happen exactly once (preprocess writes raw filtered mg/dps, models convert after resampling).

- Source rate: **104 Hz**. Preprocess applies Butterworth lowpass at 30 Hz with `fs=104.0`, order 4.
- LIMU-BERT: `up=5, down=26` (104→20.0 exact). 6-channel. Source window 208/52 → 40-sample sequences. Unit conversion: accel mg→g (×1/1000), gyro dps→rad/s (×π/180). `APPLY_ZSCORE_AFTER_UNIT_CONVERT=False`.
- HARNet10: `up=15, down=52` (104→30.0 exact). 3-channel accel only — gyro dropped immediately after `.npy` load. Source window 1040/260 → 300-sample sequences. Unit conversion: accel mg→g only.
- Split: stratified 80/20 file-level, `random_state=42`, written to `manifest['split']`. Both model scripts MUST consume the manifest's `split` column and never call `train_test_split` on windows. Leakage key = `path`.
- Val carve-out inside train: file-level (`split_train_val_by_file`), `VAL_FILE_FRACTION=0.10`.
- LOPO: 5 people (`adnan, ahmed, chi, mercy, novel`). Holds out one person, trains on the other four. Test windows for the held-out person never touch train.
- Manifest schema: `path, person, label, split, actual_fs, n_samples, npy_path`. Per-file `.npy` is shape `(N, 6)` dtype `float32`, channel order `[ax, ay, az, gx, gy, gz]`.

By-design quirk: `sample_idx` is a counter, not a timestamp, so `estimate_fs` returns the nominal `TARGET_FS=104.0` whenever deltas are unit-1. Every manifest row in the contracted dataset will show `actual_fs=104.0` — this is INFO, not WARN.

Related: [[project-promised-artifacts]].
