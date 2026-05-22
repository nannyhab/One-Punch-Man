---
name: project-sample-idx-is-counter
description: The CSV `sample_idx` column is a monotonic counter (deltas always 1), not a timestamp — so real per-file sample-rate verification from CSV alone is impossible.
metadata:
  type: project
---

The raw IMU CSVs have schema `sample_idx, acc_x_mg, acc_y_mg, acc_z_mg, gyr_x_dps, gyr_y_dps, gyr_z_dps`. `sample_idx` is a pure 0..N-1 counter; its diffs are always 1. There is no wall-clock timestamp column.

**Why:** This means `estimate_fs` cannot derive a real Hz value from a single CSV — it can only report the nominal 104 Hz when median dt == 1. The "100 vs 104 Hz" bug that motivated this whole rebuild was discovered out-of-band (from recorder firmware / wall-clock duration vs sample count), not from the CSVs themselves.

**How to apply:**
- Do not promise to "verify sample rate from CSV" — be honest that, with the current schema, this is a no-op unless the recorder format changes.
- If the recorder later adds a millisecond/second timestamp column, the heuristic in `estimate_fs` already handles it (median dt > 0.5 → ms, < 0.1 → s).
- Real fs verification has to come from another channel (file metadata, recorder logs, total-duration-vs-N-samples computed at capture time). Suggest adding a `timestamp_ms` column to future recordings.
- The QA agent should NOT flag "all actual_fs values == 104.0" as suspicious — that is correct behavior given the schema.
