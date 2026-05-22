# Punch Classifier — Inference Bundle

Two trained models for 3-class punch classification (`hook`, `straight`, `upper`) on wrist-mounted IMU data (6-axis accel+gyro at 104 Hz).

## What's in this bundle

```
inference_example.py             ← copy-paste demo, run this first
requirements.txt                 ← pip install -r this
src/
  limu_bert.py                   ← LIMU-BERT model + Predictor class
  harnet10.py                    ← HARNet10 model + Predictor class
finetuning/vendored/LIMU-BERT-Public/
  models.py                      ← LIMU-BERT transformer architecture
  utils.py                       ← two small helpers models.py needs
artifacts/
  limu_bert/
    checkpoints/
      limu_bert_partial.pt       ← BEST LIMU-BERT (use this for accuracy)
      limu_bert_frozen.pt        ← cleaner transfer-learning baseline
    metrics/
      partial_metrics.json       ← confidence threshold + per-class F1
      frozen_metrics.json
  harnet10/
    checkpoints/
      harnet10_partial.pt        ← BEST HARNet10
      harnet10_frozen.pt
    metrics/
      partial_metrics.json
      frozen_metrics.json
```

The `.pt` files contain only the learned weights (PyTorch `state_dict`). The Python files in `src/` and `finetuning/vendored/` define the model architectures so the weights can be loaded into a runnable model.

## Results (on our 976-file, 5-person dataset)

| Model     | Mode    | Window F1 | File F1 (per recording) | LOPO Window F1 |
|---        |---      |---        |---                      |---             |
| LIMU-BERT | frozen  | 0.927     | 0.970                   | 0.576 ± 0.189  |
| LIMU-BERT | partial | 0.977     | 0.995                   | (not run)      |
| HARNet10  | frozen  | 0.934     | 0.934 *                 | 0.681 ± 0.084  |
| HARNet10  | partial | 0.995     | 0.995 *                 | (not run)      |

\* HARNet10 uses one window per file (its 10s pretraining window can't fit in our ~4s recordings), so window-level and file-level metrics are identical for it.

**LOPO** = leave-one-person-out = train on 4 people, test on the 5th. The big drop vs the in-distribution numbers is the real-world expectation if your friend tests on a brand-new player.

## Setup

1. Install Python ≥ 3.10 and create a fresh virtual env.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. (Only relevant for HARNet10) The first time you load HARNet10, PyTorch downloads the backbone architecture from `OxWearables/ssl-wearables` via `torch.hub` (~40 MB, ~10 s on a fast connection). This requires internet on first run; subsequent runs use the cached copy in `~/.cache/torch/hub/`.

## Run the demo

```bash
python inference_example.py
```

Expected output (with synthetic noise as input, so labels are random):
```
input window: shape=(400, 6), dtype=float32
LIMU-BERT (partial): label='hook'     confidence=0.521
HARNet10  (partial): label='straight' confidence=0.467
Labels are one of: 'hook', 'straight', 'upper', or 'none' (low confidence).
```

## Using your own data

Each model expects ONE punch recording as a 2D numpy array shaped `(N_samples, 6)`, `float32`, at the recorder's native 104 Hz, with channel order `[ax, ay, az, gx, gy, gz]` in `mg` (acceleration in milli-g) and `dps` (gyro in degrees-per-second). This matches the raw CSV format from the recorder. The `Predictor` handles the resampling, unit conversion, and any padding internally.

Minimal usage:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from limu_bert import Predictor

p = Predictor(
    checkpoint_path="artifacts/limu_bert/checkpoints/limu_bert_partial.pt",
    threshold_path="artifacts/limu_bert/metrics/partial_metrics.json",
)
label, conf = p.predict(your_window_400x6_float32_104hz)
```

If `conf < threshold` (the threshold is loaded from `partial_metrics.json` and is the cut-off where false-trigger rate drops below 5%), the Predictor returns `("none", conf)` — useful for suppressing low-confidence triggers in a live game loop.

The HARNet10 Predictor has the same API (just `from harnet10 import Predictor`). It accepts the same 6-channel input and internally drops the gyro channels.

## Which model should you use?

- **For accuracy on people in our training set**: HARNet10 partial (0.995) ≈ LIMU-BERT partial (0.995). Either is fine.
- **For a new player the model has never seen**: HARNet10 frozen (LOPO F1 ≈ 0.68) generalizes more reliably than LIMU-BERT frozen (LOPO F1 ≈ 0.58), and its variance across subjects is also tighter (σ=0.08 vs 0.19). Use HARNet10 if your friend will test on himself.
- **For lowest latency and smallest deployable size**: LIMU-BERT partial — the checkpoint is 230 KB vs HARNet10's 41 MB, and the model has ~80k trainable params vs HARNet10's ~9M.

## Caveats

- The recordings the models were trained on are ~3.85 s long (400 samples at 104 Hz). HARNet10 was pretrained on 10 s windows; we zero-pad each recording to 300 samples at 30 Hz to fit the pretrained input shape. This works but ~62% of HARNet10's input is zeros per inference. If recordings get longer in the future, retrain HARNet10 with proper 10 s sliding windows for better numbers.
- `chi` is the hardest held-out subject in LOPO for both models. If your friend's punching style is similar to `chi`'s, expect lower accuracy.
- The confidence threshold in `partial_metrics.json` (`confidence_threshold_5pct_ft`) was tuned to keep the false-trigger rate of *incorrect* predictions below 5%. Adjust by editing the JSON if you want a different trade-off.
