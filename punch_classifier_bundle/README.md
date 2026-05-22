# Punch Classifier — Inference Bundle

LIMU-BERT-based 3-class punch classifier (`hook`, `straight`, `upper`) on wrist-mounted IMU data (6-axis accel+gyro at 104 Hz). One production checkpoint, ready to ship.

## TL;DR

```bash
unzip punch_classifier_bundle.zip
cd punch_classifier_bundle
pip install -r requirements.txt
python inference_example.py        # works offline, no torch.hub, no internet
```

That loads the production checkpoint and classifies a synthetic punch. For your friend's real-time game testing, this is everything he needs.

## Bundle layout

```
punch_classifier_bundle/
  README.md                              ← this file
  ARCHITECTURE.md                        ← full paper-ready architecture + results doc
  inference_example.py                   ← copy-paste demo
  requirements.txt                       ← pip install -r this
  src/
    limu_bert.py                         ← LIMU-BERT model + Predictor class
    train_production.py                  ← script that produced the production checkpoint
  finetuning/vendored/LIMU-BERT-Public/
    models.py                            ← transformer architecture (~20 KB)
    utils.py                             ← 2 helpers models.py needs (~600 B)
  artifacts/limu_bert/
    checkpoints/
      limu_bert_production.pt            ← THE deployment model (230 KB)
    metrics/
      production_info.json               ← provenance + confidence threshold + reference numbers
  initialtraining/                       ← the val-based experiment that picked the recipe
    README.md                            ← what's in here, why it exists
    src/harnet10.py                      ← HARNet10 architecture code (used by HARNet10 ckpts only)
    limu_bert/
      checkpoints/
        limu_bert_frozen.pt              ← head-only transfer baseline, 0.970 file F1
        limu_bert_partial.pt             ← held-out reportable model, 0.995 file F1
      metrics/
        frozen_metrics.json
        partial_metrics.json             ← source of the production threshold
    harnet10/
      checkpoints/
        harnet10_frozen.pt               ← CNN comparison, 0.934 F1
        harnet10_partial.pt              ← CNN comparison, 0.995 F1
      metrics/
        frozen_metrics.json
        partial_metrics.json
```

## What the production checkpoint is

`limu_bert_production.pt` is a 4-layer transformer + 2-layer MLP head, trained with the exact same recipe that produced the reportable `initialtraining/limu_bert/checkpoints/limu_bert_partial.pt` model (partial fine-tune, 22 epochs, AdamW, cosine-with-warmup, batch=64), but on **all 976 files** instead of the 80% train split. It's the deployment model — it has seen every recording in the dataset and therefore can't be honestly evaluated, but it's also the model that will perform best on new recordings from the same 5 subjects.

For an honest test number, refer to `initialtraining/limu_bert/checkpoints/limu_bert_partial.pt`: same recipe, 80/20 split, held-out test F1 = **0.995** (file-level) and **0.977** (window-level). Those are the numbers to quote in any report.

## Why LIMU-BERT, not HARNet10

We also trained an Oxford HARNet10 CNN (kept in `initialtraining/harnet10/` for the comparison story). At the file level both models tied at 0.995 F1 on the held-out test set. We picked LIMU-BERT for production because:

- **Size**: 230 KB checkpoint vs HARNet10's 41 MB — ~180× smaller, matters if the model ships inside the game binary.
- **Offline**: no `torch.hub` dependency at inference. HARNet10's backbone class definition lives in the OxWearables repo and is fetched at first load.
- **Window vs file distinction**: LIMU-BERT uses ~4 sliding windows per recording, so majority voting at the file level meaningfully improves over the window level. HARNet10 was forced to one window per file (our 3.85 s recordings are shorter than its 10 s pretraining window), so file F1 = window F1 trivially.

Full rationale in `ARCHITECTURE.md` §8.

## Results

### Reportable test metrics (from `initialtraining/limu_bert/checkpoints/limu_bert_partial.pt`)

| Granularity   | Accuracy | macro-F1 |
|---            |---:      |---:      |
| Window-level  | 97.7 %   | 0.977    |
| File-level    | **99.5 %** | **0.995** |

Computed on the 196-file held-out test set (no overlap with training files).

### What to expect on a brand-new player (leave-one-person-out, frozen mode)

| Held-out subject | window F1 |
|---               |---:       |
| adnan            | 0.612     |
| ahmed            | 0.644     |
| **chi**          | **0.351** (style outlier) |
| mercy            | 0.877     |
| novel            | 0.398     |
| **mean ± std**   | **0.576 ± 0.189** |

If your friend is one of the 5 training subjects, expect ~99 %. If he's a brand new person, **expect 58 % on average** with a wide spread depending on style similarity. The `("none", conf)` threshold mitigates bad triggers in the new-player case.

## Setup

1. Python ≥ 3.10 in a fresh virtual env.
2. `pip install -r requirements.txt`.
3. `python inference_example.py` — confirms the production checkpoint loads and predicts.

No internet required at any point for the production model. The LIMU-BERT architecture is fully bundled in `finetuning/vendored/LIMU-BERT-Public/` (2 small Python files).

## Using your own data

```python
import sys; sys.path.insert(0, "src")
from limu_bert import Predictor

p = Predictor(
    checkpoint_path="artifacts/limu_bert/checkpoints/limu_bert_production.pt",
    threshold_path="artifacts/limu_bert/metrics/production_info.json",
)

# window: 2D float32 array, shape (N_samples, 6), 104 Hz, channel order
#   [ax, ay, az, gx, gy, gz] in mg and dps (the raw recorder format).
label, conf = p.predict(window)
# returns ("hook" | "straight" | "upper", confidence)
# or ("none", confidence) if softmax-max < 0.980 (calibrated for <5% false-trigger rate)
```

The Predictor handles resampling (104 → 20 Hz), unit conversion (mg → g, dps → rad/s), windowing, and padding internally.

## What's in `initialtraining/`

The val-based experiment that produced the recipe (and the comparison HARNet10 models). Keep this folder around if you want:

- the reportable test metrics for the paper (use `limu_bert_partial.pt` + `partial_metrics.json`),
- the cleaner frozen transfer baseline (`limu_bert_frozen.pt`),
- the HARNet10 architecture comparison (`harnet10_*.pt` + `src/harnet10.py`),
- the original confidence threshold calibration data (`partial_metrics.json`'s confusion matrix and per-class breakdowns).

If you're deploying and don't need these, you can delete the whole folder — the production model in `artifacts/` is self-contained.

## See also

- `ARCHITECTURE.md` — full paper-ready architecture spec, hyperparameters, augmentation pool, training protocol, all result tables, reproducibility checklist.
- `initialtraining/README.md` — context on the held-out training round and the HARNet10 comparison.
