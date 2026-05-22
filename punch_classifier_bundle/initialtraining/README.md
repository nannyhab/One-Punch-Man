# initialtraining/ — held-out-test reportable models

Everything in here is from the original 80/20-with-validation training round. **None of these checkpoints are the deployment model** — they were the experiment that picked the recipe and validated it on a held-out test set. The shipped production model lives at `../artifacts/limu_bert/checkpoints/limu_bert_production.pt`.

Keep these around because:

1. **`limu_bert_partial.pt`** is the academically reportable version of the production model — same recipe, but trained on 80% of the data with the remaining 20% held out as test. Its window/file F1 numbers (0.977 / 0.995) on the 196 held-out files are the honest test metric for the production approach. The production model itself can't be evaluated honestly because it has seen the entire dataset.
2. **`limu_bert_frozen.pt`** is the cleaner transfer-learning baseline (only the classifier head trained). Useful as a reference point for "how much does unfreezing the last two transformer blocks actually help?"
3. **`harnet10_*.pt`** were the architecture comparison — a 1D CNN pretrained on UK Biobank vs the LIMU-BERT transformer. HARNet10 was competitive but rejected for deployment because (a) 41 MB vs LIMU-BERT's 230 KB, (b) needs `torch.hub` on first run, (c) forced to one-window-per-file because our recordings are shorter than its 10 s pretraining window. The numbers are in the metrics JSONs.

## Contents

```
initialtraining/
  src/
    harnet10.py                                   ← architecture code for HARNet10 checkpoints below
  limu_bert/
    checkpoints/
      limu_bert_frozen.pt    (230 KB)            ← head-only transfer, 0.970 file F1
      limu_bert_partial.pt   (230 KB)            ← head + last 2 blocks, 0.995 file F1
    metrics/
      frozen_metrics.json                         ← window/file F1, per-class, confusion, ROC AUC, threshold
      partial_metrics.json                        ← same, for partial mode (this is where the production threshold came from)
  harnet10/
    checkpoints/
      harnet10_frozen.pt     (41 MB)             ← head-only, 0.934 F1
      harnet10_partial.pt    (41 MB)             ← head + last residual stage, 0.995 F1
    metrics/
      frozen_metrics.json
      partial_metrics.json
```

## Loading any of these for evaluation

Same Predictor class, just point it at a different checkpoint. For LIMU-BERT:

```python
import sys; sys.path.insert(0, "../src")
from limu_bert import Predictor
p = Predictor(
    checkpoint_path="limu_bert/checkpoints/limu_bert_partial.pt",
    threshold_path="limu_bert/metrics/partial_metrics.json",
)
```

For HARNet10 you'd need its own Predictor class from `initialtraining/src/harnet10.py` and the OxWearables `torch.hub` backbone (first-run internet required).

## Full architecture and results context

See `../ARCHITECTURE.md` for the complete spec: dataset, preprocessing, both model architectures, all hyperparameters, augmentation pool, training protocol, results tables (including LOPO), and the rationale for picking LIMU-BERT over HARNet10 for production.
