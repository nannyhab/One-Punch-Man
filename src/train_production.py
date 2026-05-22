"""
Production-mode training: retrain the best models on ALL data (train+test combined).

Per CLAUDE.md §12 we previously held out 196 test files to report unbiased
metrics. That gave us the academic story. For real-time deployment we want
the model to have seen as much data as possible — so this script re-runs the
proven hyperparameter recipe on the union of train + test, for the exact
epoch count that the val-based runs landed on as "best".

One production checkpoint is produced:

    artifacts/limu_bert/checkpoints/limu_bert_production.pt   (partial mode, 22 ep)

LIMU-BERT was chosen as the production architecture because (a) it has a
meaningful window-vs-file majority-vote story, (b) the 230 KB checkpoint
is ~180× smaller than HARNet10's 41 MB, and (c) it has no `torch.hub`
runtime dependency for inference. HARNet10 production training can be
enabled by editing PRODUCTION_RUNS below or invoking with --only harnet.

No validation set, no early stopping. Train for EXACTLY the configured
number of epochs, then save the final state.

Why these epoch counts: from the original val-based runs in this repo,
LIMU-BERT partial's best val macro-F1 (0.9936) was at epoch 22; HARNet10
partial hit perfect val macro-F1 (1.0000) at epoch 13. Those are the
"optimal" stopping points the val signal picked out — and we trust the
val signal to also be the right epoch budget on the slightly larger
all-data training set.

CAVEAT: there is NO held-out signal for these production checkpoints.
Do not report their accuracy as a test score — the only honest evaluation
of them is real-world deployment (or LOPO on a sixth person you haven't
collected yet). The reportable test numbers stay with the existing
`*_partial.pt` checkpoints.

Usage:
    python src/train_production.py                # both models
    python src/train_production.py --only limu    # LIMU-BERT only
    python src/train_production.py --only harnet  # HARNet10 only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_REPO_ROOT = _HERE.parent
MANIFEST_PATH = _REPO_ROOT / "artifacts" / "preprocessed" / "manifest.csv"

# Configs: (script_module_name, mode, epoch_count). LIMU-BERT only by default.
PRODUCTION_RUNS = [
    ("limu_bert", "partial", 22),
    # ("harnet10", "partial", 13),  # disabled — see module docstring
]


def train_production(module_name: str, mode: str, n_epochs: int) -> Path:
    """Train one production checkpoint on all 976 files, fixed epoch count."""
    print(f"\n{'='*70}")
    print(f"PRODUCTION TRAIN — {module_name} ({mode}, {n_epochs} epochs, all data)")
    print(f"{'='*70}", flush=True)

    mod = __import__(module_name)

    # Reset seeds so both runs are deterministic and independent.
    mod.set_seed(mod.SEED)

    manifest = mod.load_manifest(MANIFEST_PATH)
    print(f"[prod] manifest rows: {len(manifest)} "
          f"(train={(manifest['split']=='train').sum()}, "
          f"test={(manifest['split']=='test').sum()}) — using ALL")

    # Build windows from EVERY file. Ignore the split column entirely.
    X, y, _meta = mod.windows_from_manifest_subset(manifest, mod.LABEL_MAP_INT)
    print(f"[prod] training set: X={X.shape}  y={y.shape}  classes={np.bincount(y).tolist()}")

    # Build the model (loads pretrained backbone where applicable).
    model = mod.build_model(mode=mode, load_pretrained=True).to(mod.DEVICE)

    lr = mod.LR_PARTIAL if mode == "partial" else mod.LR_FROZEN
    class_weights = mod.inverse_freq_weights(y).to(mod.DEVICE)
    print(f"[prod] class weights: {class_weights.cpu().numpy()}")

    loader = DataLoader(
        mod.WindowDataset(X, y, augment=True),
        batch_size=mod.BATCH_SIZE, shuffle=True, num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.trainable_params(), lr=lr, weight_decay=mod.WEIGHT_DECAY
    )
    use_amp = mod.DEVICE.type == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    t0 = time.time()
    for epoch in range(n_epochs):
        # Cosine-with-warmup LR schedule — same as the val-based runs, so the
        # final epoch reaches the same LR as the val run did at its best epoch.
        lr_scale = mod.cosine_warmup_lr(epoch, mod.WARMUP_EPOCHS, n_epochs)
        for g in optimizer.param_groups:
            g["lr"] = lr * lr_scale

        model.train()
        epoch_loss, n_seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(mod.DEVICE), yb.to(mod.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb, weight=class_weights)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_params(), mod.GRAD_CLIP)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, weight=class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_params(), mod.GRAD_CLIP)
                optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        epoch_loss /= max(1, n_seen)
        print(f"[epoch {epoch:3d}] lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"train_loss={epoch_loss:.4f}  elapsed={time.time()-t0:.0f}s",
              flush=True)

    # Save the FINAL model state — not the best-by-val.
    artifacts_dir = _REPO_ROOT / "artifacts" / module_name
    ckpt_path = artifacts_dir / "checkpoints" / f"{module_name}_production.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "mode": mode,
        "epoch": n_epochs - 1,
        "val_macro_f1": -1.0,  # no val used — explicit sentinel
        "seq_len": mod.SEQ_LEN,
        "in_channels": mod.IN_CHANNELS,
        "trained_on": "all-data (train+test)",
        "epoch_budget_source": "val-based best epoch from the held-out run",
    }
    # HARNet10 stores feature_dim too — preserve that.
    if hasattr(model, "feature_dim"):
        payload["feature_dim"] = int(model.feature_dim)
    torch.save(payload, ckpt_path)
    print(f"[prod] saved: {ckpt_path}  ({ckpt_path.stat().st_size/1024:.0f} KB)")

    # Also write a small sidecar so it's clear what this checkpoint is.
    meta_path = artifacts_dir / "metrics" / "production_train_info.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({
        "checkpoint": str(ckpt_path.relative_to(_REPO_ROOT)),
        "mode": mode,
        "n_epochs": n_epochs,
        "training_set": "all 976 files (train+test combined)",
        "no_val_or_early_stopping": True,
        "epoch_budget_chosen_via": "val-based best epoch from the prior held-out run",
        "honest_evaluation": "NONE — this checkpoint has no held-out signal. "
                             "Use the existing *_partial.pt checkpoints for reportable test metrics.",
    }, indent=2))
    print(f"[prod] info: {meta_path}")

    return ckpt_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["limu", "harnet"], default=None,
                   help="train only one model (default: both)")
    args = p.parse_args()

    runs = list(PRODUCTION_RUNS)
    if args.only == "limu":
        runs = [r for r in runs if r[0] == "limu_bert"]
    elif args.only == "harnet":
        runs = [r for r in runs if r[0] == "harnet10"]

    for module_name, mode, n_epochs in runs:
        train_production(module_name, mode, n_epochs)

    print(f"\n{'='*70}\nALL PRODUCTION CHECKPOINTS WRITTEN\n{'='*70}")


if __name__ == "__main__":
    main()
