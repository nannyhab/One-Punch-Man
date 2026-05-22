"""
harnet10.py — Oxford HARNet10 punch classifier (train + finetune + eval + LOPO).

Loads preprocessed canonical 104 Hz / 6-channel arrays from
artifacts/preprocessed/, drops gyro, resamples per-window to exactly 30 Hz,
applies the notebook's augmentations minus rotation, fine-tunes the
OxWearables/ssl-wearables harnet10 backbone with a 3-class head, evaluates,
runs LOPO, and exposes a side-effect-free Predictor class.

CLI:
    python src/harnet10.py --task {train,eval,lopo,all} --mode {frozen,partial}

Parity contract with src/limu_bert.py: shared CLI shape, shared training loop,
shared eval suite, shared plot filenames. Implementation is intentionally
duplicated (not factored) per CLAUDE.md anti-goal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # save plots headless — laptop, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Config — mirror limu_bert.py except for: 3-axis accel, 30 Hz, 300-sample windows.
# ---------------------------------------------------------------------------

SOURCE_FS: int = 104
TARGET_FS: int = 30
# 104 * 15 / 52 = 30.0 — exact, no fractional drift.
RESAMPLE_UP: int = 15
RESAMPLE_DOWN: int = 52

# DEVIATION FROM CLAUDE.md SPEC: spec asks for 10-s sliding windows at 104 Hz
# (1040/260), but our recordings are only ~3.85 s (400 samples), so any source
# window ≥ file length produces zero windows. Also, HARNet10's strided conv
# stack requires input ≥ ~100 samples or its deeper kernels overrun the
# downsampled feature length. Compromise: bypass sliding windows entirely.
# One window per file = full file resampled to 30 Hz (~115 samples) then
# zero-padded to HARNet10's pretrained 300-sample shape. Fewer windows (976
# total) but preserves the pretrained input contract.
WINDOW_SIZE_SRC: int = 0  # unused — see windows_from_manifest_subset
HOP_SIZE_SRC: int = 0  # unused

SEQ_LEN: int = 300
IN_CHANNELS: int = 3  # accel only — drop gyro after load

# Unit conversion — harnet10 expects raw accel in g, so mg ÷ 1000.
ACCEL_MG_TO_G: float = 1.0 / 1000.0
APPLY_ZSCORE_AFTER_UNIT_CONVERT: bool = False

# Head — feature dim discovered from torch.hub model at runtime.
HEAD_HIDDEN: int = 128
HEAD_DROPOUT: float = 0.3
NUM_CLASSES: int = 3
LABEL_NAMES: list[str] = ["hook", "straight", "upper"]

# Backbone feature dim — fallback used if torch.hub is unreachable.
HARNET10_FEATURE_DIM_FALLBACK: int = 1024

# Training — same shape as LIMU-BERT.
SEED: int = 42
BATCH_SIZE: int = 64
MAX_EPOCHS: int = 100
EARLY_STOP_PATIENCE: int = 10
WEIGHT_DECAY: float = 0.01
GRAD_CLIP: float = 1.0
WARMUP_EPOCHS: int = 5
VAL_FILE_FRACTION: float = 0.10
LR_FROZEN: float = 3e-4
LR_PARTIAL: float = 1e-5  # spec: 1e-5 for partial unfreeze of last residual block

PEOPLE: list[str] = ["adnan", "ahmed", "chi", "mercy", "novel"]

HARNET10_HUB: tuple[str, str] = ("OxWearables/ssl-wearables", "harnet10")

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """Deterministic across numpy/torch/cuda/python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data: load preprocessed .npy → drop gyro → window at 104 Hz → resample.
# ---------------------------------------------------------------------------


def _convert_units_accel(x3: np.ndarray) -> np.ndarray:
    """mg → g for 3-axis accel."""
    out = x3.astype(np.float32, copy=True)
    out *= ACCEL_MG_TO_G
    return out


def _resample_to_target(x_src: np.ndarray) -> np.ndarray:
    """104 → 30 Hz via polyphase. Exact integer ratio."""
    return resample_poly(x_src, up=RESAMPLE_UP, down=RESAMPLE_DOWN, axis=0).astype(
        np.float32
    )


def segment(x: np.ndarray, window_size: int, hop_size: int) -> np.ndarray:
    """Sliding window. Drops trailing remainder."""
    n = x.shape[0]
    if n < window_size:
        return np.zeros((0, window_size, x.shape[1]), dtype=x.dtype)
    n_windows = 1 + (n - window_size) // hop_size
    out = np.empty((n_windows, window_size, x.shape[1]), dtype=x.dtype)
    for i in range(n_windows):
        s = i * hop_size
        out[i] = x[s : s + window_size]
    return out


def _process_window(w_src_104hz_3ch: np.ndarray) -> np.ndarray:
    """One source window (1040,3) → model input (300,3)."""
    rs = _resample_to_target(w_src_104hz_3ch)
    rs = _convert_units_accel(rs)
    if rs.shape[0] < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - rs.shape[0], IN_CHANNELS), dtype=np.float32)
        rs = np.concatenate([rs, pad], axis=0)
    return rs[:SEQ_LEN].astype(np.float32, copy=False)


def windows_from_manifest_subset(
    manifest_subset: pd.DataFrame, label_map: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load .npy, drop gyro, one-window-per-file: full file → resample → pad to SEQ_LEN."""
    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for _, row in manifest_subset.iterrows():
        arr = np.load(row["npy_path"]).astype(np.float32, copy=False)
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(
                f"expected (N,6) at 104 Hz; got {arr.shape} for {row['npy_path']}"
            )
        arr3 = arr[:, :3]  # drop gyro
        processed = _process_window(arr3)[None, ...]  # (1, SEQ_LEN, 3)
        X_chunks.append(processed)
        label_int = label_map[row["label"]]
        y_chunks.append(np.full(1, label_int, dtype=np.int64))
        meta_rows.append(
            {
                "path": row["path"],
                "person": row["person"],
                "label": row["label"],
                "window_idx": 0,
            }
        )
    if not X_chunks:
        empty = np.zeros((0, SEQ_LEN, IN_CHANNELS), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.int64), pd.DataFrame(meta_rows)
    return (
        np.concatenate(X_chunks, axis=0),
        np.concatenate(y_chunks, axis=0),
        pd.DataFrame(meta_rows),
    )


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    m = pd.read_csv(manifest_path)
    required = {"path", "person", "label", "split", "npy_path"}
    missing = required - set(m.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    return m


def split_train_val_by_file(
    meta_train: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified file-level val carve-out from train."""
    file_df = meta_train.groupby("path").agg(label=("label", "first")).reset_index()
    n_files = len(file_df)
    n_classes_present = file_df["label"].nunique()
    n_val_files = max(1, int(round(n_files * val_fraction)))
    can_stratify = (
        n_files >= 2
        and file_df["label"].value_counts().min() >= 2
        and n_val_files >= n_classes_present
        and (n_files - n_val_files) >= 1
    )
    if can_stratify:
        train_files, val_files = train_test_split(
            file_df["path"].to_numpy(),
            test_size=val_fraction,
            stratify=file_df["label"].to_numpy(),
            random_state=seed,
        )
        val_mask = meta_train["path"].isin(set(val_files.tolist())).to_numpy()
        return ~val_mask, val_mask
    if n_files >= 2 and (n_files - n_val_files) >= 1:
        rng = np.random.RandomState(seed)
        chosen_idx = rng.choice(n_files, size=n_val_files, replace=False)
        val_files = set(file_df["path"].iloc[chosen_idx].tolist())
        val_mask = meta_train["path"].isin(val_files).to_numpy()
        return ~val_mask, val_mask
    n = len(meta_train)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction)))
    val_mask = np.zeros(n, dtype=bool)
    val_mask[idx[:n_val]] = True
    return ~val_mask, val_mask


def _fit_scaler_on(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X.reshape(-1, X.shape[-1]))
    return scaler


def _apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    flat = scaler.transform(X.reshape(-1, X.shape[-1]))
    return flat.reshape(X.shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation set — same as LIMU-BERT MINUS rotation (3-axis-only pretraining
# did not include rotation augmentation).
# ---------------------------------------------------------------------------

from tsaug import Drift, TimeWarp  # noqa: E402

_drifter = Drift(max_drift=0.2, n_drift_points=5)
_warper = TimeWarp(n_speed_change=3, max_speed_ratio=1.5)


def _jitter(w: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    return w + np.random.normal(0.0, sigma, size=w.shape).astype(w.dtype)


def _scaling(w: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    factors = np.random.normal(1.0, sigma, size=(1, w.shape[1])).astype(w.dtype)
    return w * factors


def _drift(w: np.ndarray) -> np.ndarray:
    return _drifter.augment(w[np.newaxis, ...])[0].astype(w.dtype)


def _time_warp(w: np.ndarray) -> np.ndarray:
    return _warper.augment(w[np.newaxis, ...])[0].astype(w.dtype)


# NB: no rotation here — see module docstring.
_AUGS = [
    ("jitter", _jitter),
    ("scaling", _scaling),
    ("drift", _drift),
    ("time_warp", _time_warp),
]


def augment_window(window: np.ndarray, p_each: float = 0.5, max_n: int = 2) -> np.ndarray:
    """Random subset (≤ max_n). Always apply at least one."""
    indices = np.random.permutation(len(_AUGS))[:max_n]
    out, applied = window, 0
    for idx in indices:
        if np.random.random() < p_each:
            out = _AUGS[idx][1](out)
            applied += 1
    if applied == 0:
        idx = int(np.random.choice(indices))
        out = _AUGS[idx][1](out)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Model: HARNet10 backbone + linear head.
# ---------------------------------------------------------------------------


def _load_harnet10_backbone() -> tuple[Optional[nn.Module], int, bool]:
    """Returns (backbone_module_or_None, feature_dim, loaded_ok).

    Harnet10 expects input shape (B, 3, T). We probe the feature dim by running
    a single forward through the backbone with the classification head removed.
    """
    try:
        full = torch.hub.load(*HARNET10_HUB, pretrained=True, trust_repo=True,
                              source="github")
    except Exception as e:
        print(f"[model] WARN: torch.hub load failed ({e}); "
              f"defaulting feature dim to {HARNET10_FEATURE_DIM_FALLBACK} and "
              "using random init.")
        return None, HARNET10_FEATURE_DIM_FALLBACK, False

    # Locate the classification head; OxWearables/ssl-wearables harnet10 uses
    # `classifier` for the linear projection — strip it so we get pooled features.
    backbone = full
    feature_dim: Optional[int] = None
    if hasattr(full, "classifier") and isinstance(full.classifier, nn.Module):
        head = full.classifier
        # discover in_features from the last Linear inside the head
        for m in head.modules():
            if isinstance(m, nn.Linear):
                feature_dim = m.in_features
        backbone.classifier = nn.Identity()
    elif hasattr(full, "feature_extractor"):
        backbone = full.feature_extractor

    # Probe to confirm feature dim with a dummy (B=2, C=3, T=SEQ_LEN).
    with torch.no_grad():
        backbone.eval()
        dummy = torch.zeros(2, IN_CHANNELS, SEQ_LEN)
        try:
            out = backbone(dummy)
        except Exception:
            # Some checkpoints expect (B, T, C); try that as a fallback.
            out = backbone(dummy.permute(0, 2, 1))
    feat = out
    if feat.ndim == 3:
        feat = feat.mean(dim=-1)  # GAP over time
    if feat.ndim != 2:
        feat = feat.reshape(feat.size(0), -1)
    discovered = int(feat.shape[-1])
    if feature_dim is None or feature_dim != discovered:
        feature_dim = discovered
    print(f"[model] harnet10 backbone loaded; feature_dim = {feature_dim}")
    return backbone, feature_dim, True


class HarnetClassifier(nn.Module):
    """HARNet10 backbone + LayerNorm + small MLP head → 3 classes.

    Mirrors LimuBertClassifier's head shape (LayerNorm → Linear → GELU →
    Dropout → Linear) so the comparison surface is the same.
    """

    def __init__(self, mode: str = "frozen", backbone: Optional[nn.Module] = None,
                 feature_dim: Optional[int] = None):
        super().__init__()
        self.mode = mode
        if backbone is None:
            backbone, feature_dim, _ = _load_harnet10_backbone()
        if feature_dim is None:
            feature_dim = HARNET10_FEATURE_DIM_FALLBACK
        self.feature_dim = feature_dim
        self.backbone = backbone if backbone is not None else _RandomBackbone(feature_dim)
        self.head_norm = nn.LayerNorm(feature_dim)
        self.head_fc1 = nn.Linear(feature_dim, HEAD_HIDDEN)
        self.head_dropout = nn.Dropout(HEAD_DROPOUT)
        self.head_fc2 = nn.Linear(HEAD_HIDDEN, NUM_CLASSES)
        self._apply_freeze()

    def _last_residual_params(self) -> list[nn.Parameter]:
        """Best-effort: unfreeze parameters whose name suggests the last block."""
        named = list(self.backbone.named_parameters())
        if not named:
            return []
        # Group by top-level module attribute; keep params from the last group.
        last_prefix = named[-1][0].split(".")[0]
        return [p for n, p in named if n.split(".")[0] == last_prefix]

    def _apply_freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.mode == "partial":
            for p in self._last_residual_params():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, T, C) from WindowDataset → backbone wants (B, C, T).
        if x.ndim == 3 and x.shape[-1] == IN_CHANNELS:
            x = x.permute(0, 2, 1).contiguous()
        h = self.backbone(x)
        if h.ndim == 3:
            h = h.mean(dim=-1)  # GAP over time
        if h.ndim != 2:
            h = h.reshape(h.size(0), -1)
        h = self.head_norm(h)
        h = self.head_fc1(h)
        h = F.gelu(h)
        h = self.head_dropout(h)
        return self.head_fc2(h)

    def trainable_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


class _RandomBackbone(nn.Module):
    """Fallback used only when torch.hub is unreachable. Small Conv1d stack
    so the rest of the pipeline (training loop, eval, plots) still works."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.net(x).squeeze(-1)


def build_model(
    mode: str = "frozen",
    load_pretrained: bool = True,
) -> HarnetClassifier:
    if load_pretrained:
        backbone, feature_dim, _ = _load_harnet10_backbone()
        return HarnetClassifier(mode=mode, backbone=backbone, feature_dim=feature_dim)
    # eval-time rebuild: random backbone is fine — we'll load_state_dict next.
    return HarnetClassifier(mode=mode,
                            backbone=_RandomBackbone(HARNET10_FEATURE_DIM_FALLBACK),
                            feature_dim=HARNET10_FEATURE_DIM_FALLBACK)


# ---------------------------------------------------------------------------
# Training loop — copy-paste from limu_bert.py (per CLAUDE.md anti-goal).
# ---------------------------------------------------------------------------


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.augment:
            x = augment_window(x)
        return torch.from_numpy(x.astype(np.float32, copy=False)), int(self.y[idx])


def inverse_freq_weights(y: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    inv = 1.0 / counts
    inv = inv * (num_classes / inv.sum())
    return torch.tensor(inv, dtype=torch.float32)


def cosine_warmup_lr(epoch: int, warmup_epochs: int, max_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class TrainResult:
    best_val_f1: float
    history: list[dict]
    checkpoint_path: str


def train_one_run(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mode: str,
    max_epochs: int,
    batch_size: int,
    lr: float,
    save_to: Path,
    patience: int = EARLY_STOP_PATIENCE,
    use_augment: bool = True,
    verbose: bool = True,
) -> TrainResult:
    print(f"[train] device={DEVICE}, mode={mode}, lr={lr}, max_epochs={max_epochs}")
    model = build_model(mode=mode, load_pretrained=True).to(DEVICE)
    train_loader = DataLoader(
        WindowDataset(X_train, y_train, augment=use_augment),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        WindowDataset(X_val, y_val, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    class_weights = inverse_freq_weights(y_train).to(DEVICE)
    print(f"[train] class weights: {class_weights.cpu().numpy()}")
    optimizer = torch.optim.AdamW(model.trainable_params(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)
    use_amp = DEVICE.type == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    best_f1, best_epoch, stale = -1.0, -1, 0
    history: list[dict] = []
    for epoch in range(max_epochs):
        lr_scale = cosine_warmup_lr(epoch, WARMUP_EPOCHS, max_epochs)
        for g in optimizer.param_groups:
            g["lr"] = lr * lr_scale

        model.train()
        train_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb, weight=class_weights)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_params(), GRAD_CLIP)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, weight=class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_params(), GRAD_CLIP)
                optimizer.step()
            train_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss /= max(1, n)

        model.eval()
        val_loss = 0.0
        all_p, all_l = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_loss += F.cross_entropy(logits, yb, weight=class_weights).item() * xb.size(0)
                all_p.append(logits.argmax(-1).cpu().numpy())
                all_l.append(yb.cpu().numpy())
        val_loss /= max(1, len(val_loader.dataset))
        y_pred = np.concatenate(all_p) if all_p else np.zeros(0, dtype=np.int64)
        y_true = np.concatenate(all_l) if all_l else np.zeros(0, dtype=np.int64)
        val_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        history.append(
            dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"],
                 train_loss=train_loss, val_loss=val_loss, val_macro_f1=val_f1)
        )
        if verbose:
            print(f"[epoch {epoch:3d}] lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  valF1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            save_to.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": model.state_dict(), "mode": mode, "epoch": epoch,
                 "val_macro_f1": val_f1, "seq_len": SEQ_LEN, "in_channels": IN_CHANNELS,
                 "feature_dim": model.feature_dim},
                save_to,
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"[train] early stop at {epoch} (best={best_epoch}, f1={best_f1:.4f})")
                break

    if not save_to.exists():
        save_to.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": model.state_dict(), "mode": mode, "epoch": -1,
             "val_macro_f1": best_f1, "seq_len": SEQ_LEN, "in_channels": IN_CHANNELS,
             "feature_dim": model.feature_dim},
            save_to,
        )

    return TrainResult(best_val_f1=best_f1, history=history, checkpoint_path=str(save_to))


# ---------------------------------------------------------------------------
# Eval suite — copy-paste from limu_bert.py.
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_probs(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval().to(DEVICE)
    out = []
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[s : s + batch_size].astype(np.float32)).to(DEVICE)
        out.append(F.softmax(model(xb), dim=-1).cpu().numpy())
    return (
        np.concatenate(out, axis=0)
        if out
        else np.zeros((0, NUM_CLASSES), dtype=np.float32)
    )


def save_confusion(cm: np.ndarray, labels: list[str], out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def save_roc(probs: np.ndarray, y_true: np.ndarray, labels: list[str], out_png: Path) -> dict:
    y_bin = label_binarize(y_true, classes=list(range(len(labels))))
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1, 1)
    aucs: dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, name in enumerate(labels):
        if i >= y_bin.shape[1] or y_bin[:, i].sum() == 0:
            aucs[name] = float("nan")
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        auc = roc_auc_score(y_bin[:, i], probs[:, i])
        aucs[name] = float(auc)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("one-vs-rest ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return aucs


def file_level_majority_vote(
    probs: np.ndarray, y_true: np.ndarray, meta: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    preds = probs.argmax(axis=-1)
    df = meta.copy()
    df["pred"] = preds
    df["true"] = y_true
    fp, ft = [], []
    for _, g in df.groupby("path"):
        fp.append(Counter(g["pred"].tolist()).most_common(1)[0][0])
        ft.append(g["true"].iloc[0])
    return np.array(ft), np.array(fp)


def confidence_threshold(
    probs: np.ndarray, y_true: np.ndarray, target_fp_rate: float = 0.05
) -> float:
    preds = probs.argmax(axis=-1)
    confs = probs.max(axis=-1)
    incorrect = preds != y_true
    if incorrect.sum() == 0:
        return 0.5
    inc_confs = confs[incorrect]
    grid = np.linspace(1.0 / NUM_CLASSES, 1.0, 100)
    best_t, best_rate = float(grid[-1]), 1.0
    for t in grid:
        rate = float((inc_confs >= t).mean())
        if rate < target_fp_rate:
            return float(t)
        if rate < best_rate:
            best_rate, best_t = rate, float(t)
    return best_t


def calibration_plot(
    probs: np.ndarray, y_true: np.ndarray, threshold: float, out_png: Path
) -> None:
    preds = probs.argmax(axis=-1)
    confs = probs.max(axis=-1)
    correct = preds == y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(confs[correct], bins=30, alpha=0.6, label="correct", color="tab:green")
    ax.hist(confs[~correct], bins=30, alpha=0.6, label="incorrect", color="tab:red")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.3f}")
    ax.set_xlabel("softmax max")
    ax.set_ylabel("count")
    ax.set_title("confidence histogram")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_training_curves(history: list[dict], out_path: Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axs[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("loss")
    axs[0].legend()
    axs[0].set_title("loss")
    axs[1].plot(epochs, [h["val_macro_f1"] for h in history])
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("val macro-F1")
    axs[1].set_title("val macro-F1")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _model_from_ckpt(ckpt_path: Path, mode: str) -> HarnetClassifier:
    """Rebuild model with the same backbone arch used at train time, load weights.

    Re-loads the real harnet10 backbone via torch.hub so saved keys match. Falls
    back to a small random Conv1d stack only if hub is unreachable — in which
    case strict=False will load whatever overlaps (head + matching shapes).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feature_dim = int(ckpt.get("feature_dim", HARNET10_FEATURE_DIM_FALLBACK))
    backbone, discovered_dim, hub_ok = _load_harnet10_backbone()
    if hub_ok and discovered_dim == feature_dim:
        model = HarnetClassifier(mode=mode, backbone=backbone, feature_dim=feature_dim)
        model.load_state_dict(ckpt["model_state"])
        return model
    # Hub unreachable (or shape mismatch) — use fallback, strict=False.
    print(f"[eval] WARN: rebuilding with random backbone (hub_ok={hub_ok}, "
          f"saved_dim={feature_dim}, discovered_dim={discovered_dim})")
    model = HarnetClassifier(mode=mode,
                             backbone=_RandomBackbone(feature_dim),
                             feature_dim=feature_dim)
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model


def evaluate_mode(
    mode: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    artifacts_dir: Path,
) -> tuple[dict, np.ndarray]:
    ckpt_dir = artifacts_dir / "checkpoints"
    plots_dir = artifacts_dir / "plots"
    metrics_dir = artifacts_dir / "metrics"

    ckpt_path = ckpt_dir / f"harnet10_{mode}.pt"
    model = _model_from_ckpt(ckpt_path, mode)
    probs = predict_probs(model, X_test)

    if len(probs) == 0:
        print("[eval] no test windows; skipping metrics.")
        return {"mode": mode, "note": "no test windows"}, probs

    preds = probs.argmax(axis=-1)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    win = {
        "accuracy": float((preds == y_test).mean()),
        "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
        "per_class": {
            LABEL_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
            for i in range(NUM_CLASSES)
        },
    }
    cm_w = confusion_matrix(y_test, preds, labels=list(range(NUM_CLASSES)))
    save_confusion(cm_w, LABEL_NAMES, plots_dir / f"confusion_window_{mode}.png",
                   f"window-level ({mode})")
    win["confusion_matrix"] = cm_w.tolist()
    win["roc_auc"] = save_roc(probs, y_test, LABEL_NAMES, plots_dir / f"roc_{mode}.png")

    y_ft, y_fp = file_level_majority_vote(probs, y_test, meta_test)
    file_metrics = {
        "accuracy": float((y_fp == y_ft).mean()) if len(y_ft) else 0.0,
        "macro_f1": float(f1_score(y_ft, y_fp, average="macro", zero_division=0)) if len(y_ft) else 0.0,
        "weighted_f1": float(f1_score(y_ft, y_fp, average="weighted", zero_division=0)) if len(y_ft) else 0.0,
        "confusion_matrix": confusion_matrix(y_ft, y_fp, labels=list(range(NUM_CLASSES))).tolist()
        if len(y_ft) else [],
    }
    if len(y_ft):
        cm_f = confusion_matrix(y_ft, y_fp, labels=list(range(NUM_CLASSES)))
        save_confusion(cm_f, LABEL_NAMES, plots_dir / f"confusion_file_{mode}.png",
                       f"file-level ({mode})")

    t = confidence_threshold(probs, y_test, target_fp_rate=0.05)
    calibration_plot(probs, y_test, t, plots_dir / f"confidence_hist_{mode}.png")

    metrics = {
        "mode": mode,
        "window_level": win,
        "file_level": file_metrics,
        "confidence_threshold_5pct_ft": float(t),
        "n_test_windows": int(len(y_test)),
        "n_test_files": int(meta_test["path"].nunique()) if len(meta_test) else 0,
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"{mode}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] {mode}  window F1={win['macro_f1']:.4f}  "
          f"file F1={file_metrics['macro_f1']:.4f}  threshold={t:.3f}")
    return metrics, probs


# ---------------------------------------------------------------------------
# LOPO loop.
# ---------------------------------------------------------------------------


def run_lopo(
    manifest_all: pd.DataFrame,
    artifacts_dir: Path,
    max_epochs: int,
    seed: int,
) -> None:
    ckpt_dir = artifacts_dir / "checkpoints"
    plots_dir = artifacts_dir / "plots"
    metrics_dir = artifacts_dir / "metrics"
    rows = []
    for person in PEOPLE:
        print(f"\n========== LOPO held-out = {person} ==========")
        train_m = manifest_all[manifest_all["person"] != person].reset_index(drop=True)
        test_m = manifest_all[manifest_all["person"] == person].reset_index(drop=True)
        if len(test_m) == 0:
            print(f"[lopo] no files for {person}, skipping")
            continue
        if len(train_m) == 0:
            print(f"[lopo] no train files left after holding out {person}, skipping")
            continue

        X_all, y_all, m_all = windows_from_manifest_subset(train_m, LABEL_MAP_INT)
        X_tst, y_tst, m_tst = windows_from_manifest_subset(test_m, LABEL_MAP_INT)
        if len(y_all) == 0 or len(y_tst) == 0:
            print(f"[lopo] empty windows for {person}, skipping")
            continue
        tm, vm = split_train_val_by_file(m_all, VAL_FILE_FRACTION, seed)
        Xtr, ytr = X_all[tm], y_all[tm]
        Xvl, yvl = X_all[vm], y_all[vm]
        print(f"[lopo] train={len(ytr)}  val={len(yvl)}  test={len(y_tst)}  "
              f"test files={m_tst['path'].nunique()}")

        ckpt_path = ckpt_dir / f"lopo_frozen_{person}.pt"
        res = train_one_run(
            Xtr, ytr, Xvl, yvl, mode="frozen",
            max_epochs=max_epochs, batch_size=BATCH_SIZE, lr=LR_FROZEN,
            save_to=ckpt_path, verbose=False,
        )

        model = _model_from_ckpt(ckpt_path, mode="frozen")
        probs = predict_probs(model, X_tst)
        if len(probs) == 0:
            print(f"[lopo] no test windows for {person}")
            continue
        preds = probs.argmax(axis=-1)
        win_f1 = float(f1_score(y_tst, preds, average="macro", zero_division=0))
        y_ft, y_fp = file_level_majority_vote(probs, y_tst, m_tst)
        file_f1 = float(f1_score(y_ft, y_fp, average="macro", zero_division=0)) if len(y_ft) else 0.0

        cm = confusion_matrix(y_tst, preds, labels=list(range(NUM_CLASSES)))
        save_confusion(cm, LABEL_NAMES, plots_dir / "lopo" / f"{person}_confusion.png",
                       f"LOPO held-out={person}")
        rows.append({
            "person": person,
            "n_test_windows": int(len(y_tst)),
            "n_test_files": int(m_tst["path"].nunique()),
            "val_macro_f1": float(res.best_val_f1),
            "test_window_macro_f1": win_f1,
            "test_file_macro_f1": file_f1,
        })
        print(f"[lopo] {person}: win-F1={win_f1:.4f}  file-F1={file_f1:.4f}  "
              f"val-F1={res.best_val_f1:.4f}")

    df_lopo = pd.DataFrame(rows)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df_lopo.to_csv(metrics_dir / "lopo_results.csv", index=False)
    summary = {
        "window_macro_f1_mean": float(df_lopo["test_window_macro_f1"].mean()) if len(df_lopo) else None,
        "window_macro_f1_std": float(df_lopo["test_window_macro_f1"].std(ddof=0)) if len(df_lopo) else None,
        "file_macro_f1_mean": float(df_lopo["test_file_macro_f1"].mean()) if len(df_lopo) else None,
        "file_macro_f1_std": float(df_lopo["test_file_macro_f1"].std(ddof=0)) if len(df_lopo) else None,
        "per_person": rows,
    }
    with open(metrics_dir / "lopo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if len(df_lopo):
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(df_lopo))
        ax.bar(x - 0.2, df_lopo["test_window_macro_f1"], width=0.4, label="window F1")
        ax.bar(x + 0.2, df_lopo["test_file_macro_f1"], width=0.4, label="file F1")
        ax.set_xticks(x)
        ax.set_xticklabels(df_lopo["person"], rotation=30)
        ax.set_ylim(0, 1)
        ax.set_ylabel("macro-F1")
        ax.set_title("LOPO summary (held-out person)")
        ax.legend()
        fig.tight_layout()
        (plots_dir / "lopo_summary.png").parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plots_dir / "lopo_summary.png", dpi=120)
        plt.close(fig)
        print(f"[lopo] window macroF1 mean ± std: "
              f"{summary['window_macro_f1_mean']:.4f} ± {summary['window_macro_f1_std']:.4f}")
        print(f"[lopo] file   macroF1 mean ± std: "
              f"{summary['file_macro_f1_mean']:.4f} ± {summary['file_macro_f1_std']:.4f}")


# ---------------------------------------------------------------------------
# Predictor — side-effect free, no auto-load.
# ---------------------------------------------------------------------------


class Predictor:
    """Side-effect-free predictor. Construct → call .predict(window). No writes."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        scaler_path: Optional[str | Path] = None,
        threshold_path: Optional[str | Path] = None,
    ):
        self.device = torch.device("cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        feature_dim = int(ckpt.get("feature_dim", HARNET10_FEATURE_DIM_FALLBACK))
        mode = ckpt.get("mode", "frozen")
        # Try to rebuild with the real backbone for an exact strict load.
        backbone, discovered_dim, hub_ok = _load_harnet10_backbone()
        if hub_ok and discovered_dim == feature_dim:
            self.model = HarnetClassifier(mode=mode, backbone=backbone,
                                          feature_dim=feature_dim)
            self.model.load_state_dict(ckpt["model_state"])
        else:
            self.model = HarnetClassifier(mode=mode,
                                          backbone=_RandomBackbone(feature_dim),
                                          feature_dim=feature_dim)
            self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.model.eval().to(self.device)
        self.scaler: Optional[StandardScaler] = None
        if scaler_path is not None and Path(scaler_path).is_file():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        self.threshold: float = 0.0
        if threshold_path is not None and Path(threshold_path).is_file():
            with open(threshold_path) as f:
                blob = json.load(f)
            self.threshold = float(blob.get("confidence_threshold_5pct_ft", 0.0))
        self.label_names: tuple[str, ...] = tuple(LABEL_NAMES)

    def _prepare(self, window: np.ndarray) -> torch.Tensor:
        """Accept source-rate (104 Hz, 3 or 6 ch) or model-rate (300, 3)."""
        arr = np.asarray(window, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"expected 2D window; got {arr.shape}")
        # Drop gyro if user passed all 6 channels.
        if arr.shape[1] == 6:
            arr = arr[:, :3]
        if arr.shape[1] != IN_CHANNELS:
            raise ValueError(f"expected (T,{IN_CHANNELS}) or (T,6); got {arr.shape}")
        if arr.shape[0] == SEQ_LEN:
            x = arr
        else:
            x = _resample_to_target(arr)
            x = _convert_units_accel(x)
            if x.shape[0] < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - x.shape[0], IN_CHANNELS), dtype=np.float32)
                x = np.concatenate([x, pad], axis=0)
            x = x[:SEQ_LEN]
        if self.scaler is not None:
            flat = self.scaler.transform(x.reshape(-1, x.shape[-1]))
            x = flat.reshape(x.shape).astype(np.float32)
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> tuple[str, float]:
        x = self._prepare(window)
        probs = F.softmax(self.model(x), dim=-1)[0].cpu().numpy()
        idx = int(probs.argmax())
        conf = float(probs[idx])
        if conf < self.threshold:
            return ("none", conf)
        return (self.label_names[idx], conf)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

LABEL_MAP_INT: dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}


def write_config(artifacts_dir: Path, mode: str, seed: int) -> None:
    cfg = {
        "model": "harnet10",
        "lr_frozen": LR_FROZEN,
        "lr_partial": LR_PARTIAL,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "grad_clip": GRAD_CLIP,
        "seq_len": SEQ_LEN,
        "in_channels": IN_CHANNELS,
        "source_fs": SOURCE_FS,
        "target_fs": TARGET_FS,
        "resample_up": RESAMPLE_UP,
        "resample_down": RESAMPLE_DOWN,
        "window_size_src": WINDOW_SIZE_SRC,
        "hop_size_src": HOP_SIZE_SRC,
        "apply_zscore_after_unit_convert": APPLY_ZSCORE_AFTER_UNIT_CONVERT,
        "harnet10_hub": list(HARNET10_HUB),
        "label_names": LABEL_NAMES,
        "seed": seed,
        "mode_last_trained": mode,
    }
    (artifacts_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / "checkpoints" / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def write_scaler(artifacts_dir: Path, scaler: Optional[StandardScaler]) -> None:
    """Always write — QA agent expects the file to exist even if scaler is None."""
    ckpt_dir = artifacts_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HARNet10 punch classifier.")
    p.add_argument("--task", choices=["train", "eval", "lopo", "all"], default="all")
    p.add_argument("--mode", choices=["frozen", "partial"], default="frozen")
    p.add_argument("--epochs", type=int, default=None,
                   help="override max epochs (default: MAX_EPOCHS=100)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--manifest", type=Path,
                   default=_REPO_ROOT / "artifacts" / "preprocessed" / "manifest.csv")
    p.add_argument("--artifacts-dir", type=Path,
                   default=_REPO_ROOT / "artifacts" / "harnet10")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    artifacts_dir = args.artifacts_dir
    (artifacts_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "plots" / "lopo").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    print(f"[main] manifest: {len(manifest)} files; "
          f"train={(manifest['split']=='train').sum()} test={(manifest['split']=='test').sum()}")

    max_epochs = args.epochs if args.epochs is not None else MAX_EPOCHS
    lr = LR_FROZEN if args.mode == "frozen" else LR_PARTIAL

    if args.task in ("train", "all"):
        train_m = manifest[manifest["split"] == "train"].reset_index(drop=True)
        test_m = manifest[manifest["split"] == "test"].reset_index(drop=True)
        X_all, y_all, meta_all = windows_from_manifest_subset(train_m, LABEL_MAP_INT)
        X_test, y_test, meta_test = windows_from_manifest_subset(test_m, LABEL_MAP_INT)
        print(f"[main] X_all_train={X_all.shape}  X_test={X_test.shape}")

        scaler: Optional[StandardScaler] = None
        if APPLY_ZSCORE_AFTER_UNIT_CONVERT and len(X_all):
            scaler = _fit_scaler_on(X_all)
            X_all = _apply_scaler(X_all, scaler)
            X_test = _apply_scaler(X_test, scaler) if len(X_test) else X_test
        write_scaler(artifacts_dir, scaler)

        tm, vm = split_train_val_by_file(meta_all, VAL_FILE_FRACTION, args.seed)
        X_tr, y_tr = X_all[tm], y_all[tm]
        X_vl, y_vl = X_all[vm], y_all[vm]
        print(f"[main] train windows={len(y_tr)}  val windows={len(y_vl)}  "
              f"test windows={len(y_test)}")

        ckpt_path = artifacts_dir / "checkpoints" / f"harnet10_{args.mode}.pt"
        res = train_one_run(
            X_tr, y_tr, X_vl, y_vl, mode=args.mode,
            max_epochs=max_epochs, batch_size=BATCH_SIZE, lr=lr,
            save_to=ckpt_path,
        )
        plot_training_curves(res.history,
                             artifacts_dir / "plots" / f"training_curves_{args.mode}.png")
        write_config(artifacts_dir, args.mode, args.seed)
        print(f"[main] best val macro-F1: {res.best_val_f1:.4f}  ckpt: {ckpt_path}")

        if args.task == "all" and len(X_test):
            evaluate_mode(args.mode, X_test, y_test, meta_test, artifacts_dir)

    if args.task == "eval":
        test_m = manifest[manifest["split"] == "test"].reset_index(drop=True)
        X_test, y_test, meta_test = windows_from_manifest_subset(test_m, LABEL_MAP_INT)
        evaluate_mode(args.mode, X_test, y_test, meta_test, artifacts_dir)

    if args.task in ("lopo", "all"):
        run_lopo(manifest, artifacts_dir, max_epochs=max_epochs, seed=args.seed)


if __name__ == "__main__":
    main()
