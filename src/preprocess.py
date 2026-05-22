"""
preprocess.py — CSV → canonical 104 Hz / 6-channel filtered arrays.

Walks data/{person}/{label}/*.csv, verifies sample rate, applies a 30 Hz
Butterworth lowpass at fs=104.0, and writes one float32 .npy per file plus a
manifest.csv. Stratified 80/20 file-level split (random_state=42) lives in
the manifest. No normalization, no windowing, no unit conversion — those are
model-specific concerns owned by the training scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split


# Canonical contract: 104 Hz, 6 channels in this order.
TARGET_FS: float = 104.0
FS_TOLERANCE_HZ: float = 0.5
LOWPASS_CUTOFF_HZ: float = 30.0
LOWPASS_ORDER: int = 4
CHANNELS: list[str] = ["ax", "ay", "az", "gx", "gy", "gz"]

# Mirrors old-notebooks/preprocessing/data_cleaning.py LABEL_MAP.
LABEL_MAP: dict[str, int] = {"hook": 0, "straight": 1, "upper": 2}

RENAME: dict[str, str] = {
    "acc_x_mg": "ax",
    "acc_y_mg": "ay",
    "acc_z_mg": "az",
    "gyr_x_dps": "gx",
    "gyr_y_dps": "gy",
    "gyr_z_dps": "gz",
}


def build_manifest(data_dir: Path) -> pd.DataFrame:
    """Walk data_dir/{person}/{label}/*.csv. Skip unknown label dirs."""
    records = []
    for csv_path in sorted(data_dir.rglob("*.csv")):
        label = csv_path.parent.name
        person = csv_path.parent.parent.name
        if label not in LABEL_MAP:
            continue
        records.append({"path": str(csv_path), "person": person, "label": label})
    return pd.DataFrame(records, columns=["path", "person", "label"])


def assign_split(manifest: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Stratified file-level split. random_state=42 is contractual."""
    out = manifest.copy()
    if len(manifest) == 0:
        out["split"] = pd.Series(dtype=str)
        return out
    # Need at least 2 files to make any split, and >=5 with all-class >=2 to stratify.
    n = len(manifest)
    n_test = int(round(n * test_size))
    if n_test < 1 or (n - n_test) < 1:
        print(f"  NOTE: only {n} file(s); assigning all to train (cannot 80/20 split).", flush=True)
        out["split"] = "train"
        return out
    label_counts = manifest["label"].value_counts()
    stratify = manifest["label"] if (label_counts.min() >= 2 and n >= 5) else None
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=stratify
    )
    out["split"] = "train"
    out.loc[out.index[test_idx], "split"] = "test"
    return out


def load_csv(path: Path) -> Tuple[pd.DataFrame, pd.Series | None]:
    """Returns (renamed dataframe with CHANNELS columns, sample_idx series or None)."""
    df = pd.read_csv(path)
    df = df.rename(columns=RENAME)
    missing = [c for c in CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing channels {missing} after rename")
    sample_idx = df["sample_idx"] if "sample_idx" in df.columns else None
    return df[CHANNELS].copy(), sample_idx


def estimate_fs(sample_idx: pd.Series | None, n_samples: int) -> float:
    """
    Estimate sample rate. The recorder writes a monotonic sample_idx counter
    (not a timestamp), so we cannot derive fs from sample_idx alone — counter
    deltas are always 1. In that case we return the nominal target (no info to
    contradict it). If sample_idx happens to carry millisecond timestamps
    (non-unit deltas), use median dt to derive Hz.
    """
    if sample_idx is None or len(sample_idx) < 2:
        return float("nan")
    diffs = np.diff(sample_idx.to_numpy(dtype=np.float64))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return float("nan")
    median_dt = float(np.median(diffs))
    if abs(median_dt - 1.0) < 1e-9:
        # Pure counter — no real time information. Report nominal.
        return TARGET_FS
    # Heuristic: if median_dt looks like milliseconds (>0.5), treat as ms;
    # if it looks like seconds (<0.1), treat as seconds.
    if median_dt >= 0.5:
        return 1000.0 / median_dt  # ms
    return 1.0 / median_dt  # s


def lowpass_filter(data: np.ndarray) -> np.ndarray:
    """Zero-phase Butterworth lowpass at 30 Hz, fs=104.0, order 4."""
    nyq = TARGET_FS / 2.0
    b, a = signal.butter(LOWPASS_ORDER, LOWPASS_CUTOFF_HZ / nyq, btype="low")
    return signal.filtfilt(b, a, data, axis=0).astype(np.float32)


def out_name(person: str, label: str, stem: str) -> str:
    """Double-underscore separators so downstream can split on '__' cleanly."""
    return f"{person}__{label}__{stem}.npy"


def process_file(
    src: Path, dst: Path, force: bool
) -> Tuple[str, float, int, str | None]:
    """
    Returns (status, actual_fs, n_samples, error). status ∈
    {"processed", "cached", "failed"}.
    """
    try:
        if (not force) and dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            arr = np.load(dst, mmap_mode="r")
            return ("cached", float("nan"), int(arr.shape[0]), None)
        df, sample_idx = load_csv(src)
        raw = df.to_numpy(dtype=np.float32)
        actual_fs = estimate_fs(sample_idx, len(raw))
        if raw.shape[0] < 4 * LOWPASS_ORDER:
            return ("failed", actual_fs, raw.shape[0], f"too short for filtfilt (n={raw.shape[0]})")
        filtered = lowpass_filter(raw)
        np.save(dst, filtered.astype(np.float32, copy=False), allow_pickle=False)
        return ("processed", actual_fs, filtered.shape[0], None)
    except Exception as e:  # surface, do not swallow
        return ("failed", float("nan"), 0, f"{type(e).__name__}: {e}")


def main() -> int:
    p = argparse.ArgumentParser(description="Preprocess punch IMU CSVs to canonical 104 Hz arrays.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("artifacts/preprocessed"))
    p.add_argument("--force", action="store_true", help="Re-process even cached files")
    args = p.parse_args()

    if not args.data_dir.exists() or not args.data_dir.is_dir():
        print(f"ERROR: --data-dir does not exist or is not a directory: {args.data_dir}", flush=True)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args.data_dir)
    if len(manifest) == 0:
        print(f"ERROR: no CSV files found under {args.data_dir}/{{person}}/{{label}}/", flush=True)
        return 2
    manifest = assign_split(manifest)

    print(f"Found {len(manifest)} files across {manifest['person'].nunique()} persons "
          f"and {manifest['label'].nunique()} labels.", flush=True)

    actual_fs_col: list[float] = []
    n_samples_col: list[int] = []
    npy_path_col: list[str] = []
    n_processed = n_cached = 0
    warnings: list[tuple[str, float]] = []
    failures: list[tuple[str, str]] = []

    for _, row in manifest.iterrows():
        src = Path(row["path"])
        dst = args.out / out_name(row["person"], row["label"], src.stem)
        status, fs, n, err = process_file(src, dst, args.force)
        actual_fs_col.append(fs)
        n_samples_col.append(n)
        npy_path_col.append(str(dst))

        if status == "processed":
            n_processed += 1
            if not np.isnan(fs) and abs(fs - TARGET_FS) > FS_TOLERANCE_HZ:
                warnings.append((str(src), fs))
                print(f"  WARN sample-rate: {src} actual_fs={fs:.3f} Hz (expected {TARGET_FS})", flush=True)
            print(f"  processed: {dst.name} (n={n}, fs~={fs:.2f})", flush=True)
        elif status == "cached":
            n_cached += 1
            print(f"  skipped (cached): {dst.name}", flush=True)
        else:
            failures.append((str(src), err or "unknown"))
            print(f"  FAILED: {src} -> {err}", flush=True)

    manifest["actual_fs"] = actual_fs_col
    manifest["n_samples"] = n_samples_col
    manifest["npy_path"] = npy_path_col
    # Always rewrite manifest, in declared column order.
    cols = ["path", "person", "label", "split", "actual_fs", "n_samples", "npy_path"]
    manifest[cols].to_csv(args.out / "manifest.csv", index=False)

    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"Total files:      {len(manifest)}", flush=True)
    print(f"Processed:        {n_processed}", flush=True)
    print(f"Cached (skipped): {n_cached}", flush=True)
    print(f"Failed:           {len(failures)}", flush=True)
    print(f"FS warnings:      {len(warnings)}", flush=True)
    for path, fs in warnings:
        print(f"  - {path}  actual_fs={fs:.3f}", flush=True)
    for path, err in failures:
        print(f"  FAIL {path}: {err}", flush=True)
    print(f"Manifest:         {args.out / 'manifest.csv'}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
