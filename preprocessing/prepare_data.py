"""
prepare_data.py — run the full preprocessing + feature extraction pipeline.

Run directly:
    python preprocessing/prepare_data.py

Returns X_train, y_train, X_test, y_test as numpy arrays saved to:
    preprocessing/output/X_train.npy
    preprocessing/output/y_train.npy
    preprocessing/output/X_test.npy
    preprocessing/output/y_test.npy
    preprocessing/output/meta_train.csv
    preprocessing/output/meta_test.csv

Your teammates can load these with:
    import numpy as np
    X_train = np.load("preprocessing/output/X_train.npy")
    y_train = np.load("preprocessing/output/y_train.npy")
"""

import numpy as np
from pathlib import Path

from pipeline import build_manifest, train_test_split_by_file, fit_scaler, build_dataset, LABEL_MAP
from feature_extraction import extract_features

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent.parent          # FinalProject396-496/
DATA_DIR   = ROOT / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def run():
    # ── 1. manifest ───────────────────────────────────────────────────────────
    print("building manifest...")
    manifest = build_manifest(DATA_DIR)
    print(f"  {len(manifest)} files found")
    print(f"  {manifest.groupby('label').size().to_dict()}")

    # ── 2. train / test split ─────────────────────────────────────────────────
    print("\nsplitting train / test (80/20 by file, stratified)...")
    train, test = train_test_split_by_file(manifest, test_ratio=0.2)
    print(f"  train: {len(train)} files")
    print(f"  test : {len(test)} files")

    # ── 3. fit scaler on training data only ───────────────────────────────────
    print("\nfitting scaler on training files...")
    scaler = fit_scaler(train)
    print(f"  channel means: {scaler.mean_.round(1)}")
    print(f"  channel stds : {scaler.scale_.round(1)}")

    # ── 4. build datasets ─────────────────────────────────────────────────────
    print("\nbuilding training set...")
    X_train, y_train, meta_train = build_dataset(
        train, scaler, extract_features_fn=extract_features
    )
    print(f"  X_train : {X_train.shape}   dtype: {X_train.dtype}")
    print(f"  y_train : {y_train.shape}")
    print(f"  label counts: { {k: int((y_train == v).sum()) for k, v in LABEL_MAP.items()} }")

    print("\nbuilding test set...")
    X_test, y_test, meta_test = build_dataset(
        test, scaler, extract_features_fn=extract_features
    )
    print(f"  X_test  : {X_test.shape}   dtype: {X_test.dtype}")
    print(f"  y_test  : {y_test.shape}")
    print(f"  label counts: { {k: int((y_test == v).sum()) for k, v in LABEL_MAP.items()} }")

    # ── 5. save ───────────────────────────────────────────────────────────────
    print(f"\nsaving to {OUTPUT_DIR}/...")
    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "X_test.npy",  X_test)
    np.save(OUTPUT_DIR / "y_test.npy",  y_test)
    meta_train.to_csv(OUTPUT_DIR / "meta_train.csv", index=False)
    meta_test.to_csv(OUTPUT_DIR / "meta_test.csv",   index=False)
    print("  done.")

    return X_train, y_train, X_test, y_test, meta_train, meta_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, meta_train, meta_test = run()

    print("\n── ready for model ───────────────────────────────────────────────")
    print(f"  X_train : {X_train.shape}  →  feed to model.fit()")
    print(f"  y_train : {y_train.shape}  →  labels  {{0: hook, 1: straight, 2: upper}}")
    print(f"  X_test  : {X_test.shape}   →  feed to model.predict()")
    print(f"  y_test  : {y_test.shape}   →  ground truth for evaluation")
