"""
pipeline.py — preprocessing and feature extraction pipeline for punch IMU data.

Data lives at:  data/{person}/{label}/{label_timestamp}.csv
Columns:        sample_idx, acc_x_mg, acc_y_mg, acc_z_mg, gyr_x_dps, gyr_y_dps, gyr_z_dps

Intended usage (in order):
    1. build_manifest(data_dir)              → know what files you have
    2. train_test_split_by_file(manifest)    → split before touching any data
    3. fit_scaler(train_manifest)            → fit on train files only
    4. build_dataset(manifest, scaler)       → X (n_windows, n_features), y (n_windows,), meta
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ── Internal column names used throughout the pipeline ────────────────────────
# Real data uses acc_x_mg etc. — we rename immediately on load so all
# downstream functions can assume these standard names.
ACCEL_COLS = ["ax", "ay", "az"]
GYRO_COLS  = ["gx", "gy", "gz"]
MAG_COLS   = ["accel_mag", "gyro_mag"]
ALL_COLS   = ACCEL_COLS + GYRO_COLS           # 6 channels (before magnitude)
ALL_COLS_MAG = ALL_COLS + MAG_COLS            # 8 channels (after magnitude)

LABEL_MAP  = {"hook": 0, "straight": 1, "upper": 2}


# ── 1. MANIFEST ───────────────────────────────────────────────────────────────

def build_manifest(data_dir: str | Path) -> pd.DataFrame:
    """
    Walk the data directory and return a DataFrame describing every CSV file.

    Directory structure expected:
        data_dir/{person}/{label}/{label_timestamp}.csv

    Returns
    -------
    pd.DataFrame with columns:
        path    — absolute Path to the CSV
        person  — e.g. "mercy", "ahmed"
        label   — e.g. "hook", "straight", "upper"

    One row per CSV file. No data is loaded yet.
    """
    data_dir = Path(data_dir)
    records = []

    for csv_path in sorted(data_dir.rglob("*.csv")):
        label  = csv_path.parent.name    # e.g. "hook"
        person = csv_path.parent.parent.name  # e.g. "mercy"

        if label not in LABEL_MAP:
            continue  # skip any stray files not in a known class folder

        records.append({
            "path":   csv_path,
            "person": person,
            "label":  label,
        })

    return pd.DataFrame(records, columns=["path", "person", "label"])


# ── 2. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────

def train_test_split_by_file(
    manifest: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the manifest into train and test subsets by FILE (not by window).

    Splitting by file prevents data leakage: windows from the same punch
    cannot appear in both train and test sets.

    Stratified by label so class balance is preserved in both splits.

    Parameters
    ----------
    manifest     : output of build_manifest()
    test_ratio   : fraction of files held out for test (default 0.2)
    random_state : for reproducibility

    Returns
    -------
    train_manifest, test_manifest  — two DataFrames with the same columns as manifest
    """
    train_rows, test_rows = train_test_split(
        manifest,
        test_size=test_ratio,
        stratify=manifest["label"],
        random_state=random_state,
    )

    return train_rows.reset_index(drop=True), test_rows.reset_index(drop=True)


# ── 3. PREPROCESSING (per file) ───────────────────────────────────────────────

def load_file(path: str | Path) -> pd.DataFrame:
    """
    Load a single CSV and rename columns to the internal standard names.

    Raw columns:   sample_idx, acc_x_mg, acc_y_mg, acc_z_mg, gyr_x_dps, gyr_y_dps, gyr_z_dps
    Output columns: ax, ay, az, gx, gy, gz   (sample_idx dropped)

    Returns
    -------
    pd.DataFrame  shape (n_samples, 6)
    """
    RENAME = {
        "acc_x_mg":  "ax",
        "acc_y_mg":  "ay",
        "acc_z_mg":  "az",
        "gyr_x_dps": "gx",
        "gyr_y_dps": "gy",
        "gyr_z_dps": "gz",
    }

    df = pd.read_csv(path)
    df = df.rename(columns=RENAME)
    df = df[ALL_COLS]   # drop sample_idx, keep only ax ay az gx gy gz in order
    return df


def lowpass_filter(data: np.ndarray, cutoff_hz: float = 30.0, fs: float = 100.0, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter along the time axis.

    Removes high-frequency noise above cutoff_hz. Punches live in ~2–20 Hz,
    so 30 Hz cutoff keeps all relevant signal while removing vibration noise.

    Parameters
    ----------
    data       : np.ndarray shape (n_samples, 6)
    cutoff_hz  : filter cutoff in Hz (default 30)
    fs         : sampling rate in Hz (default 100)
    order      : filter order (default 4)

    Returns
    -------
    np.ndarray  same shape as input, filtered values
    """
    from scipy import signal

    nyq = fs / 2.0
    b, a = signal.butter(order, cutoff_hz / nyq, btype="low")
    return signal.filtfilt(b, a, data, axis=0).astype(np.float32)


def fit_scaler(train_manifest: pd.DataFrame) -> object:
    """
    Fit a StandardScaler on the raw signal from all training files.

    Loads and concatenates every file in train_manifest, fits the scaler
    on that combined signal. The scaler is fit on training data ONLY —
    never call this with test files included.

    Parameters
    ----------
    train_manifest : subset of manifest containing only training files

    Returns
    -------
    fitted sklearn.preprocessing.StandardScaler
    """
    from sklearn.preprocessing import StandardScaler

    all_data = []
    for path in train_manifest["path"]:
        df = load_file(path)
        filtered = lowpass_filter(df.to_numpy(dtype=np.float32))
        all_data.append(filtered)

    combined = np.concatenate(all_data, axis=0)   # (total_samples_across_all_train_files, 6)

    scaler = StandardScaler()
    scaler.fit(combined)
    return scaler

def normalize(data: np.ndarray, scaler) -> np.ndarray:
    """
    Apply a fitted StandardScaler to a single recording.

    Does NOT fit — the scaler must already be fitted on training data via
    fit_scaler(). Calling this on test data is safe because no fitting happens.

    Parameters
    ----------
    data   : np.ndarray shape (n_samples, 6)  — filtered, raw-unit values
    scaler : fitted StandardScaler from fit_scaler()

    Returns
    -------
    np.ndarray  shape (n_samples, 6)  — zero mean, unit variance per channel
    """
    return scaler.transform(data).astype(np.float32)

def add_magnitude(data: np.ndarray) -> np.ndarray:
    """
    Append accel magnitude and gyro magnitude as two extra channels.

    accel_mag = sqrt(ax² + ay² + az²)
    gyro_mag  = sqrt(gx² + gy² + gz²)

    Do this AFTER normalization so magnitudes are on the same scale.

    Parameters
    ----------
    data : np.ndarray shape (n_samples, 6)

    Returns
    -------
    np.ndarray  shape (n_samples, 8)
    """
    accel_mag = np.linalg.norm(data[:, :3], axis=1, keepdims=True)  # (n_samples, 1)
    gyro_mag  = np.linalg.norm(data[:, 3:], axis=1, keepdims=True)  # (n_samples, 1)
    return np.concatenate([data, accel_mag, gyro_mag], axis=1).astype(np.float32)

def preprocess_file(path: str | Path, scaler) -> np.ndarray:
    """
    Full preprocessing pipeline for a single CSV file.

    Steps (in order):
        1. load_file      → rename columns, drop sample_idx
        2. lowpass_filter → remove high-frequency noise
        3. normalize      → zero mean, unit variance per channel
        4. add_magnitude  → append accel_mag and gyro_mag channels

    Parameters
    ----------
    path   : path to a single CSV punch recording
    scaler : fitted StandardScaler from fit_scaler()

    Returns
    -------
    np.ndarray  shape (n_samples, 8)
    """
    df       = load_file(path)
    filtered = lowpass_filter(df.to_numpy(dtype=np.float32))
    normed   = normalize(filtered, scaler)
    return add_magnitude(normed)


# ── 4. WINDOWING ──────────────────────────────────────────────────────────────

def segment(
    data: np.ndarray,
    window_size: int = 100,
    hop_size: int = 25,
) -> np.ndarray:
    """
    Slice a single recording into overlapping windows.

    At 100 Hz: window_size=100 → 1 second per window
                hop_size=25    → 0.25 second hop (75% overlap)

    A 400-sample recording produces ~13 windows with these defaults.
    Files shorter than window_size are skipped (returns empty array).

    Parameters
    ----------
    data        : np.ndarray shape (n_samples, n_channels)
    window_size : number of samples per window (default 100)
    hop_size    : step between window starts in samples (default 25)

    Returns
    -------
    np.ndarray  shape (n_windows, window_size, n_channels)
                returns shape (0, window_size, n_channels) if file is too short
    """
    n_samples, n_channels = data.shape

    if n_samples < window_size:
        return np.empty((0, window_size, n_channels), dtype=np.float32)

    starts  = np.arange(0, n_samples - window_size + 1, hop_size)
    windows = np.stack([data[s : s + window_size] for s in starts], axis=0)
    return windows.astype(np.float32)


# ── 5. DATASET ASSEMBLY ───────────────────────────────────────────────────────

def build_dataset(
    manifest: pd.DataFrame,
    scaler,
    window_size: int = 100,
    hop_size: int = 25,
    extract_features_fn=None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Process every file in the manifest and assemble the final dataset.

    For each file:
        1. preprocess_file  → (n_samples, 8)
        2. segment          → (n_windows, window_size, 8)
        3. extract_features_fn (if provided) → (n_windows, n_features)
           otherwise keeps raw windows      → (n_windows, window_size, 8)
        4. label every window with the file's label

    Parameters
    ----------
    manifest           : output of build_manifest() (train or test subset)
    scaler             : fitted StandardScaler from fit_scaler()
    window_size        : passed to segment()
    hop_size           : passed to segment()
    extract_features_fn: callable (n_windows, window_size, n_channels) → (n_windows, n_features)
                         if None, raw windows are returned (useful for deep learning models)

    Returns
    -------
    X    : np.ndarray shape (total_windows, n_features) or (total_windows, window_size, 8)
    y    : np.ndarray shape (total_windows,)  — integer labels (see LABEL_MAP)
    meta : pd.DataFrame  shape (total_windows, 3)
               columns: path, person, label, window_idx
               tracks which file and window each row in X came from
               use this to avoid leakage and for per-person analysis
    """
    all_X, all_y, all_meta = [], [], []

    for _, row in manifest.iterrows():
        data    = preprocess_file(row["path"], scaler)   # (n_samples, 8)
        windows = segment(data, window_size, hop_size)   # (n_windows, window_size, 8)

        if len(windows) == 0:
            continue

        if extract_features_fn is not None:
            X_file = extract_features_fn(windows)        # (n_windows, n_features)
        else:
            X_file = windows                             # (n_windows, window_size, 8)

        n_windows = len(windows)
        all_X.append(X_file)
        all_y.append(np.full(n_windows, LABEL_MAP[row["label"]], dtype=np.int32))
        all_meta.append(pd.DataFrame({
            "path":       str(row["path"]),
            "person":     row["person"],
            "label":      row["label"],
            "window_idx": np.arange(n_windows),
        }))

    X    = np.concatenate(all_X,  axis=0)
    y    = np.concatenate(all_y,  axis=0)
    meta = pd.concat(all_meta, ignore_index=True)

    return X, y, meta
