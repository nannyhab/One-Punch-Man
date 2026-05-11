# Pipeline Usage Guide

How to go from raw CSV files to a labeled feature matrix ready for a classifier.

---

# You can also use prepare_data.py to use a pre-made pipeline or follow these steps manually.
- call in terminal or read the prepare_data file.
```
 python preprocessing/prepare_data.py
```


## Data layout

```
data/
  {person}/          ← adnan, ahmed, chi, mercy, novel
    {label}/         ← hook, straight, upper
      {label}_{timestamp}.csv
```

Each CSV is **one punch recording** (~400 rows × 7 columns at 100 Hz):

```
sample_idx | acc_x_mg | acc_y_mg | acc_z_mg | gyr_x_dps | gyr_y_dps | gyr_z_dps
         0 |      624 |      424 |      686 |      13.5 |       9.9 |      17.8
         1 |      630 |      412 |      694 |      13.4 |      10.8 |      17.0
       ...
```

- Accelerometer unit: **mg** (milligravity) — values ~600–700 range
- Gyroscope unit: **deg/s** — values ~10–18 range
- These are on completely different scales, which is why normalization is essential

---

## Quick start — full pipeline in 5 lines

```python
from preprocessing.pipeline import (
    build_manifest, train_test_split_by_file, fit_scaler, build_dataset
)
from preprocessing.feature_extraction import extract_features

manifest          = build_manifest("data")
train, test       = train_test_split_by_file(manifest)
scaler            = fit_scaler(train)
X_train, y_train, meta_train = build_dataset(train, scaler, extract_features_fn=extract_features)
X_test,  y_test,  meta_test  = build_dataset(test,  scaler, extract_features_fn=extract_features)
```

**Output shapes:**
```
X_train : (10044, 189)   ← 10,044 windows × 189 features
y_train : (10044,)       ← integer label per window  {0: hook, 1: straight, 2: upper}
X_test  : (2544,  189)
y_test  : (2544,)
```

---

## Step-by-step walkthrough

### Step 1 — Build manifest

```python
from preprocessing.pipeline import build_manifest

manifest = build_manifest("data")
# DataFrame shape: (976, 3)
```

| path | person | label |
|------|--------|-------|
| data/adnan/hook/hook_20260508_193144.csv | adnan | hook |
| data/mercy/straight/straight_20260508_200006.csv | mercy | straight |
| ... | ... | ... |

One row per CSV file. No data loaded yet. Label comes from the folder name.

```
976 files total:
  hook     : 349
  straight : 323
  upper    : 304
```

---

### Step 2 — Train/test split by file

```python
from preprocessing.pipeline import train_test_split_by_file

train, test = train_test_split_by_file(manifest, test_ratio=0.2)
# train shape: (780, 3)
# test  shape: (196, 3)
```

Split is done at the **file level**, not the window level. This prevents leakage — windows from the same punch recording cannot appear in both train and test.

Stratified by label, so class proportions are preserved in both splits:
```
train → hook: 279  straight: 258  upper: 243
test  → hook:  70  straight:  65  upper:  61
```

---

### Step 3 — Fit scaler on training data only

```python
from preprocessing.pipeline import fit_scaler

scaler = fit_scaler(train)
```

Internally this:
1. Loads and filters every training file → list of `(400, 6)` arrays
2. Concatenates them → `(312_000, 6)`  (780 files × 400 samples)
3. Fits a `StandardScaler` — computes mean and std for each of the 6 channels

```
channel means (raw units):   ax=279  ay=556  az=425  gx=0.9  gy=-7.5  gz=-1.8
channel stds  (raw units):   ax=794  ay=724  az=613  gx=65   gy=66    gz=84
```

The scaler is fit on **training data only** — never include test files here.

---

### Step 4 — Preprocess a single file (what happens inside build_dataset)

```python
from preprocessing.pipeline import load_file, lowpass_filter, normalize, add_magnitude

# 4a. Load and rename columns
df = load_file("data/mercy/hook/hook_20260508_200006.csv")
# DataFrame shape: (400, 6)
# columns: ax, ay, az, gx, gy, gz   (sample_idx dropped, columns renamed)

# 4b. Low-pass filter — remove noise above 30 Hz
filtered = lowpass_filter(df.to_numpy(dtype="float32"))
# ndarray shape: (400, 6)   ← same shape, values smoothed

# 4c. Normalize — zero mean, unit variance per channel
normed = normalize(filtered, scaler)
# ndarray shape: (400, 6)   ← values now roughly in [-3, 3] range

# 4d. Add magnitude channels
data = add_magnitude(normed)
# ndarray shape: (400, 8)
# columns: ax  ay  az  gx  gy  gz  accel_mag  gyro_mag
```

Or equivalently, use the single wrapper:

```python
from preprocessing.pipeline import preprocess_file

data = preprocess_file("data/mercy/hook/hook_20260508_200006.csv", scaler)
# ndarray shape: (400, 8)
```

---

### Step 5 — Window the recording

```python
from preprocessing.pipeline import segment

windows = segment(data, window_size=100, hop_size=25)
# ndarray shape: (13, 100, 8)
#                 ↑   ↑    ↑
#            n_windows  │  n_channels (ax ay az gx gy gz accel_mag gyro_mag)
#                  window_size (100 samples = 1 second at 100 Hz)
```

How the 13 windows are cut from 400 samples:
```
sample: 0         100       200       300       400
        [----w0----]
             [----w1----]
                  [----w2----]
                       ...       (hop = 25 samples = 0.25 s)
                                            [---w12---]
```

All 13 windows inherit the same label as the file they came from.

---

### Step 6 — Extract features from one window

```python
from preprocessing.feature_extraction import (
    extract_time_domain, extract_zcr, extract_mad, extract_jerk,
    extract_fft_features, extract_band_energy, extract_correlations,
    extract_kinematics
)

w = windows[0]    # shape: (100, 8)   ← one window

extract_time_domain(w)    # → (64,)   8 stats × 8 channels
extract_zcr(w)            # → (8,)    1 per channel
extract_mad(w)            # → (8,)    1 per channel
extract_jerk(w)           # → (3,)    accel axes only
extract_fft_features(w)   # → (32,)   4 stats × 8 channels
extract_band_energy(w)    # → (32,)   4 bands × 8 channels
extract_correlations(w)   # → (28,)   8×7/2 channel pairs
extract_kinematics(w)     # → (14,)   quaternion + rotation features
```

---

### Step 7 — Extract features from all windows of all files

```python
from preprocessing.feature_extraction import extract_features

# for a single file's windows
X_file = extract_features(windows)
# ndarray shape: (13, 189)   ← 13 windows, 189 features each

# for the full dataset — pass to build_dataset
X_train, y_train, meta_train = build_dataset(
    train, scaler, extract_features_fn=extract_features
)
# X_train shape: (10044, 189)   ← all windows from all training files
# y_train shape: (10044,)
# meta    shape: (10044, 4)     ← path, person, label, window_idx per row
```

**Feature breakdown (189 total):**

| Group | Function | Output shape | Notes |
|-------|----------|-------------|-------|
| Time domain | `extract_time_domain` | (64,) | mean, std, rms, max, min, peak, skew, kurtosis per channel |
| Zero crossing rate | `extract_zcr` | (8,) | how often signal crosses zero |
| Mean abs deviation | `extract_mad` | (8,) | robust alternative to std |
| Jerk | `extract_jerk` | (3,) | rate of change of accel — sharpness of punch onset |
| FFT | `extract_fft_features` | (32,) | dominant freq, spectral energy, mean, std |
| Band energy | `extract_band_energy` | (32,) | energy in 0–2, 2–10, 10–20, 20–50 Hz bands |
| Correlations | `extract_correlations` | (28,) | pairwise Pearson correlations between channels |
| Kinematics | `extract_kinematics` | (14,) | orientation quaternion + rotation stats from gyro integration |

---

## Using raw windows (for neural networks)

Omit `extract_features_fn` to get raw 3D windows instead of a flat feature matrix:

```python
X_train, y_train, meta = build_dataset(train, scaler)
# X_train shape: (10044, 100, 8)   ← (n_windows, window_size, n_channels)
# y_train shape: (10044,)
```

---

## Using the meta DataFrame

`meta` lets you trace any row in `X` back to its source file and window:

```python
X_train, y_train, meta = build_dataset(train, scaler, extract_features_fn=extract_features)

# meta shape: (10044, 4)
# columns: path, person, label, window_idx

# e.g. check which person each window came from
print(meta.groupby("person")["label"].value_counts())

# e.g. find all windows from a specific file
mask = meta["path"].str.contains("mercy")
X_mercy = X_train[mask]
```

---

## Label map

```python
from preprocessing.pipeline import LABEL_MAP
# {"hook": 0, "straight": 1, "upper": 2}
```

`y_train` and `y_test` contain integers. Use `LABEL_MAP` to convert back to strings.
