"""
feature_extraction.py — feature extraction pipeline for windowed punch IMU data.

Input to all per-window functions:
    window : np.ndarray shape (window_size, n_channels)
              channels are [ax, ay, az, gx, gy, gz, accel_mag, gyro_mag]
              data is already filtered and normalized (output of pipeline.preprocess_file)

Master function:
    extract_features(windows)  →  (n_windows, n_features)
    Pass this as extract_features_fn to pipeline.build_dataset().

Feature groups and approximate counts (8 channels):
    time_domain   →  8 channels × 8 stats        =  64
    zcr           →  8 channels × 1              =   8
    mad           →  8 channels × 1              =   8
    jerk          →  3 accel axes × 1            =   3
    fft           →  8 channels × 4 stats        =  32
    band_energy   →  8 channels × 4 bands        =  32
    correlations  →  8×7/2                       =  28
    kinematics    →  quaternion + rotation stats  =  14
                                               ──────
                                               ~189 features total
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis


# channel index constants — matches pipeline.ALL_COLS_MAG order
ACCEL_IDX = [0, 1, 2]   # ax, ay, az
GYRO_IDX  = [3, 4, 5]   # gx, gy, gz
# index 6 = accel_mag, index 7 = gyro_mag


# ── 1. TIME DOMAIN ────────────────────────────────────────────────────────────

def extract_time_domain(window: np.ndarray) -> np.ndarray:
    """
    8 statistical features per channel.

    Input:  (window_size, n_channels)
    Output: (n_channels * 8,)

    Features: mean, std, rms, max, min, peak (max abs), skewness, kurtosis
    """
    features = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        features.extend([
            np.mean(x),
            np.std(x),
            np.sqrt(np.mean(x ** 2)),        # RMS
            np.max(x),
            np.min(x),
            np.max(np.abs(x)),               # peak
            float(skew(x)),
            float(kurtosis(x)),
        ])
    return np.array(features, dtype=np.float32)


def extract_zcr(window: np.ndarray) -> np.ndarray:
    """
    Zero crossing rate per channel — how often the signal crosses zero.
    High during an active punch, near-zero at rest.

    Input:  (window_size, n_channels)
    Output: (n_channels,)
    """
    signs = np.sign(window)
    crossings = np.sum(np.abs(np.diff(signs, axis=0)) > 0, axis=0)
    return crossings.astype(np.float32)


def extract_mad(window: np.ndarray) -> np.ndarray:
    """
    Mean absolute deviation per channel.
    Robust alternative to std — less sensitive to spike outliers.

    Input:  (window_size, n_channels)
    Output: (n_channels,)
    """
    return np.mean(np.abs(window - np.mean(window, axis=0)), axis=0).astype(np.float32)


def extract_jerk(window: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    Mean absolute jerk (rate of change of acceleration) per accel axis.
    Captures the sharpness of the punch onset — straight punches have a
    clean sharp jerk spike, uppercuts have a more gradual profile.

    Input:  (window_size, n_channels)
    Output: (3,)  — accel axes only
    """
    accel = window[:, ACCEL_IDX]                   # (window_size, 3)
    jerk  = np.diff(accel, axis=0) * fs            # (window_size-1, 3)
    return np.mean(np.abs(jerk), axis=0).astype(np.float32)


# ── 2. FREQUENCY DOMAIN ───────────────────────────────────────────────────────

def extract_fft_features(window: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    4 FFT-based features per channel: dominant frequency, spectral energy,
    spectral mean, spectral std.

    Input:  (window_size, n_channels)
    Output: (n_channels * 4,)
    """
    features = []
    n = window.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    for ch in range(window.shape[1]):
        mag = np.abs(np.fft.rfft(window[:, ch]))

        dominant_freq   = freqs[np.argmax(mag)]
        spectral_energy = np.sum(mag ** 2) / n
        spectral_mean   = np.sum(freqs * mag) / (np.sum(mag) + 1e-9)
        spectral_std    = np.sqrt(
            np.sum(((freqs - spectral_mean) ** 2) * mag) / (np.sum(mag) + 1e-9)
        )
        features.extend([dominant_freq, spectral_energy, spectral_mean, spectral_std])

    return np.array(features, dtype=np.float32)


def extract_band_energy(
    window: np.ndarray,
    fs: float = 100.0,
    bands: list = None,
) -> np.ndarray:
    """
    Energy in specific frequency bands per channel.
    Tuned for punch data:
        (0,  2) Hz  — drift / very slow motion
        (2, 10) Hz  — main punch motion band
        (10,20) Hz  — wrist snap / fast motion
        (20,50) Hz  — noise band (should be near zero after lowpass filter)

    Input:  (window_size, n_channels)
    Output: (n_channels * 4,)
    """
    if bands is None:
        bands = [(0, 2), (2, 10), (10, 20), (20, 50)]

    features = []
    n = window.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    for ch in range(window.shape[1]):
        mag_sq = np.abs(np.fft.rfft(window[:, ch])) ** 2
        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            features.append(np.sum(mag_sq[mask]) / (n + 1e-9))

    return np.array(features, dtype=np.float32)


# ── 3. CORRELATION FEATURES ───────────────────────────────────────────────────

def extract_correlations(window: np.ndarray) -> np.ndarray:
    """
    Pairwise Pearson correlations between all channel pairs.
    Captures cross-axis relationships — e.g. hook has high ax/gy correlation
    (linear motion and rotation coupled), straight does not.

    Input:  (window_size, n_channels)
    Output: (n_channels*(n_channels-1)/2,)  — upper triangle, no diagonal
    """
    corr = np.corrcoef(window.T)
    idx  = np.triu_indices(window.shape[1], k=1)
    return corr[idx].astype(np.float32)


# ── 4. KINEMATIC FEATURES (scikit-kinematics) ─────────────────────────────────

def extract_kinematics(window: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    Orientation and rotation features derived from gyro integration.

    Uses scikit-kinematics to integrate the gyroscope signal into a sequence
    of orientation quaternions. Each quaternion describes the 3D rotation of
    the wrist at that moment relative to the start of the window.

    This is particularly discriminative for punch types:
        straight  — small total rotation, mostly along one axis
        hook      — large rotation around the vertical (gz) axis
        uppercut  — rotation concentrated in the horizontal (gx/gy) axes

    Features extracted (14 total):
        quat_mean   (4) — mean quaternion components over the window
        quat_std    (4) — std of quaternion components (how much it varies)
        total_rotation  — total angle rotated from start to end of window (deg)
        peak_rot_rate   — max instantaneous angular velocity magnitude (deg/s)
        mean_rot_rate   — mean angular velocity magnitude (deg/s)
        rot_axis_ratio  — ratio of gz rotation to total (0=horizontal, 1=vertical)
        rot_range       — range of rotation angles (max - min)
        net_rotation    — signed net rotation angle at window end (deg)

    Input:  (window_size, n_channels)  — channels must include gyro at GYRO_IDX
    Output: (14,)
    """
    try:
        from skinematics.quat import calc_quat
    except ImportError:
        # if scikit-kinematics is unavailable, return zeros so the rest still works
        return np.zeros(14, dtype=np.float32)

    # gyro in deg/s → convert to rad/s for calc_quat
    gyro_dps = window[:, GYRO_IDX]                        # (window_size, 3)
    gyro_rps = np.deg2rad(gyro_dps)

    # integrate gyro → quaternion sequence (N, 4)
    # q0=[0,0,0] means we start at identity orientation
    # CStype="sf" = space-fixed frame (standard choice)
    quats = calc_quat(gyro_rps, q0=[0, 0, 0], rate=fs, CStype="sf")

    # quaternion stats
    quat_mean = quats.mean(axis=0)                         # (4,)
    quat_std  = quats.std(axis=0)                          # (4,)

    # rotation angle at each timestep: 2 * arccos(|w component|)
    w_component = np.clip(np.abs(quats[:, 0]), 0, 1)
    angles_deg  = np.rad2deg(2 * np.arccos(w_component))  # (N,)

    total_rotation = angles_deg[-1]                        # angle at end of window
    rot_range      = angles_deg.max() - angles_deg.min()
    net_rotation   = angles_deg[-1] - angles_deg[0]

    # angular velocity magnitude per timestep
    rot_rate = np.linalg.norm(gyro_dps, axis=1)           # (window_size,)
    peak_rot_rate = rot_rate.max()
    mean_rot_rate = rot_rate.mean()

    # how much of the rotation is around the vertical (gz) axis vs total
    gz_contribution = np.abs(gyro_dps[:, 2]).mean()
    total_contribution = rot_rate.mean() + 1e-9
    rot_axis_ratio = gz_contribution / total_contribution

    return np.array([
        *quat_mean,
        *quat_std,
        total_rotation,
        peak_rot_rate,
        mean_rot_rate,
        rot_axis_ratio,
        rot_range,
        net_rotation,
    ], dtype=np.float32)


# ── 5. MASTER EXTRACTOR ───────────────────────────────────────────────────────

def extract_features(
    windows: np.ndarray,
    fs: float = 100.0,
    use_fft: bool = True,
    use_band_energy: bool = True,
    use_correlations: bool = True,
    use_jerk: bool = True,
    use_zcr: bool = True,
    use_mad: bool = True,
    use_kinematics: bool = True,
) -> np.ndarray:
    """
    Run all feature extractors over every window and return a flat feature matrix.

    Input:  windows — (n_windows, window_size, n_channels)
    Output: X       — (n_windows, n_features)

    Pass this function as extract_features_fn to pipeline.build_dataset():
        X_train, y_train, meta = build_dataset(train, scaler,
                                               extract_features_fn=extract_features)

    Toggle individual feature groups on/off with the boolean flags.
    Start with use_stft=False — only enable if accuracy plateaus.
    """
    all_features = []

    for i in range(windows.shape[0]):
        w = windows[i]    # (window_size, n_channels)
        feats = [extract_time_domain(w)]

        if use_zcr:
            feats.append(extract_zcr(w))
        if use_mad:
            feats.append(extract_mad(w))
        if use_jerk:
            feats.append(extract_jerk(w, fs=fs))
        if use_fft:
            feats.append(extract_fft_features(w, fs=fs))
        if use_band_energy:
            feats.append(extract_band_energy(w, fs=fs))
        if use_correlations:
            feats.append(extract_correlations(w))
        if use_kinematics:
            feats.append(extract_kinematics(w, fs=fs))

        all_features.append(np.concatenate(feats))

    return np.stack(all_features, axis=0).astype(np.float32)
