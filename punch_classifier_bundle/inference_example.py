"""
Copy-paste runnable inference demo.

Loads BOTH trained models (LIMU-BERT and HARNet10) in their best mode
(`partial`) and classifies a synthetic 4-second IMU recording.

To run with your own data:
- Replace `make_fake_window()` with code that loads ONE punch recording.
- The input must be a 2D float32 array of shape (N_samples, 6) at 104 Hz,
  with channel order [ax, ay, az, gx, gy, gz] in mg and dps respectively.
- For HARNet10, the gyro channels are dropped automatically — you can
  still pass all 6 channels, the Predictor handles it.

Run:
    python inference_example.py
"""
import sys
from pathlib import Path

import numpy as np

# Make src/ importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from limu_bert import Predictor as LimuBertPredictor  # noqa: E402
from harnet10 import Predictor as HarnetPredictor  # noqa: E402


def make_fake_window() -> np.ndarray:
    """Synthetic 4-second IMU window at 104 Hz, 6 channels (ax,ay,az,gx,gy,gz)."""
    rng = np.random.default_rng(0)
    n = 400  # 400 samples ≈ 3.85 s at 104 Hz — same shape as real recordings
    x = rng.normal(0, 1, size=(n, 6)).astype(np.float32)
    x[:, :3] *= 500  # accel in mg
    x[:, 3:] *= 100  # gyro in dps
    return x


def main() -> None:
    window = make_fake_window()
    print(f"input window: shape={window.shape}, dtype={window.dtype}")

    # --- LIMU-BERT (transformer, 6-channel, 20 Hz) ---------------------------
    limu = LimuBertPredictor(
        checkpoint_path=HERE / "artifacts/limu_bert/checkpoints/limu_bert_partial.pt",
        scaler_path=None,  # z-scoring is disabled (matches training)
        threshold_path=HERE / "artifacts/limu_bert/metrics/partial_metrics.json",
    )
    label, conf = limu.predict(window)
    print(f"LIMU-BERT (partial): label={label!r}  confidence={conf:.3f}")

    # --- HARNet10 (1D CNN, 3-channel accel only, 30 Hz) ----------------------
    # First call downloads the OxWearables backbone arch via torch.hub
    # (~40 MB, ~10s on a fresh machine). Cached afterwards.
    harnet = HarnetPredictor(
        checkpoint_path=HERE / "artifacts/harnet10/checkpoints/harnet10_partial.pt",
        scaler_path=None,
        threshold_path=HERE / "artifacts/harnet10/metrics/partial_metrics.json",
    )
    label, conf = harnet.predict(window)
    print(f"HARNet10  (partial): label={label!r}  confidence={conf:.3f}")

    print("\nLabels are one of: 'hook', 'straight', 'upper', or 'none' (low confidence).")


if __name__ == "__main__":
    main()
