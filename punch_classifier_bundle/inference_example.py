"""
Copy-paste runnable inference demo.

Loads LIMU-BERT in PRODUCTION mode (trained on all 976 files, partial
fine-tuning, 22-epoch budget) and classifies a synthetic 4-second IMU
recording. This is the checkpoint to use for real-time game testing.

If you want the academically reportable model instead (trained on the
80% train split with a held-out 20% test split, 99.5% file F1 on
unseen recordings from the same 5 subjects), switch the
checkpoint_path below to `limu_bert_partial.pt`.

To run with your own data:
- Replace `make_fake_window()` with code that loads ONE punch recording.
- The input must be a 2D float32 array of shape (N_samples, 6) at 104 Hz,
  with channel order [ax, ay, az, gx, gy, gz] in mg and dps respectively.
- The Predictor handles resampling (104→20 Hz), unit conversion
  (mg→g, dps→rad/s), windowing, and padding internally.

Run:
    python inference_example.py
"""
import sys
from pathlib import Path

import numpy as np

# Make src/ importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from limu_bert import Predictor  # noqa: E402


def make_fake_window() -> np.ndarray:
    """Synthetic 4-second IMU window at 104 Hz, 6 channels (ax,ay,az,gx,gy,gz)."""
    rng = np.random.default_rng(0)
    n = 400  # ≈ 3.85 s at 104 Hz, matches the real recording length
    x = rng.normal(0, 1, size=(n, 6)).astype(np.float32)
    x[:, :3] *= 500  # accel in mg
    x[:, 3:] *= 100  # gyro in dps
    return x


def main() -> None:
    window = make_fake_window()
    print(f"input window: shape={window.shape}, dtype={window.dtype}")

    # PRODUCTION checkpoint — partial fine-tune, trained on ALL 976 files.
    # Use this for live game inference.
    p = Predictor(
        checkpoint_path=HERE / "artifacts/limu_bert/checkpoints/limu_bert_production.pt",
        scaler_path=None,  # z-scoring disabled (matches training)
        # production_info.json carries the confidence threshold inherited from
        # the val-based partial-mode run, plus full provenance metadata.
        threshold_path=HERE / "artifacts/limu_bert/metrics/production_info.json",
    )
    label, conf = p.predict(window)
    print(f"LIMU-BERT production: label={label!r}  confidence={conf:.3f}")

    print("\nLabels are one of: 'hook', 'straight', 'upper', or 'none' (low confidence).")
    print("\nFor reference / metrics reporting, the val-based checkpoints are in")
    print("./initialtraining/  along with the HARNet10 comparison models.")


if __name__ == "__main__":
    main()
