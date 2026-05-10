# Pretrained IMU Models — Benchmark Set for *One Punch Man*

Wrist-worn, 6-axis, 104 Hz, 3 punch classes.
## Quick Comparison

| # | Model | Family | Axes | Rate | Pretrain Data | Repo |
|---|---|---|---|---|---|---|
| 1 | **LIMU-BERT** | BERT-style MAE | 6 | 20 Hz | HHAR, UCI, MotionSense, Shoaib | `dapowan/LIMU-BERT-Public` |
| 2 | **Oxford harnet10** | SSL CNN (ResNet-18) | 3 (accel) | 30 Hz | UK Biobank, 700k person-days, **wrist** | `OxWearables/ssl-wearables` |
| 3 | **UniMTS** | Graph + CLIP-style | 6 | flexible | HumanML3D synthetic IMU, all locations | `xiyuanzh/UniMTS` |
| 4 | **PRIMUS** | Cross-modal contrastive (CNN) | 6 | 50 / 200 Hz | Ego4D + Ego-Exo4D (head) | `Nokia-Bell-Labs/pretrained-imu-encoders` |


## Per-Model Notes (essentials only)

### 1. LIMU-BERT *(your current plan)*
Lite BERT-like masked-prediction transformer. Pretrained on 4 public smartphone HAR datasets. **Decimate 100→20 Hz**, drop mag, segment to 120 samples. Repo ships frozen-BERT + GRU classifier head. License: research-permissive.

### 2. Oxford harnet10 — top CNN pick
Multi-task SSL (TPN-style). Wrist-pretrained on the largest open IMU corpus. **3-axis accel only** is the cost. Decimate 100→30 Hz with anti-alias filter, take x/y/z accel, pad to 300 samples, freeze backbone, train 3-class head. Reported median F1 +24% over strong baselines across 8 HAR datasets. License: Oxford academic-only (fine for class project).

### 3. UniMTS — top graph/cross-modal pick
GNN encoder + text-contrastive (CLIP-style). Pretrained on synthetic IMU from HumanML3D mocap, **all body locations including wrist**. ~5M params, fine-tunes in <2 min on UCI-HAR. Rotation invariance helps placement robustness but may flatten gravity-direction cues that distinguish straight vs uppercut — train a non-augmented variant as a control. License: MIT-style.

### 4. PRIMUS — top contrastive/multimodal pick
Improves on IMU2CLIP with combined SSL + video/text contrastive + nearest-neighbor objective. Public Zenodo checkpoint. Best few-shot performance in literature (<500 samples/class), directly relevant to your dataset size. **Head-mounted pretraining (Aria glasses)** is the main mismatch — head IMU sees head bob, not arm strikes. License: CC-BY-NC.


## Caveats


- **Sampling rate**: punches contain content up to 30–40 Hz (esp. gyro). Models running at ≤30 Hz partially throw that away. Your 20 Hz Butterworth already pre-bandlimits — keep this in mind when interpreting results from low-rate models.
- **Domain shift**: wrist (Oxford, UniMTS-wrist) > smartphone-pocket (LIMU-BERT) > head (PRIMUS). Weight your expectations accordingly.
- **Fine-tuning recipe**: start with frozen-encoder + linear head; if data permits, also unfreeze the last 1–2 blocks at low LR. Report both numbers.
- **Hidden gap in the literature**: no public model is wrist + 9-axis + ≥100 Hz. If transfer underwhelms, **self-pretrain** LIMU-BERT on your own unlabeled wrist data before the 3-class fine-tune — often beats transferring from a larger but mismatched corpus.
