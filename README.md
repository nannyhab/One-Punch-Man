# One Punch Man
### Real-Time IMU-Based Punch Detection for Interactive Gameplay

**Ahmed Alzahrani · Adnan Alhabian · Chibueze Anyachebelu · Mercy Muiruri · Novel Alam**

CS 396/496 | Northwestern University | Spring 2026

---

## Abstract

We present an end-to-end system that recognizes boxing punches from a wrist-worn 6-axis IMU and maps them to a fighting game in real time: a straight, hook, or uppercut drives the corresponding in-game attack. From 976 labeled recordings collected from five people, we built two classifiers — a laptop-side LIMU-BERT transfer-learning model and a fully on-device quantized CNN. An on-device activity gate suppresses idle and other non-punch motion so classification fires only on real punches, and recognized punches are injected as keypresses into a modified build of Ikemen GO. For real-time play we deployed the on-device model, which classifies directly on the chip and keeps latency low enough to feel responsive.

🎥 **Demo video:** [https://youtu.be/LgNwIqmmV0w](https://youtu.be/LgNwIqmmV0w)

---

## Table of Contents

1. [Goal and Motivation](#1-goal-and-motivation)
2. [Method](#2-method)
   - [Sensing: Hardware](#21-sensing-hardware)
   - [Sensing: Data Collection and Signal Processing](#22-sensing-data-collection-and-signal-processing)
   - [Machine Learning: LIMU-BERT (Method 1)](#23-machine-learning-limu-bert-method-1)
   - [Machine Learning: On-Device CNN (Method 2)](#24-machine-learning-on-device-cnn-method-2)
   - [Real-Time System and Game Integration](#25-real-time-system-and-game-integration)
3. [Key Results and Takeaways](#3-key-results-and-takeaways)
4. [Future Work](#4-future-work)

---

## 1. Goal and Motivation

Most fighting games translate button presses into on-screen actions. We reverse that process: a wrist-worn IMU recognizes real punches and maps them to in-game attacks, so throwing a straight, hook, or uppercut makes the character perform the matching move in real time.

Our interest was the gap between offline accuracy and real-world usability. A model can score well on a test set yet feel unresponsive in actual use, so we connected the classifier to a game and evaluated the whole pipeline — from sensor to in-game action — rather than the model in isolation. The driving question was not whether we could classify punches accurately, but whether the predictions were good enough to support a real interaction.

---

## 2. Method

Our approach pairs a wrist-worn sensing front end with two alternative classifiers that differ in size, latency, and where they run: a laptop-side transfer-learning model (Method 1) and a fully on-device CNN (Method 2).

**End-to-end pipeline:**

```
Wrist IMU → Activity Gate (VAD) → Punch Classifier (LIMU-BERT or on-device CNN) → Keypress → Ikemen GO
```

### 2.1 Sensing: Hardware

The sensor used was the **Seeed XIAO nRF52840 Sense**, which integrates an nRF52840 microcontroller and a built-in LSM6DS3TR-C 6-axis IMU. Its small footprint allowed wrist-mounting without hindering punching, and it enumerates as a USB serial device for easy logging.

| Component | Specification |
|-----------|---------------|
| Board | Seeed XIAO nRF52840 Sense |
| Microcontroller | nRF52840 |
| IMU | LSM6DS3TR-C (6-axis accel + gyro) |
| Sample rate | 104 Hz |
| Sensor ranges | ±4 g accel, ±250 dps gyro |
| Firmware | Zephyr + TensorFlow Lite Micro |
| Interface | USB serial, 115 200 baud |

The ±4 g and ±250 dps ranges leave headroom for hard punches, and samples are read from the IMU FIFO by interrupt rather than tight-loop polling.

### 2.2 Sensing: Data Collection and Signal Processing

Each participant wore the sensor and threw repeated punches of three target classes (straight, hook, and uppercut), plus negative examples (idle and random arm motion under `NULL CLASS`) to teach the model what to ignore. Recordings were logged as CSV — one row per timestep: three accelerometer channels (mg) and three gyroscope channels (dps).

| Punch Type | Recordings |
|------------|-----------|
| Hook | 349 |
| Straight | 323 |
| Uppercut | 304 |
| **Total** | **976** |

*Dataset collected from five people across three punch types.*

A copy of the dataset (`data_raw16/`) was also exported at the sensor's native int16 fixed-point scaling, so the int8-quantized on-device models train on the same raw representation the hardware produces at deployment.

**Preprocessing** applies only sensor-level cleaning so every model trains on identical data. Columns are renamed to a fixed six-channel layout (`ax, ay, az, gx, gy, gz`). The IMU's true output data rate is 104 Hz; after rate verification, a **30 Hz Butterworth low-pass filter** removes jitter while preserving punch motion. Normalization, windowing, and unit conversion are deferred to each model. Train/test split is **80/20 by whole recording** (780/196) to prevent testing on motion nearly identical to training data.

### 2.3 Machine Learning: LIMU-BERT (Method 1)

LIMU-BERT is an encoder pretrained on large amounts of unlabeled IMU data, capturing general human movement. We attached a small classification head and fine-tuned it on our punches — far more data-efficient than training from scratch. Input was resampled to 20 Hz, converted to g and rad/s, and cut into **2-second windows with 25% overlap**.

| Metric | Frozen | Partial Fine-Tuning |
|--------|--------|---------------------|
| Window macro-F1 | 0.927 | **0.977** |
| File macro-F1 | 0.970 | **0.995** |
| LOPO window macro-F1 | 0.576 ± 0.189 | — |

Partial fine-tuning was best, with the gain coming from overlapping windows voting out single-window errors. Under **leave-one-person-out (LOPO)** evaluation, accuracy falls to ~0.57, as expected since the model targets our five-person group rather than a general population.

**Why LIMU-BERT over HARNet10:**

| | LIMU-BERT | HARNet10 |
|--|-----------|---------|
| Window length | 2 s | 10 s |
| Windows per recording | Several (voting) | One (~62% padding) |
| Model size | ~226 kB | ~41 MB |
| Run-time download | None | Required |

### 2.4 Machine Learning: On-Device CNN (Method 2)

Method 2 moves classification entirely onto the microcontroller. We trained a **two-stage pipeline**:

1. **Binary punch detector** — gates incoming motion
2. **3-class type classifier** (hook / straight / uppercut) — runs only when the binary score exceeds ~70%

Both are int8 TFLite models embedded as C arrays in the firmware, consuming raw int16-scaled IMU windows directly (no unit conversion). Later firmware scores **five overlapping 50-sample slices** across a ~1.7 s ring buffer and keeps the best score.

**LOPO evaluation results:**

| Stage | Accuracy | F1 |
|-------|----------|-----|
| Binary punch detector | 87.4% | 85.7% |
| Type classifier (gated windows) | 91.5% | 86.3% macro |

Running inference on-device removed most of the ~3-second streaming delay, since the chip sends only the label. This was the configuration used for live gameplay.

### 2.5 Real-Time System and Game Integration

**Activity detection.** The device runs a simple activity gate: it buffers recent IMU samples and classifies only when enough exceed basic accel/gyro thresholds (~2000 mg, ~200 dps). On the streaming route a lightweight binary detector further reduces false positives by transmitting a burst to the laptop only when a real punch is detected.

**Game integration.** Built on [Ikemen GO](https://github.com/ikemen-engine/Ikemen-GO), a free cross-platform engine compatible with M.U.G.E.N. Modifications for the demo:
- Remapped uppercut to a single key
- Lowered AI difficulty
- Renamed players
- Moved opponent closer for short-range punches

**Runtime pipeline.** The sensor acts only on a detected punch. A keypress thread reads `(label, confidence)` results and injects OS-level keypresses into Ikemen GO via `pynput`.

---

## 3. Key Results and Takeaways

| Metric | Result |
|--------|--------|
| Straight / uppercut confidence | 85–95% |
| Hook confidence | 75–90% |
| Keypress latency (detection → classification → injection) | 0.1–0.3 s |
| End-to-end lag: streaming vs. on-device | ~3 s vs. sub-second |

> *Confidence ranges and observations are from live testing, not a controlled methodological study.*

**Headline results:**
- LIMU-BERT achieved **0.977 / 0.995 window / file macro-F1** on the 80/20 split
- On-device CNN reached **87.4% punch-detection** and **91.5% punch-type accuracy** (LOPO)
- Moving to on-device inference reduced end-to-end latency from ~3 seconds to under 1 second

The on-device model was deployed for gameplay because it brings streaming latency under a second while staying accurate on unseen users. Hook was the most misclassified class; no false positives arose from idle motion, though ambiguous punch-like motions occasionally scored near the threshold.

---

## 4. Future Work

- Support additional actions: blocking, dodging, and movement
- Punch combination recognition
- Tighter game integration (drive characters directly without a keyboard layer)
- User studies with participants outside the data-collection group
