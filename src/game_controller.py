"""
Real-time punch-to-game controller.

Reads IMU data from a BLE sensor (or a keyboard stub for testing),
classifies punches with LIMU-BERT, and injects keypresses into IKEMEN GO.

Key mappings (Player 1):
    straight → A
    hook     → S
    upper    → D

Usage
-----
Stub mode (test without sensor):
    python src/game_controller.py --mode stub

BLE mode (live sensor):
    python src/game_controller.py --mode ble \\
        --ble-address "XX:XX:XX:XX:XX:XX" \\
        --ble-uuid "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"

    Omit --ble-address to scan and list nearby BLE devices instead.

BLE byte format flags:
    --format float32x6   (default) 6 × little-endian float32, 24 bytes total
    --format int16x6     6 × int16; supply --accel-scale and --gyro-scale
                         (e.g. 16384 LSB/g → scale=1000/16.384 for mg output)

macOS note: grant Accessibility permission to Terminal.app once under
    System Settings → Privacy & Security → Accessibility
    Without it pynput keypresses are silently ignored by the game.
"""

from __future__ import annotations

import argparse
import asyncio
import queue
import signal
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 104          # Hz — must match training data
WINDOW_SAMPLES = 400       # ~3.85 s window fed to Predictor
HOP_SAMPLES = 52           # fire inference every 52 new samples (~0.5 s)
DEBOUNCE_S = 1.5           # seconds between same-class triggers
KEYPRESS_HOLD_S = 0.1      # how long to hold the key down (100 ms)

KEY_MAP: dict[str, str] = {
    "straight": "a",
    "hook":     "s",
    "upper":    "d",
}

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent / "punch_classifier_bundle"
CHECKPOINT = BUNDLE / "artifacts/limu_bert/checkpoints/limu_bert_production.pt"
THRESHOLD  = BUNDLE / "artifacts/limu_bert/metrics/production_info.json"

# ---------------------------------------------------------------------------
# Shared stop signal
# ---------------------------------------------------------------------------

_stop = threading.Event()


def _handle_sigint(*_) -> None:
    print("\n[controller] stopping…")
    _stop.set()


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------

class RingBuffer:
    """Thread-local sliding window over raw IMU samples.

    Not thread-safe — intended to live entirely inside one worker thread.
    """

    def __init__(self, maxlen: int = WINDOW_SAMPLES) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def push(self, sample: np.ndarray) -> None:
        """Append one (6,) float32 sample."""
        self._buf.append(sample)

    def ready(self) -> bool:
        return len(self._buf) == self._maxlen

    def get_window(self) -> np.ndarray:
        """Return a (WINDOW_SAMPLES, 6) float32 copy."""
        return np.array(self._buf, dtype=np.float32)


# ---------------------------------------------------------------------------
# Key press worker
# ---------------------------------------------------------------------------

def _keypress_worker(result_q: "queue.Queue[tuple[str, float]]") -> None:
    """Consume (label, confidence) pairs and inject OS-level keypresses."""
    from pynput.keyboard import Controller

    kb = Controller()
    last_label: Optional[str] = None
    last_time: float = 0.0

    print("[keypress] ready")

    while not _stop.is_set():
        try:
            label, conf = result_q.get(timeout=0.2)
        except queue.Empty:
            continue

        if label == "none":
            continue

        now = time.monotonic()
        # Debounce: skip if same class fired too recently
        if label == last_label and (now - last_time) < DEBOUNCE_S:
            continue

        key = KEY_MAP[label]
        kb.press(key)
        time.sleep(KEYPRESS_HOLD_S)
        kb.release(key)

        last_label = label
        last_time = now
        print(f"[keypress] {label:>8}  conf={conf:.2f}  → '{key}'")


# ---------------------------------------------------------------------------
# Inference worker
# ---------------------------------------------------------------------------

def _inference_worker(
    raw_q: "queue.Queue[np.ndarray]",
    result_q: "queue.Queue[tuple[str, float]]",
) -> None:
    """Drain raw_q into a ring buffer; fire Predictor every HOP_SAMPLES."""
    # Import here so the BLE/stub threads can start while the model loads.
    sys.path.insert(0, str(BUNDLE / "src"))
    from limu_bert import Predictor  # noqa: PLC0415

    print("[inference] loading model…")
    predictor = Predictor(
        checkpoint_path=CHECKPOINT,
        scaler_path=None,
        threshold_path=THRESHOLD,
    )
    print("[inference] model ready")

    ring = RingBuffer()
    new_samples = 0

    while not _stop.is_set():
        try:
            sample = raw_q.get(timeout=0.2)
        except queue.Empty:
            continue

        ring.push(sample)
        new_samples += 1

        if ring.ready() and new_samples >= HOP_SAMPLES:
            window = ring.get_window()          # (400, 6) float32, mg + dps
            label, conf = predictor.predict(window)
            result_q.put((label, conf))
            new_samples = 0


# ---------------------------------------------------------------------------
# BLE source
# ---------------------------------------------------------------------------

def _parse_float32x6(data: bytes) -> np.ndarray:
    """6 × little-endian float32 → (6,) array in mg and dps."""
    return np.array(struct.unpack_from("<6f", data), dtype=np.float32)


def _make_parse_int16x6(accel_scale: float, gyro_scale: float):
    """Return a parser for 6 × int16 with sensor-specific scale factors."""
    def _parse(data: bytes) -> np.ndarray:
        raw = struct.unpack_from("<6h", data)
        arr = np.array(raw, dtype=np.float32)
        arr[:3] *= accel_scale   # → mg
        arr[3:] *= gyro_scale    # → dps
        return arr
    return _parse


async def _ble_main(
    address: str,
    char_uuid: str,
    raw_q: "queue.Queue[np.ndarray]",
    parse_fn,
) -> None:
    from bleak import BleakClient

    def _on_notify(_handle, data: bytes) -> None:
        try:
            sample = parse_fn(data)
            raw_q.put_nowait(sample)
        except Exception as exc:
            print(f"[ble] parse error: {exc}")

    print(f"[ble] connecting to {address}…")
    async with BleakClient(address) as client:
        print(f"[ble] connected, subscribing to {char_uuid}")
        await client.start_notify(char_uuid, _on_notify)
        # Wait until stop is signalled, checking every 200 ms.
        while not _stop.is_set():
            await asyncio.sleep(0.2)
        await client.stop_notify(char_uuid)
    print("[ble] disconnected")


async def _ble_scan() -> None:
    """Print nearby BLE devices so the user can find their sensor address."""
    from bleak import BleakScanner
    print("[ble] scanning for 5 seconds…")
    devices = await BleakScanner.discover(timeout=5.0)
    if not devices:
        print("[ble] no devices found")
        return
    print(f"[ble] found {len(devices)} device(s):")
    for d in sorted(devices, key=lambda x: x.rssi if x.rssi else -999, reverse=True):
        print(f"  rssi={d.rssi:>4}  address={d.address}  name={d.name or '—'}")


def _ble_thread(
    address: str,
    char_uuid: str,
    raw_q: "queue.Queue[np.ndarray]",
    parse_fn,
) -> None:
    asyncio.run(_ble_main(address, char_uuid, raw_q, parse_fn))


# ---------------------------------------------------------------------------
# Stub source (global hotkeys 1/2/3 → synthetic label)
# ---------------------------------------------------------------------------

def _stub_thread(result_q: "queue.Queue[tuple[str, float]]") -> None:
    """Listen globally for 1/2/3 and push the corresponding label directly.

    Uses suppress=True so the game window never sees the '1'/'2'/'3' keys —
    only the translated 'a'/'s'/'d' keypresses arrive there.
    Keep the game window in focus; press 1, 2, or 3 to trigger a punch.
    Press Escape to quit.
    """
    from pynput import keyboard as _kb

    STUB_MAP = {
        _kb.KeyCode.from_char("1"): "straight",
        _kb.KeyCode.from_char("2"): "hook",
        _kb.KeyCode.from_char("3"): "upper",
    }

    print("[stub] ready — focus the game window, then press:")
    print("         1 → straight (A)   2 → hook (S)   3 → uppercut (D)   Esc → quit")

    def _on_press(key):
        if _stop.is_set():
            return False  # stop listener

        if key == _kb.Key.esc:
            _stop.set()
            return False  # stop listener

        label = STUB_MAP.get(key)
        if label:
            result_q.put((label, 1.0))

    with _kb.Listener(on_press=_on_press) as listener:
        listener.join()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="One Punch Man game controller")
    parser.add_argument("--mode", choices=["ble", "stub"], default="stub",
                        help="'ble' for live sensor, 'stub' for keyboard testing")
    parser.add_argument("--ble-address", default=None,
                        help="BLE MAC/UUID of the IMU sensor; omit to scan")
    parser.add_argument("--ble-uuid", default=None,
                        help="GATT characteristic UUID for IMU notifications")
    parser.add_argument("--format", choices=["float32x6", "int16x6"], default="float32x6",
                        help="BLE packet byte layout (default: float32x6)")
    parser.add_argument("--accel-scale", type=float, default=1.0,
                        help="int16x6 only: multiply raw int16 → mg (e.g. 1000/16.384)")
    parser.add_argument("--gyro-scale", type=float, default=1.0,
                        help="int16x6 only: multiply raw int16 → dps (e.g. 1000/131.0)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    # --- BLE scan shortcut ---------------------------------------------------
    if args.mode == "ble" and args.ble_address is None:
        asyncio.run(_ble_scan())
        return

    if args.mode == "ble" and args.ble_uuid is None:
        parser.error("--ble-uuid is required with --mode ble")

    # --- Shared queues -------------------------------------------------------
    raw_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=2048)
    result_q: queue.Queue[tuple[str, float]] = queue.Queue(maxsize=128)

    threads: list[threading.Thread] = []

    # --- Key press worker (always running) -----------------------------------
    kp = threading.Thread(target=_keypress_worker, args=(result_q,), daemon=True)
    kp.start()
    threads.append(kp)

    # --- Mode-specific threads -----------------------------------------------
    if args.mode == "stub":
        # Stub bypasses inference — pushes labels directly to result_q.
        src = threading.Thread(target=_stub_thread, args=(result_q,), daemon=True)
        src.start()
        threads.append(src)
        print("[controller] stub mode — game window must be in focus for keypresses")

    else:
        # BLE mode: sensor → raw_q → inference → result_q
        if args.format == "float32x6":
            parse_fn = _parse_float32x6
        else:
            parse_fn = _make_parse_int16x6(args.accel_scale, args.gyro_scale)

        inf = threading.Thread(
            target=_inference_worker, args=(raw_q, result_q), daemon=True
        )
        inf.start()
        threads.append(inf)

        ble = threading.Thread(
            target=_ble_thread,
            args=(args.ble_address, args.ble_uuid, raw_q, parse_fn),
            daemon=True,
        )
        ble.start()
        threads.append(ble)

        print("[controller] BLE mode — throw a punch, watch the game")

    # --- Wait for stop signal ------------------------------------------------
    try:
        while not _stop.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        _stop.set()

    for t in threads:
        t.join(timeout=2.0)

    print("[controller] done")


if __name__ == "__main__":
    main()
