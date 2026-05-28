"""
Real-time punch-to-game controller.

Reads IMU data from a BLE sensor, a USB/serial sensor, or a keyboard stub
for testing, classifies punches with LIMU-BERT, and injects keypresses into
IKEMEN GO.

Key mappings (One Punch Man):
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

Serial/USB mode (when Bluetooth is unavailable):
    python src/game_controller.py --mode serial
    python src/game_controller.py --mode serial --serial-port /dev/tty.usbmodem1101

    Omit --serial-port to auto-detect (auto-selects if one port found, prompts if many).

    The sensor sends ASCII bursts framed by IMU_STREAM_BEGIN / IMU_STREAM_END.
    Everything outside those delimiters is ignored.
    Each line inside the burst must be: [idx] ax ay az gx gy gz
    Values must already be in mg (accel) and dps (gyro) — no scale factors needed.
    Default baud rate is 115200; override with --baud.

    To list available serial ports run:
        python -m serial.tools.list_ports

BLE byte format flags (BLE mode only):
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
    # load the model from the other files

    print("[inference] loading model…")
    predictor = Predictor(
        checkpoint_path=CHECKPOINT,
        scaler_path=None,
        threshold_path=THRESHOLD,
    ) # load model with th specified weights at this checkpoint
    print("[inference] model ready")

    ring = RingBuffer()
    new_samples = 0

    while not _stop.is_set():
        try:
            sample = raw_q.get(timeout=0.2) # get samples from the raw_q and move to the ringbuffer
        except queue.Empty:
            continue

        ring.push(sample)
        new_samples += 1

        if ring.ready() and new_samples >= HOP_SAMPLES:
            window = ring.get_window()          # (400, 6) float32, mg + dps
            label, conf = predictor.predict(window) # predict requires a minimum number of samples to make a prediction,
            # so thats why we hve a ring buffer, once full drain it and predict
            result_q.put((label, conf))
            new_samples = 0


# ---------------------------------------------------------------------------
# Burst inference worker (serial mode — one prediction per burst)
# ---------------------------------------------------------------------------

def _burst_inference_worker(
    burst_q: "queue.Queue[np.ndarray]",
    result_q: "queue.Queue[tuple[str, float]]",
) -> None:
    """Read complete (N,6) bursts from burst_q and classify each one directly.

    Bypasses the ring buffer — the sensor already windowed the punch,
    so we call predictor.predict() once per burst.
    """
    sys.path.insert(0, str(BUNDLE / "src"))
    from limu_bert import Predictor  # noqa: PLC0415

    print("[inference] loading model…")
    predictor = Predictor(
        checkpoint_path=CHECKPOINT,
        scaler_path=None,
        threshold_path=THRESHOLD,
    )
    predictor.threshold = 0.1
    print("[inference] model ready")

    while not _stop.is_set():
        try:
            burst = burst_q.get(timeout=0.2)  # (N, 6) float32, mg + dps
        except queue.Empty:
            continue

        label, conf = predictor.predict(burst)
        print(f"[inference] prediction: {label}  conf={conf:.2f}")
        result_q.put((label, conf))


# ---------------------------------------------------------------------------
# BLE source
# ---------------------------------------------------------------------------

def _parse_float32x6(data: bytes) -> np.ndarray:
    """6 × little-endian float32 → (6,) array in mg and dps."""
    return np.array(struct.unpack_from("<6f", data), dtype=np.float32) 
    # unpack 6 little-endian float32 values from the byte data and return them as a numpy array of type float32


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
    # takes in BLE packets, parses them and adds them to the raw_q
    from bleak import BleakClient # bluetooth client library for connecting to the BLE sensor 

    def _on_notify(_handle, data: bytes) -> None:
        try:
            sample = parse_fn(data) # parse into a np array of 6 values
            raw_q.put_nowait(sample)# addt eh sample to the q
        except Exception as exc:
            print(f"[ble] parse error: {exc}")

    print(f"[ble] connecting to {address}…")
    async with BleakClient(address) as client:
        print(f"[ble] connected, subscribing to {char_uuid}")
        await client.start_notify(char_uuid, _on_notify) # call on notify when a new packet is in this cahnnel
        # Wait until stop is signalled, checking every 200 ms.
        while not _stop.is_set(): # checks if the stop signal has been set every 200 ms
            await asyncio.sleep(0.2)
        await client.stop_notify(char_uuid) # when stop is signalled, disconnect 
    print("[ble] disconnected")


async def _ble_scan() -> None:
    # async to allow the CPU to do other stuff like loaidng the model, or setting up other threads while waiting for the BLE scan to complete; since BLE scanning can take several seconds, we don't want to block the main thread during this time
    # this scans nearby for Bluetooth devices advertising their presence, and prints their RSSI (signal strength), MAC address, and name (if available) so the user can identify their sensor and get its address for the --ble-address argument
    """Print nearby BLE devices so the user can find their sensor address."""
    from bleak import BleakScanner# library for bluetooth communication, used here to scan for nearby BLE devices and to connect to the specified device and subscribe to its notifications
    print("[ble] scanning for 5 seconds…")
    devices = await BleakScanner.discover(timeout=5.0)
    if not devices:
        print("[ble] no devices found")
        return
    print(f"[ble] found {len(devices)} device(s):")
    for d in sorted(devices, key=lambda x: x.rssi if x.rssi else -999, reverse=True):
        print(f"  rssi={d.rssi:>4}  address={d.address}  name={d.name or '—'}") # print devices found in a sorted order


def _ble_thread(
    address: str, # sensors address
    char_uuid: str, #channel 
    raw_q: "queue.Queue[np.ndarray]", # queue
    parse_fn,
) -> None:
    asyncio.run(_ble_main(address, char_uuid, raw_q, parse_fn)) # actually run the ble main fn.


# ---------------------------------------------------------------------------
# Serial / USB source
# ---------------------------------------------------------------------------

def _serial_autodetect() -> str:
    """Return a serial port to use, auto-selecting if unambiguous.

    - 0 ports found → exit with an error.
    - 1 port found  → use it automatically.
    - 2+ ports found → print a numbered list and prompt the user to pick.
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        print("[serial] pyserial not installed — run: pip install pyserial")
        sys.exit(1)

    ports = list(list_ports.comports())

    if not ports:
        print("[serial] no serial ports found — is the IMU plugged in?")
        sys.exit(1)

    if len(ports) == 1:
        print(f"[serial] auto-selected port: {ports[0].device}")
        return ports[0].device

    print("[serial] multiple ports found, pick one:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}  —  {p.description}")

    while True:
        try:
            choice = int(input("enter number: "))
            if 0 <= choice < len(ports):
                return ports[choice].device
        except (ValueError, EOFError):
            pass
        print(f"  enter a number between 0 and {len(ports) - 1}")


def _serial_thread(
    port: str,
    baud: int,
    burst_q: "queue.Queue[np.ndarray]",
) -> None:
    """Read ASCII burst packets from a USB serial port and push complete bursts to burst_q.

    Expected format:
        IMU_STREAM_BEGIN
        [000] ax ay az gx gy gz
        [001] ax ay az gx gy gz
        ...
        IMU_STREAM_END

    Values are already in mg (accel) and dps (gyro) — no scale factors needed.
    Each complete burst is pushed as a single (N, 6) float32 array.
    """
    try:
        import serial  # pyserial
    except ImportError:
        print("[serial] pyserial not installed — run: pip install pyserial")
        _stop.set()
        return

    print(f"[serial] opening {port} at {baud} baud")
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=0.5)
    except serial.SerialException as exc:
        print(f"[serial] failed to open port: {exc}")
        _stop.set()
        return

    print(f"[serial] connected — waiting for IMU_STREAM_BEGIN on {port}")
    in_burst = False
    current_burst: list[np.ndarray] = []

    try:
        while not _stop.is_set():
            raw_line = ser.readline()
            if not raw_line:
                continue

            try:
                line = raw_line.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            if line == "IMU_STREAM_BEGIN":
                in_burst = True
                current_burst = []
                continue

            if line == "IMU_STREAM_END":
                in_burst = False
                if current_burst:
                    burst = np.stack(current_burst, axis=0)  # (N, 6)
                    print(f"[serial] burst received — {len(current_burst)} samples")
                    try:
                        burst_q.put_nowait(burst)
                    except queue.Full:
                        print("[serial] burst_q full — dropping burst (inference too slow?)")
                current_burst = []
                continue

            # Ignore everything outside a BEGIN/END block.
            if not in_burst:
                continue

            # Expected: "[000] ax ay az gx gy gz" — values already in mg and dps
            try:
                values_str = line.split("]", 1)[1].strip()
                vals = [float(v) for v in values_str.split()]
                if len(vals) != 6:
                    print(f"[serial] unexpected column count: {line!r}")
                    continue
                current_burst.append(np.array(vals, dtype=np.float32))
            except Exception as exc:
                print(f"[serial] parse error on {line!r}: {exc}")
    finally:
        ser.close()
        print("[serial] port closed")


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
    parser.add_argument("--mode", choices=["ble", "serial", "stub"], default="stub",
                        help="'ble' for BLE sensor, 'serial' for USB/serial sensor, 'stub' for keyboard testing")
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
    parser.add_argument("--serial-port", default=None,
                        help="serial mode: port path, e.g. /dev/tty.usbmodem1101 or COM3")
    parser.add_argument("--baud", type=int, default=115200,
                        help="serial mode: baud rate (default 115200)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)#what to do when ctrl+c is pressed -- clean up and exit

    # --- BLE scan shortcut ---------------------------------------------------
    if args.mode == "ble" and args.ble_address is None:
        asyncio.run(_ble_scan()) # look for nearby BLE devices and print their addresses so the user can find theirs
        # one you see yours, run it again with --ble-address "XX:XX:XX:XX:XX:XX" to connect to it and start the game controller
        return

    if args.mode == "ble" and args.ble_uuid is None:
        parser.error("--ble-uuid is required with --mode ble") # BLE mode requires a characteristic UUID to subscribe which channel to read IMU data

    if args.mode == "serial" and args.serial_port is None:
        args.serial_port = _serial_autodetect()

    # --- Shared queues -------------------------------------------------------
    result_q: queue.Queue[tuple[str, float]] = queue.Queue(maxsize=128) # (label, confidence) pairs consumed by keypress worker
    threads: list[threading.Thread] = []

    # --- Key press worker (always running) -----------------------------------
    kp = threading.Thread(target=_keypress_worker, args=(result_q,), daemon=True)
    kp.start()
    threads.append(kp)

    # --- Mode-specific threads -----------------------------------------------
    if args.mode == "stub":
        # Stub bypasses inference — pushes labels directly to result_q.
        # starts the keyboard listener thread that listens for 1/2/3 keypresses and pushes the corresponding label directly to result_q, 
        # which then triggers the keypress worker to inject the 'a'/'s'/'d' keypresses into the game. 
        # This allows testing the keypress injection and game response without needing a BLE sensor or the inference model.
        src = threading.Thread(target=_stub_thread, args=(result_q,), daemon=True)
        src.start()
        threads.append(src)
        print("[controller] stub mode — game window must be in focus for keypresses")

    elif args.mode == "serial":
        # Serial burst mode: sensor → burst_q → burst_inference → result_q
        # Bypasses the ring buffer — predictor.predict() is called once per burst.
        burst_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)

        inf = threading.Thread(
            target=_burst_inference_worker, args=(burst_q, result_q), daemon=True
        )
        inf.start()
        threads.append(inf)

        src = threading.Thread(
            target=_serial_thread,
            args=(args.serial_port, args.baud, burst_q),
            daemon=True,
        )
        src.start()
        threads.append(src)
        print(f"[controller] serial mode — reading from {args.serial_port}, throw a punch, watch the game")

    else:
        # BLE mode: sensor → raw_q → inference → result_q
        raw_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=2048)

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
        # loop to chcek every 100ms if the stop signal has been set, 
        # if not, keep waiting; this allows the main thread to remain responsive to Ctrl+C and to allow the worker threads to run concurrently without blocking the main thread
        # if yes, break the loop, join the threads and exit
        while not _stop.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        _stop.set()

    for t in threads: # after stop is signalled, join all threads with a timeout to allow them to clean up and exit gracefully
        t.join(timeout=2.0)

    print("[controller] done")


if __name__ == "__main__":
    main()
