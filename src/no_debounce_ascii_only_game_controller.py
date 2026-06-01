"""
Debug controller — parses ASCII serial output from the on-device ML classifier.

Reads the verbose punch-pass log printed by the firmware and fires a keypress
for the highest-confidence punch type found in each pass.

Expected format (one punch pass block):
    --- punch pass (ring 125, 3 x 50) ---
      win[0] off=0 punch=0%
      win[1] off=37 punch=85%
      win[2] off=75 punch=100%
      win[2] off=75 type=upper (99%) h=0.00 s=0.00 u=0.99
    --- end ---

Only lines containing "type=" are classified detections. If a pass has more
than one, the one with the highest percentage wins. Passes with no type lines
are ignored (no punch detected).

Key mappings (One Punch Man):
    straight → A
    hook     → S
    upper    → D

Usage
-----
Auto-detect port:
    python src/debug_controller.py

Specify port:
    python src/debug_controller.py --port /dev/tty.usbmodem1101

Override baud rate (default 115200):
    python src/debug_controller.py --baud 9600
"""

from __future__ import annotations

import argparse
import re
import signal
import sys
import threading
import time

BAUD_DEFAULT = 115200
KEYPRESS_HOLD_S = 0.1
DEBOUNCE_S = 0.05

KEY_MAP: dict[str, str] = {
    "straight": "a",
    "hook":     "s",
    "upper":    "d",
}

# Matches: "  win[2] off=75 type=upper (99%) h=0.00 s=0.00 u=0.99"
_TYPE_RE = re.compile(r"type=(upper|straight|hook)\s+\((\d+)%\)")

_stop = threading.Event()


def _handle_sigint(*_) -> None:
    print("\n[controller] stopping…")
    _stop.set()


# ---------------------------------------------------------------------------
# Port auto-detection (same as punch_controller)
# ---------------------------------------------------------------------------

def _autodetect_port() -> str:
    try:
        from serial.tools import list_ports
    except ImportError:
        print("[serial] pyserial not installed — run: pip install pyserial")
        sys.exit(1)

    ports = list(list_ports.comports())

    if not ports:
        print("[serial] no serial ports found — is the device plugged in?")
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


# ---------------------------------------------------------------------------
# Keypress injector
# ---------------------------------------------------------------------------

def _make_punch_handler():
    from pynput.keyboard import Controller

    kb = Controller()
    last_label: list[str] = [None]
    last_time: list[float] = [0.0]
    lock = threading.Lock()

    def on_punch(label: str, confidence: int, t_detected: float) -> None:
        now = time.monotonic()
        with lock:
            if label == last_label[0] and (now - last_time[0]) < DEBOUNCE_S:
                return
            last_label[0] = label
            last_time[0] = now

        t_pre_press = time.monotonic()
        latency_ms = (t_pre_press - t_detected) * 1000

        key = KEY_MAP[label]
        print(f"[controller] {label} ({confidence}%) → pressing '{key}'", flush=True)
        kb.press(key)
        time.sleep(KEYPRESS_HOLD_S)
        kb.release(key)

    return on_punch


# ---------------------------------------------------------------------------
# Serial reader + punch-pass parser
# ---------------------------------------------------------------------------

def _serial_thread(port: str, baud: int, on_punch) -> None:
    try:
        import serial
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

    print(f"[serial] connected — watching for punch type lines on {port}")

    try:
        while not _stop.is_set():
            try:
                raw = ser.readline()
            except Exception as exc:
                print(f"[serial] device disconnected: {exc}")
                _stop.set()
                break

            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()

            m = _TYPE_RE.search(line)
            if m:
                t_detected = time.monotonic()
                label, pct = m.group(1), int(m.group(2))
                print(f"[serial]   detected {label} {pct}%")
                on_punch(label, pct, t_detected)

    finally:
        ser.close()
        print("[serial] port closed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Debug punch controller — ASCII serial log parser")
    parser.add_argument("--port", default=None,
                        help="serial port, e.g. /dev/tty.usbmodem1101; auto-detected if omitted")
    parser.add_argument("--baud", type=int, default=BAUD_DEFAULT,
                        help=f"baud rate (default {BAUD_DEFAULT})")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    port = args.port or _autodetect_port()
    on_punch = _make_punch_handler()

    reader = threading.Thread(
        target=_serial_thread,
        args=(port, args.baud, on_punch),
        daemon=True,
    )
    reader.start()

    print("[controller] running — focus the game window and throw a punch (Ctrl-C to quit)")

    try:
        while not _stop.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        _stop.set()

    reader.join(timeout=2.0)
    print("[controller] done")


if __name__ == "__main__":
    main()
