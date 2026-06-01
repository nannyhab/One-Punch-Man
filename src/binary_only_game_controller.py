"""
Simple punch-to-game controller — pre-classified 3-byte USB protocol.

The sensor sends exactly 3 bytes per punch event:
    byte 0 : 0x2D (45)  — start marker
    byte 1 : ASCII char  — punch type: 'u' uppercut, 's' straight, 'h' hook
    byte 2 : 0x43 (67)  — end marker

Packets outside this frame are silently ignored.

Key mappings (One Punch Man):
    straight → A
    hook     → S
    uppercut → D

Usage
-----
Auto-detect port (use if only one USB device is connected):
    python src/punch_controller.py

Specify port explicitly:
    python src/punch_controller.py --port /dev/tty.usbmodem1101

Override baud rate (default 115200):
    python src/punch_controller.py --baud 9600

macOS note: grant Accessibility permission to Terminal.app once under
    System Settings → Privacy & Security → Accessibility
    Without it, pynput keypresses are silently ignored by the game.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

BAUD_DEFAULT = 115200
KEYPRESS_HOLD_S = 0.1      # how long to hold the key down
DEBOUNCE_S = 0.5           # minimum seconds between keypresses

START_BYTE = 45            # 0x2D
END_BYTE   = 67            # 0x43

PUNCH_MAP: dict[int, str] = {
    ord("u"): "upper",
    ord("s"): "straight",
    ord("h"): "hook",
}

KEY_MAP: dict[str, str] = {
    "straight": "a",
    "hook":     "s",
    "upper":    "d",
}

_stop = threading.Event()


def _handle_sigint(*_) -> None:
    print("\n[controller] stopping…")
    _stop.set()


# ---------------------------------------------------------------------------
# Port auto-detection
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
# Serial reader
# ---------------------------------------------------------------------------

def _serial_thread(port: str, baud: int, on_punch) -> None:
    """Read the serial stream byte-by-byte and call on_punch(label) per valid packet."""
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

    print(f"[serial] connected — waiting for punch packets on {port}")

    try:
        while not _stop.is_set():
            # Read one byte; returns b'' on timeout — just loop again.
            b0 = ser.read(1)
            if not b0:
                continue

            if b0[0] != START_BYTE:
                continue  # not a start marker — keep scanning

            # Read the next two bytes (punch type + end marker).
            rest = ser.read(2)
            if len(rest) < 2:
                continue  # incomplete packet

            punch_byte, end_byte = rest[0], rest[1]

            if end_byte != END_BYTE:
                print(f"[serial] bad end byte 0x{end_byte:02X} — dropping packet")
                continue

            label = PUNCH_MAP.get(punch_byte)
            if label is None:
                print(f"[serial] unknown punch byte 0x{punch_byte:02X} — dropping packet")
                continue

            on_punch(label)

    finally:
        ser.close()
        print("[serial] port closed")


# ---------------------------------------------------------------------------
# Keypress injector
# ---------------------------------------------------------------------------

def _make_punch_handler():
    """Return a thread-safe on_punch callback that injects OS keypresses."""
    from pynput.keyboard import Controller

    kb = Controller()
    last_label: list[str] = [None]
    last_time: list[float] = [0.0]
    lock = threading.Lock()

    def on_punch(label: str) -> None:
        now = time.monotonic()
        with lock:
            if label == last_label[0] and (now - last_time[0]) < DEBOUNCE_S:
                return  # debounce same punch
            last_label[0] = label
            last_time[0] = now

        key = KEY_MAP[label]
        print(f"[controller] {label:>8} → '{key}'")
        kb.press(key)
        time.sleep(KEYPRESS_HOLD_S)
        kb.release(key)

    return on_punch


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="One Punch Man — 3-byte USB punch controller")
    parser.add_argument("--port", default=None,
                        help="serial port, e.g. /dev/tty.usbmodem1101 or COM3; "
                             "auto-detected if omitted")
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
