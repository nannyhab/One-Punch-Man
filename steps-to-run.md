# How to Run the One Punch Man Game Controller

## Step 1 — One-time macOS permission
Go to **System Settings → Privacy & Security → Accessibility** and make sure **Terminal.app** is checked.
Without this, keypresses are silently ignored by the game.

---

## Step 2 — Set up the virtual environment (one-time)
```bash
cd /Users/mercy/Desktop/CS396ML/One-Punch-Man
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Step 3 — Launch the game
```bash
Ikemen_GO-dev-macos/Ikemen_GO.command
```
In the game: **VS Mode → Quick Match → Opponent 1**. Leave the game window in focus.

---

## Step 4 — Plug in the IMU via USB-C, then start the controller (new terminal)
```bash
cd /Users/mercy/Desktop/CS396ML/One-Punch-Man
source .venv/bin/activate
python src/game_controller.py --mode serial
```

The port is auto-detected. Wait until you see all three ready messages:
```
[serial] auto-selected port: /dev/tty.usbmodem1101
[inference] model ready
[keypress] ready
```

---

## Step 5 — Switch focus to the game window and throw a punch

You should see in the controller terminal:
```
[serial] burst received — 104 samples
[inference] prediction: straight  conf=0.84
[keypress]  straight  conf=0.84  → 'a'
```
And the character attacks in the game.

**Ctrl+C** to stop the controller.

---
if using the punch_controller.py

python src/punch_controller.py



---

## Troubleshooting

| Problem | Fix |
|---|---|
| Port not found | Check IMU is plugged in, re-run `python -m serial.tools.list_ports` |
| Multiple ports listed | Run with `--serial-port /dev/tty.usbmodemXXXX` to specify manually |
| Predictions show `none` | Model confidence below threshold — try throwing a sharper punch |
| Predictions print but game ignores them | Game window not in focus, or Accessibility permission missing |
| `pyserial not installed` | `pip install pyserial` |


