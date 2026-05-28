Here's the exact sequence:

## Step 1 — Accessibility permission (one-time, macOS)
Go to System Settings → Privacy & Security → Accessibility and make sure Terminal.app is checked. Without this, keypresses are silently ignored by the game.

## Step 2 — Launch the game

Ikemen_GO-dev-macos/Ikemen_GO.command
In the game: VS Mode → Quick Match → Opponent 1. Leave the game window in focus.

## Step 3 — Plug in your IMU sensor via USB
Then find its port:

python -m serial.tools.list_ports
It'll show something like /dev/tty.usbmodem1101. Note that down.

Step 4 — Open a second terminal, start the controller


cd /Users/mercy/Desktop/CS396ML/One-Punch-Man

python src/game_controller.py --mode serial --serial-port /dev/tty.usbmodem1101
Add --format int16x6 --accel-scale <N> --gyro-scale <N> if your sensor sends int16 instead of float32.

Step 5 — Go back to the game window and throw punches

You should see the controller terminal print lines like:


[serial] connected — reading IMU samples from /dev/tty.usbmodem1101
[inference] model ready
[keypress]  straight  conf=0.87  → 'a'
Ctrl+C in the controller terminal to stop cleanly.

If something goes wrong:

failed to open port → wrong port name, try python -m serial.tools.list_ports again
pyserial not installed → pip install pyserial
keypresses not reaching the game → game window isn't in focus, or Accessibility permission is missing
