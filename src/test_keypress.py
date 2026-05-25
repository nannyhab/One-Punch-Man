"""
Minimal keypress test — run this, then check what app has focus.

Step 1: Open TextEdit (or any text field), give it focus, then run:
    .venv/bin/python src/test_keypress.py
  → You should see 'asd' typed into TextEdit after 3 seconds.
  → If nothing appears: pynput lacks Accessibility permission.

Step 2: Open IKEMEN GO, give it focus, then run:
    .venv/bin/python src/test_keypress.py
  → You should see the character throw three punches.
  → If nothing happens but Step 1 worked: the game blocks synthetic events.
"""

import time
from pynput.keyboard import Controller

kb = Controller()

print("You have 3 seconds to click the target window...")
time.sleep(3)

for key in ["a", "s", "d"]:
    print(f"pressing '{key}'")
    kb.press(key)
    time.sleep(0.1)
    kb.release(key)
    time.sleep(0.5)

print("done")
