
# Game Controller Implementation Plan
## Goal
Connect the LIMU-BERT punch classifier to IKEMEN GO by building src/game_controller.py — a single Python script that reads live IMU data over BLE, classifies punches, and injects the corresponding keypress into the game.

## Key Mappings (from game-setup.md)
ML Class	Key	Punch
- straight	a	Straight punch
- hook	    s	Hook
- upper	    d	Uppercut
none	—	No action (low confidence)

## Files Involved
- src/game_controller.py	New — the entire pipeline
- punch_classifier_bundle/src/limu_bert.py	Existing — Predictor class, used as-is
- punch_classifier_bundle/artifacts/limu_bert/checkpoints/limu_bert_production.pt	Existing — production checkpoint
- punch_classifier_bundle/artifacts/limu_bert/metrics/production_info.json	Existing — confidence threshold
_No existing files are modified._


## Architecture: Single Process, Three Threads
Thread 1: BLE (bleak async event loop)
  └─► raw_queue  (thread-safe queue.Queue)

Thread 2: Inference worker  (reads raw_queue)
  └─► RingBuffer (deque, maxlen=400)
  └─► every 52 new samples → Predictor.predict()
  └─► result_queue (thread-safe queue.Queue)

Thread 3: Key press worker  (reads result_queue)
  └─► debounce check
  └─► pynput → press 'a' / 's' / 'd'

Main thread: startup, graceful shutdown on Ctrl+C


### Why three threads:
BLE uses an async event loop (asyncio) — it must own its thread
Inference is CPU-bound and blocking — keeping it off the BLE thread prevents dropped BLE packets
Key press is near-instant but logically separate — isolates the debounce state
Communication between threads: queue.Queue (Python stdlib, thread-safe, no locks to manage manually). Queues are the only shared state between threads.

## Module-by-module breakdown
1. Constants block (top of file)
SAMPLE_RATE     = 104        # Hz — must match preprocessing
WINDOW_SAMPLES  = 400        # ~3.85s — what Predictor expects
HOP_SAMPLES     = 52         # 0.5s — how often inference fires
DEBOUNCE_S      = 1.5        # seconds — min gap between same-punch triggers
CHECKPOINT      = Path(...)  # absolute path to limu_bert_production.pt
THRESHOLD_PATH  = Path(...)  # absolute path to production_info.json
KEY_MAP         = {"straight": "a", "hook": "s", "upper": "d"}
DEBOUNCE_S = 1.5 prevents the same punch class from firing twice in rapid succession (e.g. the model confidently predicts "hook" on 3 overlapping windows of the same physical punch — you only want one keypress).


2. RingBuffer
A thin wrapper around collections.deque(maxlen=WINDOW_SAMPLES):
class RingBuffer:
    def push(self, sample: np.ndarray)  # shape (6,)
    def ready(self) -> bool             # True when len == WINDOW_SAMPLES
    def get_window(self) -> np.ndarray  # returns copy, shape (400, 6)
The deque is not shared between threads — it lives entirely inside the inference worker thread. No locking needed.

3. BLE thread
Uses bleak.BleakClient in an asyncio event loop running on a dedicated thread (threading.Thread(target=asyncio.run, args=(ble_main(),))).

What it does:

Scan for the sensor by name or address (provided as CLI arg --ble-address)
Subscribe to the IMU notification characteristic (UUID also as CLI arg --ble-uuid)
In the notification callback: parse the raw bytes into a (6,) float32 array [ax, ay, az, gx, gy, gz] in mg and dps, then raw_queue.put(sample)
Byte parsing: This is sensor-specific. The script will expect a --format arg with two options initially:

float32x6 — 6 consecutive little-endian float32s (24 bytes) — most common
int16x6 — 6 int16s with scale factors --accel-scale and --gyro-scale (common in Nordic/ST sensors)
If a new sensor format is needed later, only this parsing function changes.


4. Inference worker thread
Runs a while True loop:

1. sample = raw_queue.get()          # blocks until BLE pushes something
2. ring.push(sample)
3. new_sample_count += 1
4. if ring.ready() and new_sample_count >= HOP_SAMPLES:
       window = ring.get_window()    # (400, 6) float32, mg + dps
       label, conf = predictor.predict(window)
       new_sample_count = 0
       result_queue.put((label, conf))
The Predictor from limu_bert.py handles all internal transforms (resample 104→20 Hz, unit conversion mg→g / dps→rad/s, windowing to 40 samples). You feed it raw sensor data exactly matching what was recorded.

_Important: Predictor is instantiated once at startup (model load ~0.5s) and reused — not reloaded per inference call._

5. Key press worker thread
Runs a while True loop:

1. label, conf = result_queue.get()    # blocks
2. if label == "none": continue        # below threshold — skip
3. now = time.monotonic()
4. if label == last_label and (now - last_trigger_time) < DEBOUNCE_S: continue
5. key = KEY_MAP[label]
6. keyboard.press(key); time.sleep(0.05); keyboard.release(key)
7. last_label = label; last_trigger_time = now
8. print(f"[{label}] → '{key}'")      # console feedback
keyboard is a pynput.keyboard.Controller() instance created once at startup.

The time.sleep(0.05) on the keypress holds the key for 50ms — long enough for IKEMEN GO to register it as a keydown event, not so long it double-triggers.


6. Stub mode (--mode stub)
For testing without the sensor. Replaces the BLE thread with a stdin reader:
> 1   →  straight → press 'a'
> 2   →  hook     → press 's'
> 3   →  uppercut → press 'd'
Stub mode bypasses the ring buffer and inference entirely — it puts a hardcoded (label, 1.0) directly into result_queue. This lets you verify:

The game window receives keypresses correctly
The debounce logic works
The key mapping is correct
...all before touching the sensor.

7. Graceful shutdown
signal.signal(SIGINT, handler) sets a threading.Event named stop_event. All worker loops check stop_event.is_set() at the top of each iteration. The BLE async loop uses asyncio.get_event_loop().stop(). Main thread joins all threads with a 2s timeout.

8. CLI interface

python src/game_controller.py --mode ble \
    --ble-address "XX:XX:XX:XX:XX:XX" \
    --ble-uuid "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" \
    --format float32x6

python src/game_controller.py --mode stub
All BLE params have defaults of None — if omitted in ble mode, the script scans and lists nearby BLE devices so you can find the right address.

Latency budget (realistic end-to-end)
Stage	Time
BLE notification delivery	~10ms
Ring buffer fill (first window)	~3.85s — unavoidable
Sliding window re-fire (after first)	~0.5s per hop
LIMU-BERT inference (55K params, CPU)	<10ms
Key event delivery to game	<5ms
Total lag after punch completes	~0–500ms
The 3.85s fill time only applies to the very first punch after launch. After that, because the ring buffer is always full and sliding, each new inference fires 0.5s after the previous one — meaning a punch that completes will be classified within ~0.5s of its last sample arriving.


New dependencies
Package	Purpose	Install
bleak	Async BLE on macOS	pip install bleak
pynput	OS-level key injection	pip install pynput
Both need to be added to requirements.txt.

**macOS one-time setup: Grant Accessibility permission to Terminal.app (System Settings → Privacy & Security → Accessibility → add Terminal). Without this, pynput keypresses are silently swallowed.**

What is explicitly NOT in scope
No CSV file writes during inference (that was your concern — confirmed not happening)
No new ML training or model changes
No modification to limu_bert.py or preprocess.py
No multi-player support (Player 1 / a/s/d only)
No GUI — console output only

Testing sequence
Run --mode stub, launch game, verify 1/2/3 triggers animations
Scan for BLE sensor with --mode ble (no address) to confirm discovery
Run full --mode ble pipeline, throw punches, verify classifications print to console
Verify game receives keypresses with game window in focus
That's the full plan. Let me know if anything needs adjusting before I write the code.

## to run it:
src/game_controller.py — 280 lines, three threads, zero file I/O.

To test right now (no sensor needed):


### 1. Launch IKEMEN GO and make sure the game window is in focus
### 2. In a second terminal:
.venv/bin/python src/game_controller.py --mode stub
### Then type: 1 (straight), 2 (hook), 3 (upper), q (quit)
When your BLE sensor is ready:


### Scan to find your sensor's address:
.venv/bin/python src/game_controller.py --mode ble

### Then run live:
.venv/bin/python src/game_controller.py --mode ble \
    --ble-address "XX:XX:XX:XX:XX:XX" \
    --ble-uuid "YOUR-CHAR-UUID"
One macOS step before testing keypresses: System Settings → Privacy & Security → Accessibility → add Terminal.app. Without it pynput is silently blocked.

The venv lives at .venv/ in the project root — activate with source .venv/bin/activate if you want to run other scripts from the same environment.

