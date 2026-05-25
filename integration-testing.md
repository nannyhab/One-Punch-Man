Step 1 — Find your sensor's BLE address
Run the controller with no address — it'll scan and list everything nearby:


.venv/bin/python src/game_controller.py --mode ble
You'll get output like:


[ble] scanning for 5 seconds…
[ble] found 4 device(s):
  rssi= -45  address=AA:BB:CC:DD:EE:FF  name=MyIMUSensor
  rssi= -72  address=11:22:33:44:55:66  name=iPhone
  ...
Note down the address of your IMU sensor.

Step 2 — Find the characteristic UUID
This is the GATT characteristic the sensor streams IMU data on. Two ways to get it:

Check your sensor's datasheet/app — it's usually documented
Or let bleak inspect it — run this one-liner with your address:

.venv/bin/python -c "
import asyncio
from bleak import BleakClient

async def inspect(addr):
    async with BleakClient(addr) as c:
        for s in c.services:
            print('SERVICE:', s)
            for ch in s.characteristics:
                print('  CHAR:', ch.uuid, ch.properties)

asyncio.run(inspect('AA:BB:CC:DD:EE:FF'))
"
Look for a characteristic with notify in its properties — that's the one streaming data.

Step 3 — Figure out the byte format
When the notification fires, what does the payload look like? Two common formats:

float32x6 (default): 24 bytes, 6 floats already in physical units (mg + dps)
int16x6: 12 bytes, 6 raw integers — needs scale factors to convert to mg/dps
Check your sensor's firmware docs. If unsure, you can print raw bytes first:


.venv/bin/python -c "
import asyncio
from bleak import BleakClient

async def sniff(addr, uuid):
    def cb(_, data): print(len(data), 'bytes:', data.hex())
    async with BleakClient(addr) as c:
        await c.start_notify(uuid, cb)
        await asyncio.sleep(5)

asyncio.run(sniff('AA:BB:CC:DD:EE:FF', 'YOUR-CHAR-UUID'))
"
Step 4 — Run live

.venv/bin/python src/game_controller.py --mode ble \
    --ble-address "AA:BB:CC:DD:EE:FF" \
    --ble-uuid "YOUR-CHAR-UUID"
You'll see:


[inference] loading model…
[inference] model ready
[ble] connecting to AA:BB:CC:DD:EE:FF…
[ble] connected, subscribing to YOUR-CHAR-UUID
Then throw a punch — after the first ~3.85s window fills you'll start seeing:


[keypress]  straight  conf=0.94  → 'a'