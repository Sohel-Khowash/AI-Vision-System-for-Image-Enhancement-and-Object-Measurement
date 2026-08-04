from brping import Ping1D
import time

ping = Ping1D()

try:
    ping.connect_serial("COM7", 115200)
    print("Port opened successfully.")
except Exception as e:
    print("Error opening port:", e)
    exit()

if ping.initialize() is False:
    print("Failed to initialize Ping")
    exit()

print("Ping Connected!")

while True:
    try:
        data = ping.get_distance()
        if data:
            print("Distance (mm):", data["distance"])
        time.sleep(0.1)   # small delay prevents overflow
    except Exception as e:
        print("Temporary error:", e)
        time.sleep(1)