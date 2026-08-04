import cv2
import time
import numpy as np
from ultralytics import YOLO
from brping import Ping1D

# ==============================
# 🔧 USER CONFIGURATION
# ==============================

COM_PORT = "COM7"
BAUD_RATE = 115200

# Replace with your calibration values
FOCAL_LENGTH_X = 1215   # example value (pixels)
FOCAL_LENGTH_Y = 1215   # example value (pixels)

# ==============================
# 🔌 Initialize Sonar
# ==============================

ping = Ping1D()

try:
    ping.connect_serial(COM_PORT, BAUD_RATE)
    print("Sonar port opened.")
except Exception as e:
    print("Error opening sonar:", e)
    exit()

if ping.initialize() is False:
    print("Failed to initialize Ping.")
    exit()

print("Ping Connected!")

# ==============================
# 🤖 Load YOLO Model
# ==============================

model = YOLO("yolov8n.pt")

# ==============================
# 🎥 Start Camera
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

print("Starting YOLO + Sonar size estimation...")

# ==============================
# 🔁 Main Loop
# ==============================

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # ------------------------------
    # 📡 Get Sonar Distance
    # ------------------------------
    distance_mm = None
    try:
        data = ping.get_distance()
        if data:
            distance_mm = data["distance"]/100.0
    except:
        pass

    # ------------------------------
    # 🧠 Run YOLO Detection
    # ------------------------------
    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            pixel_width = x2 - x1
            pixel_height = y2 - y1

            label_text = "No Distance"

            if distance_mm is not None and FOCAL_LENGTH_X != 0:
                real_width_mm = (pixel_width * distance_mm) / FOCAL_LENGTH_X
                real_height_mm = (pixel_height * distance_mm) / FOCAL_LENGTH_Y

                label_text = f"W:{real_width_mm:.1f}mm H:{real_height_mm:.1f}mm"

                print(f"Distance: {distance_mm} mm | "
                      f"Width: {real_width_mm:.1f} mm | "
                      f"Height: {real_height_mm:.1f} mm")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Draw size label
            cv2.putText(frame, label_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2)

    # Show sonar distance
    if distance_mm:
        cv2.putText(frame,
                    f"Sonar Distance: {distance_mm} mm",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2)

    cv2.imshow("YOLO + Sonar Size Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.05)

# ==============================
# 🧹 Cleanup
# ==============================

cap.release()
cv2.destroyAllWindows()