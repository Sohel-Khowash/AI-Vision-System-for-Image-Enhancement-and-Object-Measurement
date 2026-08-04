import numpy as np
import cv2

# --- Configuration ---
CALIBRATION_FILE = 'calibration_data.npz'
# Use the same resolution you calibrated with
RESOLUTION = (1280, 720) 
# 0 is usually your built-in webcam, 1 is your external webcam
CAMERA_INDEX = 0  # <--- IMPORTANT: Change this to your Zebronics index
# --- End Configuration ---

# --- 1. Load Calibration Data ---
try:
    with np.load(CALIBRATION_FILE) as data:
        FOCAL_LENGTH_X = data['fx']
        FOCAL_LENGTH_Y = data['fy']
    print("Calibration data loaded successfully.")
    print(f"  Focal Length (fx): {FOCAL_LENGTH_X:.2f} px")
    print(f"  Focal Length (fy): {FOCAL_LENGTH_Y:.2f} px")
except FileNotFoundError:
    print(f"Error: Calibration file '{CALIBRATION_FILE}' not found.")
    print("Please run the 'calibrate_camera.py' script first.")
    exit()

# --- 2. Get Known Distance (Placeholder) ---
def get_known_distance():
    try:
        dist_cm = float(input("\nEnter the distance from camera to object (in cm): "))
        if dist_cm <= 0:
            print("Please enter a positive value.")
            return get_known_distance()
        return dist_cm
    except ValueError:
        print("Invalid input. Please enter a number.")
        return get_known_distance()

# --- 3. Get Bounding Box (Manual) ---
def get_bounding_box(frame):
    print("\n--- DRAW A BOX ---")
    print("A new window will open. Click and drag to draw a box around the object.")
    print("Press 'ENTER' to confirm, 'c' to cancel.")
    
    bbox = cv2.selectROI("Draw Bounding Box", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Draw Bounding Box")
    
    (x, y, w, h) = bbox
    if w > 0 and h > 0:
        print(f"Box selected at (x,y,w,h): {bbox}")
        return bbox
    else:
        print("No box drawn.")
        return None

# --- 4. Main Test Loop ---

# Get the known distance from the user ONCE.
KNOWN_DISTANCE_CM = get_known_distance()
print(f"Using fixed distance: {KNOWN_DISTANCE_CM} cm")

# Initialize your camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Cannot open camera at index {CAMERA_INDEX}.")
    exit()

# Set the resolution to match your calibration
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

# Check if resolution was set
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Webcam opened. Running at: {width}x{height}")

print("\n--- Starting Measurement Test ---")
print("Press 's' to select a new object to measure.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break
    
    cv2.putText(frame, "Press 's' to measure an object", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Live Feed', frame)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    if key == ord('s'):
        bbox = get_bounding_box(frame)
        
        if bbox is not None:
            (x, y, pixel_width, pixel_height) = bbox
            
            # --- APPLY THE FORMULA ---
            real_width_cm = (pixel_width * KNOWN_DISTANCE_CM) / FOCAL_LENGTH_X
            real_height_cm = (pixel_height * KNOWN_DISTANCE_CM) / FOCAL_LENGTH_Y

            print("\n--- MEASUREMENT RESULTS ---")
            print(f"  Known Distance: {KNOWN_DISTANCE_CM} cm")
            print(f"  Pixel Dims (w, h): {pixel_width}, {pixel_height}")
            print(f"  REAL Dims (W, H): {real_width_cm:.2f} cm, {real_height_cm:.2f} cm")
            print("----------------------------")
            
            result_frame = frame.copy()
            cv2.rectangle(result_frame, (x, y), (x + pixel_width, y + pixel_height), (0, 255, 0), 2)
            cv2.putText(result_frame, f"Width: {real_width_cm:.2f} cm", (x, y - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Height: {real_height_cm:.2f} cm", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Measurement Result", result_frame)
            print("Press any key in the 'Result' window to continue...")
            cv2.waitKey(0)
            cv2.destroyWindow("Measurement Result")

# --- 5. Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("\nLoop stopped.")