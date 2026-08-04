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

# --- 2. Get Real Object Dimensions ---
def get_real_object_width():
    try:
        print("\n--- OBJECT SETUP ---")
        print("To calculate distance, the code needs to know the real size of the object.")
        width_cm = float(input("Enter the REAL WIDTH of the object (in cm): "))
        if width_cm <= 0:
            print("Please enter a positive value.")
            return get_real_object_width()
        return width_cm
    except ValueError:
        print("Invalid input. Please enter a number.")
        return get_real_object_width()

# --- 3. Get Bounding Box (Manual) ---
def get_bounding_box(frame):
    print("\n--- DRAW A BOX ---")
    print("A new window will open. Click and drag to draw a box around the object.")
    print("Press 'ENTER' to confirm, 'c' to cancel.")
    
    # Manual selection of the object
    bbox = cv2.selectROI("Draw Bounding Box", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Draw Bounding Box")
    
    (x, y, w, h) = bbox
    if w > 0 and h > 0:
        print(f"Box selected at (x,y,w,h): {bbox}")
        return bbox
    else:
        print("No box drawn.")
        return None

# --- 4. Main Loop ---

# Get the known real-world width ONCE.
REAL_WIDTH_CM = get_real_object_width()
print(f"Tracking object with real width: {REAL_WIDTH_CM} cm")

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

print("\n--- Starting Distance Detection ---")
print("Press 's' to select the object and calculate distance.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break
    
    # Instructions on the live feed
    cv2.putText(frame, "Press 's' to measure distance", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Live Feed', frame)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    if key == ord('s'):
        # 1. Get the bounding box from user input
        bbox = get_bounding_box(frame)
        
        if bbox is not None:
            (x, y, pixel_width, pixel_height) = bbox
            
            # --- APPLY THE DISTANCE FORMULA ---
            # Distance = (Real Width * Focal Length) / Pixel Width
            # We use FOCAL_LENGTH_X because we are using the object's WIDTH
            if pixel_width > 0:
                distance_cm = (REAL_WIDTH_CM * FOCAL_LENGTH_X) / pixel_width

                print("\n--- DISTANCE RESULT ---")
                print(f"  Real Width: {REAL_WIDTH_CM} cm")
                print(f"  Pixel Width: {pixel_width} px")
                print(f"  CALCULATED DISTANCE: {distance_cm:.2f} cm")
                print("----------------------------")
                
                # Display result on a still frame
                result_frame = frame.copy()
                cv2.rectangle(result_frame, (x, y), (x + pixel_width, y + pixel_height), (0, 255, 0), 2)
                
                label = f"Dist: {distance_cm:.1f} cm"
                # Draw a background rectangle for the text for better readability
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(result_frame, (x, y - 30), (x + w_text, y), (0, 255, 0), -1)
                
                cv2.putText(result_frame, label, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                cv2.imshow("Distance Result", result_frame)
                print("Press any key in the 'Result' window to continue...")
                cv2.waitKey(0)
                cv2.destroyWindow("Distance Result")
            else:
                print("Error: Pixel width is 0. Cannot calculate distance.")

# --- 5. Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("\nLoop stopped.")