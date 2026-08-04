import cv2
import os
import time

# --- Configuration ---
IMAGES_DIR = 'calibration_images'
# 0 is usually your built-in webcam, 1 is your external webcam
CAMERA_INDEX = 0  # <--- CHANGE THIS TO 1 (or 2)
RESOLUTION = (1280, 720) # 720p is a good, fast resolution
# --- End Configuration ---

# Create the folder if it doesn't exist
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Cannot open camera at index {CAMERA_INDEX}")
    exit()

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

# Check if the resolution was set correctly
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera opened. Resolution: {width}x{height}")

count = 0

print("\n--- Starting Capture ---")
print(f"  Saving images to '{IMAGES_DIR}' folder.")
print("  Press 's' to save a photo.")
print("  Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame.")
        break
    
    # Display the feed
    cv2.imshow('Webcam Feed - Press "s" to save', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    elif key == ord('s'):
        # Save the frame
        img_name = os.path.join(IMAGES_DIR, f'calib_{count:02d}.png')
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        count += 1
        
        # Show a "Saved!" message on the frame for 1 second
        saved_frame = frame.copy()
        cv2.putText(saved_frame, "SAVED!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam Feed - Press "s" to save', saved_frame)
        cv2.waitKey(1000) # Show for 1 second

cap.release()
cv2.destroyAllWindows()
print(f"Capture complete. {count} images saved.")