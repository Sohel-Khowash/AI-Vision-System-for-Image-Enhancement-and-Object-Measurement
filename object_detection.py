import cv2
from ultralytics import YOLO

# 1. Load the YOLO model
# 'yolov8n.pt' is the "Nano" model (smallest & fastest). 
# It will automatically download the weight file the first time you run this.
model = YOLO('yolov8n.pt')

# 2. Setup the video source
# Use '0' for your webcam, or replace with a file path like 'video.mp4'
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()
    
    if success:
        # 3. Run detection
        # stream=True is efficient for video
        results = model(frame, stream=True)

        # 4. Process results and draw boxes
        for result in results:
            # The 'plot()' method automatically draws boxes and labels on the frame
            annotated_frame = result.plot()
            
            # Display the resulting frame
            cv2.imshow("YOLO Inference", annotated_frame)

        # Press 'q' on your keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()