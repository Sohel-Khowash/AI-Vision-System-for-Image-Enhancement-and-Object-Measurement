import cv2
import numpy as np

# =========================
# SETTINGS
# =========================
CHESSBOARD_SIZE = (9, 6)     # 9x6 inner corners
SQUARE_SIZE = 0.02           # 20mm squares -> 0.02 meters
REQUIRED_IMAGES = 20
CAMERA_INDEX = 0             # Change if camera not detected

# =========================
# PREPARE OBJECT POINTS
# =========================
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

# =========================
# START CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)

# Force HD resolution (optional but recommended)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Camera not detected!")
    exit()

print("\n====== CAMERA CALIBRATION MODE ======")
print("Move chessboard around.")
print("Press 's' to capture when corners detected.")
print("Press 'q' to quit early.\n")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_corners, corners = cv2.findChessboardCorners(
        gray, CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret_corners:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners2, ret_corners)

    cv2.imshow("Calibration", frame)

    key = cv2.waitKey(1)

    if key == ord('s') and ret_corners:
        objpoints.append(objp)
        imgpoints.append(corners2)
        count += 1
        print(f"Captured {count}/{REQUIRED_IMAGES}")

    if key == ord('q') or count >= REQUIRED_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()

if len(objpoints) < 5:
    print("❌ Not enough captures for calibration.")
    exit()

print("\n🔄 Calibrating...")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("\n===== RESULTS =====")
print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)

# =========================
# REPROJECTION ERROR
# =========================
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("\nMean Reprojection Error:", mean_error / len(objpoints))

# =========================
# SAVE CALIBRATION
# =========================
np.savez(
    "calibration_data.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

print("✅ Calibration saved as calibration_data.npz")

# =========================
# LIVE UNDISTORT TEST
# =========================
print("\nStarting live undistortion test...")
print("Press 'q' to exit.")

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (w, h),
        1,
        (w, h)
    )

    dst = cv2.undistort(
        frame,
        camera_matrix,
        dist_coeffs,
        None,
        newcameramtx
    )

    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Done.")