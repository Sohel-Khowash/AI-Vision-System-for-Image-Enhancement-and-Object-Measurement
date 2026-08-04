import numpy as np

data = np.load("calibration_data.npz")

print("Available keys:", data.files)

if "camera_matrix" in data.files:
    camera_matrix = data["camera_matrix"]
    print("Camera Matrix:\n", camera_matrix)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    print("Focal Length fx:", fx)
    print("Focal Length fy:", fy)

elif "fx" in data.files and "fy" in data.files:
    fx = data["fx"]
    fy = data["fy"]

    print("Focal Length fx:", fx)
    print("Focal Length fy:", fy)

else:
    print("Could not find focal length values.")