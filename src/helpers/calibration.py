import numpy as np
import cv2 as cv
import os

# Define termination criteria for corner refinement
termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Generate a grid of 3D object points for the chessboard pattern
grid_points = np.zeros((10 * 7, 3), np.float32)
grid_points[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)

real_world_points = []  # 3D points in the real world
image_points = []       # 2D points in image plane

image_dir = '../../dataset/Camera Calibration(Samsung S20 fe)'
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv.imread(image_path)
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Detect chessboard corners in the grayscale image
    found, corners = cv.findChessboardCorners(grayscale_image, (10, 7), None)

    if found:
        real_world_points.append(grid_points)
        refined_corners = cv.cornerSubPix(grayscale_image, corners, (11, 11), (-1, -1), termination_criteria)
        image_points.append(refined_corners)

        cv.drawChessboardCorners(image, (10, 7), refined_corners, found)
        cv.imshow('Detected Corners', image)
        cv.waitKey(500)
    else:
        print(f"Corners not detected in {image_file}")

cv.destroyAllWindows()

calibration_success, camera_matrix, distortion_coeffs, rotation_vecs, translation_vecs = cv.calibrateCamera(
    real_world_points, image_points, grayscale_image.shape[::-1], None, None
)

print("intrinsics matrix: ", camera_matrix)
print("distortion coefficients: ", distortion_coeffs)

np.savez('calibration.npz', mtx=camera_matrix, dist=distortion_coeffs)

# store these in a text file also
with open('calibration.txt', 'w') as f:
    f.write(np.array2string(camera_matrix) + '\n')
    f.write(np.array2string(distortion_coeffs) + '\n')

# distortion_correction_example = 'datasets/Camera Calibration(Samsung S20 fe)/20241116_210603.jpg'
# img = cv.imread(distortion_correction_example)
# h, w = img.shape[:2]
# optimal_matrix, valid_roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))
# corrected_image = cv.undistort(img, camera_matrix, distortion_coeffs, None, optimal_matrix)
# x, y, w, h = valid_roi
# corrected_image = corrected_image[y:y+h, x:x+w]
# cv.imwrite('calibration_result.png', corrected_image)
