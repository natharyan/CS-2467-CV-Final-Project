# Week 1: Feature Detection, Matching, and Initial Pair Selection

## Feature Detection and Extraction:
We’ll first implement SIFT or ORB from scratch or using OpenCV functions to detect and extract keypoints and descriptors from the images.(done - left to integrate)(Manya)
- Output: Keypoints and descriptors for each image (Manya).
- Deliverables: Visualization of the detected key points on sample images to confirm quality and density (Manya).

## Feature Matching:
We’ll then employ a matching algorithm like Brute-Force Matcher (BFMatcher)(done)(Aryan)
 or FLANN-based matcher for descriptor matching between image pairs. (Aryan)
 We’ll perform filter matching using Lowe’s ratio test to remove ambiguous matches.(done)(Aryan)
- Output: Matched keypoints between image pairs.(Manya)
- Deliverables: Plotted matches between image pairs to ensure correctness.(done)(Manya)

## Initial Pair Selection:
We’ll use a geometric criterion (e.g., maximum number of inliers from epipolar constraint) to select the best initial image pair(done).(Aryan)
We’ll implement the 8-point algorithm or use RANSAC to estimate the fundamental matrix and remove outliers.(Manya)
- Output: Fundamental matrix and inlier matches.
- Deliverables: Visualization of the epipolar lines and inliers.(done)(Aryan)

#########################################################################################

# Week 2: Camera Pose Estimation, Triangulation, and Point Cloud Generation

## Camera Pose Estimation:
We’ll decompose the essential matrix (computed from the fundamental matrix) to obtain the rotation and translation between the initial image pair. (Aryan)
We’ll choose the correct pose by ensuring positive depth for the reconstructed points.(Aryan)
- Output: Rotation and translation matrices for the initial pair.
- Deliverables: We’ll document the selection process for the correct pose.

## Triangulation:
(Manya)
We’ll then implement a triangulation method (e.g., Direct Linear Transform) to reconstruct 3D points from the matched inliers. We’ll ensure depth consistency and remove points that fall behind the cameras.(Aryan)
- Output: 3D point cloud for the initial image pair.(Manya)
- Deliverables: Visualization of the initial 3D point cloud using tools like matplotlib or Open3D.(Manya)

#########################################################################################

# Week 3: Incremental Reconstruction(Incremental SFM), Dense Reconstruction, and Final Presentation

## Incremental Image Addition:
Implement an image addition loop where new images are added based on overlapping 2D-3D correspondences (Manya).
Estimate the new camera pose using PnP (Perspective-n-Point) with RANSAC. Triangulate new points and integrate them into the existing 3D model (Manya).
- Output: Incremental 3D point cloud with more views added (Manya).
- Deliverables: Visualize step-by-step growth of the 3D model (Manya).

## Bundle Adjustment(on each increment perform bundle adjustment to refine the camera parameters and 3D points):
We’ll implement or integrate a basic bundle adjustment using libraries like Ceres Solver or g2o to refine camera parameters and 3D points (Manya).
- Output: Optimized camera poses and 3D structure (Manya).
- Deliverables: We’ll show the before-and-after comparison of the point cloud (Manya).
