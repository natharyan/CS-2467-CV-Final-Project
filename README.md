# Part 1: Feature Detection, Matching, and Initial Pair Selection

## Feature Detection and Extraction:
We first implemented ORB from scratch to detect and extract keypoints and descriptors from the images.*(done)(Manya)*
Output: Keypoints and descriptors for each image. *(done)*
Deliverables: Visualization of the detected key points on sample images to confirm quality and density. *(done)*

## Feature Matching:
We then employed a matching algorithm like Brute-Force Matcher (BFMatcher) *(done)(Aryan)* for descriptor matching between image pairs. *(Aryan)*
We performed filter matching using Lowe’s ratio test to remove ambiguous matches. *(done)(Aryan)*
Output: Matched keypoints between image pairs. *(Manya)*
Deliverables: Plotted matches between image pairs to ensure correctness. *(done)(Manya)*

## Initial Pair Selection:
We used a geometric criterion (e.g., maximum number of inliers from epipolar constraint) to select the best initial image pair *(done).(Aryan)*
We implemented the 8-point algorithm and RANSAC to estimate the fundamental matrix and remove outliers.*(done)(Manya)*
Output: Fundamental matrix and inlier matches.*(done)(Aryan)*
Deliverables: Visualization of the epipolar lines and inliers. *(done)(Aryan)*

#########################################################################################

# Part 2: Camera Pose Estimation, Triangulation, and Point Cloud Generation

## Camera Pose Estimation: *(Aryan)*
We then decomposed the essential matrix (computed from the fundamental matrix) to obtain the rotation and translation between the initial image pair.*(done) (Aryan)*
We chose the correct pose by ensuring positive depth for the reconstructed points.*(done) (Aryan)*(NOTE: cheirality test passed only after normalizing and undistorting the images)*
Output: Rotation and translation matrices for the initial pair. *(done) (Aryan)*

## Triangulation: *(done) (Manya) and (Aryan)*
We implemented a triangulation method (using SVD) to reconstruct 3D points from the matched inliers. We ensured somewhat of depth consistency and removed points that fall behind the cameras. *(done) (Manya) and (Aryan)*
Output: 3D point cloud for the initial image pair. *(done) (Manya) and (Aryan)*
Deliverables: Visualization of the initial 3D point cloud using tools like matplotlib or Open3D. *(done) (Aryan)*
Triangulation: *(done) (Manya) and (Aryan)*

#########################################################################################

# Part 3: Incremental Reconstruction(Incremental SFM), Dense Reconstruction, and Final Presentation

## Incremental Image Addition: *(Manya) and (Aryan)*
We implemented an image addition loop where new images are added based on overlapping 2D-3D correspondences. *(done) (Manya)*
We estimated the new camera pose using PnP (Perspective-n-Point) with RANSAC. Then we triangulated new points and integrated them into the existing 3D model. *(done) (Manya)*
Output: Incremental 3D point cloud with more views added. *(done)*
Deliverables: Visualize step-by-step growth of the 3D model.

## Bundle Adjustment(on each incremwent perform bundle adjustment to refine the camera parameters and 3D points):
We integrated a basic bundle adjustment using libraries to refine camera parameters and 3D points.*(done) (Manya)*
Output: Optimized camera poses and 3D structure.*(done) (Manya)*

## Dense Reconstruction 
We implemented a dense reconstruction of our 3d points for various datasets, namely the book statue, water canon, and maingate statue. *(done) (Aryan)*

