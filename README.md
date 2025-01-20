# 3D Reconstruction Pipeline

This project implements a 3D reconstruction pipeline starting from feature detection and matching, through camera pose estimation, to dense point cloud generation. The pipeline is modular and can be extended or customized for various datasets and reconstruction requirements.

---

## Table of Contents
1. [Steps in the Pipeline](#steps-in-the-pipeline)
2. [How to Run](#how-to-run)
3. [Modifying the Code](#modifying-the-code)
5. [License](#license)
6. [Contribution](#contribution)

---

## Steps in the Pipeline

### Part 1: Feature Detection, Matching, and Initial Pair Selection

1. **Feature Detection and Extraction**:  
   ORB (Oriented FAST and Rotated BRIEF) is used to detect and extract keypoints and descriptors from images.  
   - **Output**: Keypoints and descriptors for each image.  

2. **Feature Matching**:  
   Brute-Force Matcher (BFMatcher) matches descriptors between image pairs, and Lowe’s ratio test filters ambiguous matches.  
   - **Output**: Matched keypoints between image pairs.  

3. **Initial Pair Selection**:  
   The initial image pair is selected based on a geometric criterion, such as the maximum number of inliers from the epipolar constraint. The fundamental matrix is estimated using the 8-point algorithm with RANSAC.  
   - **Output**: Fundamental matrix and inlier matches.  

### Part 2: Camera Pose Estimation, Triangulation, and Point Cloud Generation

1. **Camera Pose Estimation**:  
   The essential matrix is decomposed to retrieve rotation and translation between the initial image pair. A cheirality test ensures correct pose selection.  
   - **Output**: Rotation and translation matrices.  

2. **Triangulation**:  
   A triangulation algorithm reconstructs 3D points from the matched inliers. Points with inconsistent depth or those behind the cameras are removed.  
   - **Output**: Initial 3D point cloud.  

### Part 3: Incremental Reconstruction and Dense Reconstruction

1. **Incremental Image Addition**:  
   New images are added iteratively based on overlapping 2D-3D correspondences. Camera poses are estimated using PnP (Perspective-n-Point) with RANSAC, and new points are triangulated and integrated.  
   - **Output**: Incremental 3D point cloud.  

2. **Bundle Adjustment**:  
   A bundle adjustment is performed after each increment to refine camera parameters and 3D points.  
   - **Output**: Optimized camera poses and 3D structure.  

3. **Dense Reconstruction**:  
   Dense reconstruction refines the 3D model for datasets such as `book statue`, `water cannon`, and `main gate statue`.  
   - **Output**: Densified 3D model.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repository-name.git
   cd repository-name
2. Running the MakeFile
   Make sure you're in the right repository on your terminal and run the MakeFile 
   ```bash
   make
3. Run the executable 
   ```bash 
   bin/app
---

## Modifying the Code

Adjusting Parameters
- Feature extraction parameters (e.g., ORB settings) can be modified in `orb.cpp`.
- Matching thresholds (e.g., Lowe's ratio) are adjustable in `bfmatcher.cpp`.
- Own image datasets can be added in `datasets/<dataset_name>/`. 
- The current calibration data can be replaced in `main.cpp`.

Visualizing
- Visualizations are generated and saved automatically as `.ply` files. For different visualization, modify the code in `main.cpp`. 

--- 

## License

This project is licensed under the MIT License.

---

## Contribution

We welcome contributions to improve and extend this project.
