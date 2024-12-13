#ifndef BUNDLE_H
#define BUNDLE_H

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

using namespace std;

using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;

struct Observation {
    int point_id;    // Index of 3D point
    int camera_id;   // Index of camera
    Eigen::Vector2d uv;     // Observed 2D point
};

void bundleAdjustment(
    std::vector<Eigen::Vector3d>& points3D,        // 3D points
    std::vector<Eigen::Matrix3d>& rotations,       // Camera rotation matrices
    std::vector<Eigen::Vector3d>& translations,    // Camera translation vectors
    const std::vector<Observation>& observations, // Observations
    int iterations = 100, double learning_rate = 1e-4
); 

#endif // BUNDLE_H
