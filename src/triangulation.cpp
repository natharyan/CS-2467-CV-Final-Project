#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <climits> 
#include <cmath>
#include "epipolar.hpp"
using namespace std;

// need to get intrinsic matrix K, R, and t from calibration + epipolar 

/*
from notes.md: 
inputs for triangulation: K (3x3), points in image 1 and corresponding points in image 2, R (3x3), T (3x1) 
outputs: vector of 3d points - homogenous/euclidean coords 
deliverables: visualization using o3d (.ply file) 
*/

// reference: https://staff.fnwi.uva.nl/r.vandenboomgaard/ComputerVision/LectureNotes/CV/StereoVision/triangulation.html

cv::Mat computeProjection(const cv::Mat& K, const cv::Mat& R, const cv::Mat& T) {
    cv::Mat RT = cv::Mat::zeros(3, 4, R.type()); // Create a 3x4 matrix
    R.copyTo(RT(cv::Rect(0, 0, 3, 3)));         // Copy R into the first 3 columns
    T.copyTo(RT(cv::Rect(3, 0, 1, 3)));         // Copy T into the last column
    cv::Mat P = K * RT;                         // Compute the projection matrix
    return P;
}

std::vector<cv::Point3f> triangulation(const cv::Mat& P1, const cv::Mat& P2, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2) {
    std::vector<cv::Point3f> points3D; // Store 3D points

    for (size_t i = 0; i < points1.size(); ++i) {
        double x1 = points1[i].x, y1 = points1[i].y; // First image point
        double x2 = points2[i].x, y2 = points2[i].y; // Second image point

        // Build the A matrix
        cv::Mat A(4, 4, CV_64F); 
        A.row(0) = x1 * P1.row(2) - P1.row(0); // x1 * p3^T - p1^T
        A.row(1) = y1 * P1.row(2) - P1.row(1); // y1 * p3^T - p2^T
        A.row(2) = x2 * P2.row(2) - P2.row(0); // x2 * p3'^T - p1'^T
        A.row(3) = y2 * P2.row(2) - P2.row(1); // y2 * p3'^T - p2'^T

        // Solve the linear system using SVD
        cv::Mat U, S, Vt;
        cv::SVD::compute(A, S, U, Vt); 
        cv::Mat X = Vt.row(3).t(); // Last row of V^T (transposed)

        // Convert from homogeneous to Euclidean coordinates
        X /= X.at<double>(3); 
        points3D.emplace_back(X.at<double>(0), X.at<double>(1), X.at<double>(2));
    }

    return points3D;
}
