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

cv::Mat computeProjection(const cv::Mat& K, const cv::Mat& R, const cv::Mat& T){
    cv::Mat RT = cv::Mat::zeros(3, 4, R.type()); // creating a matrix with syntax: rows, cols, mimicking the data type of R 
    // putting R in the first three cols and T in the last 
    R.copyTo(RT(cv::Rect(0, 0, 3, 3))); 
    T.copyTo(RT(cv::Rect(3,0,1,3)));

    cv::Mat P = K * RT; 
    
    return P; 
}

// get inlier points from epipolar.cpp - assuming they're point1 and point2 for now 

// compute SVD 
/*from notes.md:
*/

cv::Mat triangulation(const cv::Mat& P, const vector<cv::Point2f>& point1, const vector<cv::Point2f>& point2){
    cv::Mat A(2,4, CV_64F); // 2 x 4 matrix of 64-bit floating point 
    cv::Mat X(4,1, CV_64F); // 4x1 matrix for X 
    vector<cv::Point3f> points3D; // storing the 3d points 

    for (size_t i = 0; i < point1.size(); ++i){
        cout << "size: " << point1.size() << endl;
        double x1 = point1[i].x; // x coordinate in the first image
        cout << "this is x " << x1 << " this was x" << endl;
        double y1 = point1[i].y; // y coordinate in the first image
        cout << "this is y " << y1 << " this was y" << endl;
        double x2 = point2[i].x; // x coordinate in the second image
        cout << "this is x2 " << x2 << " this was x2" << endl;
        double y2 = point2[i].y; // y coordinate in the second image
        cout << "this is y2 " << y2 << " this was y2" << endl;

        // svd 

        // homogenous -> euclidean 

    }

    // return threeDPoints 
}

// just running samples to check 
int main() {
    // Sample intrinsic matrix K
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        1000, 0, 320,
        0, 1000, 240,
        0, 0, 1);

    // Sample rotation matrix R (identity matrix for simplicity)
    cv::Mat R = (cv::Mat_<double>(3, 3) << 
        1, 0, 0,
        0, 1, 0,
        0, 0, 1);

    // Sample translation vector T
    cv::Mat T = (cv::Mat_<double>(3, 1) << 
        1,
        2,
        3);

    // Compute the projection matrix
    cv::Mat P = computeProjection(K, R, T);

    // Print the projection matrix
    cout << "Projection Matrix: " << endl << P << endl;

    // Sample points that we could have in image 1 and image 2
    vector<cv::Point2f> point1 = {cv::Point2f(150, 150), cv::Point2f(300, 300)};
    vector<cv::Point2f> point2 = {cv::Point2f(160, 160), cv::Point2f(310, 310)};

    // Triangulate points 
    cv::Mat threeDPoints = triangulation(P, point1, point2);

    // Print the 3D points
    cout << "3D Points: " << endl << threeDPoints << endl;

    return 0;
}