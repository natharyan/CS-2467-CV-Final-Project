#ifndef EPIPOLAR
#define EPIPOLAR

#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <fstream>

using namespace std;

// get the epipolar lines using the fundamental matrix and the matched keypoints from the first image in the pair selected
cv::Mat epipolar_line(cv::Mat fundamental_matrix, cv::Point2d keypoint1);

// get inliers based on the epipolar constraint
vector<bool> getInliers(cv::Mat fundamental_matrix, vector<cv::Point2f> &points1, vector<cv::Point2f> &points2, double epsilon = 0.01);

// returns true for points that are within epsilon error from zero for the epipolar constraint
bool epipolar_contraint(cv::Mat fundamental_matrix, cv::Point2d keypoint1, cv::Point2d keypoint2, double epsilon = 0.01);

// finds the initial image pair from a directory of images
pair<string,string> initial_image_pair(vector<string> images);

// plot the inliers and the epipolar lines for all keypoints from the first image that have been matched with keypoints from the second image using the bfmatcher implementation
void plotEpipolarLinesAndInliers(cv::Mat &img1, cv::Mat &img2, vector<cv::Point2f> &points1, vector<cv::Point2f> &points2, cv::Mat &fundamental_matrix, vector<bool> &inliers);

#endif //EPIPOLAR