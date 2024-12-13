#ifndef ORB_H
#define ORB_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

using namespace std;

cv::Mat createBaseImage(const std::string& imagePath);

std::vector<cv::KeyPoint> FAST9(const cv::Mat& image, int threshold);

double orientationAssignment(const cv::Mat& image, const cv::KeyPoint& keypoint, int patchSize = 7);

double harrisResponse(const cv::Mat& gray, const cv::KeyPoint& keypoint, int blockSize = 3, double k = 0.04) ;

cv::Mat rBRIEF(const cv::Mat& image, vector<cv::KeyPoint>& keypoints, int patchSize = 31) ;

struct KeypointWithResponse {
    cv::KeyPoint point;
    double harrisResponse;
};

#endif // ORB_HPP