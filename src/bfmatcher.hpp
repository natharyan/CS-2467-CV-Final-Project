#ifndef BFMACTHER_H
#define BFMACTHER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

//compute the hamming distances between the feature descriptors
int computeHammingDistance(const cv::Mat& desc1, const cv::Mat& desc2, int idx1, int idx2);

//get the keypoint matches
vector<pair<cv::KeyPoint, cv::KeyPoint>> getMatches_Keypoints(const cv::Mat& descriptors1, const cv::Mat& descriptors2, const vector<cv::KeyPoint> keypoints1, const vector<cv::KeyPoint> keypoints2, const float ratio);

void plotMatches(const cv::Mat& img1, const cv::Mat& img2, const vector<pair<cv::KeyPoint, cv::KeyPoint>>& matches);

#endif // BFMACTHER_H