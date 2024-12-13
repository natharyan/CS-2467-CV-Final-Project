#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <bitset>
#include <algorithm>
#include "bfmatcher.hpp"

using namespace std;
using namespace cv;

// Compute the Hamming distance between two descriptors
int computeHammingDistance(const cv::Mat& desc1, const cv::Mat& desc2, int idx1, int idx2) {
    int distance = 0;
    for (int i = 0; i < desc1.cols; ++i) {
        uchar val1 = desc1.at<uchar>(idx1, i);
        uchar val2 = desc2.at<uchar>(idx2, i);
        distance += std::bitset<8>(val1 ^ val2).count(); // XOR and count differing bits
    }
    return distance;
}

// Find matches between two sets of descriptors and keypoints
vector<pair<cv::KeyPoint, cv::KeyPoint>> getMatches_Keypoints(const cv::Mat& descriptors1, const cv::Mat& descriptors2, const vector<cv::KeyPoint> keypoints1, const vector<cv::KeyPoint> keypoints2, const float ratio = 0.75) {
    vector<pair<cv::KeyPoint, cv::KeyPoint>> matches;
    vector<bool> matched_descriptors2(descriptors2.rows, false); // Track matched descriptors

    for (int i = 0; i < descriptors1.rows; ++i) {
        int best_idx = -1, second_best_idx = -1;
        int best_dist = numeric_limits<int>::max();
        int second_best_dist = numeric_limits<int>::max();

        // Find the best and second-best matches
        for (int j = 0; j < descriptors2.rows; ++j) {
            int dist = computeHammingDistance(descriptors1, descriptors2, i, j);
            if (dist < best_dist) {
                second_best_dist = best_dist;
                second_best_idx = best_idx;
                best_dist = dist;
                best_idx = j;
            } else if (dist < second_best_dist) {
                second_best_dist = dist;
                second_best_idx = j;
            }
        }

        // Apply Lowe's ratio test
        if (best_idx != -1 && best_dist < ratio * second_best_dist && !matched_descriptors2[best_idx]) {
            matches.push_back({keypoints1[i], keypoints2[best_idx]});
            matched_descriptors2[best_idx] = true;
        }
    }

    return matches;
}

// Plot matches between two images
void plotMatches(const cv::Mat& img1, const cv::Mat& img2, const vector<pair<cv::KeyPoint, cv::KeyPoint>>& matches) {
    cv::Mat img1_color, img2_color;
    // get the keypoints from the matches
    vector<cv::KeyPoint> keypoints1, keypoints2;
    for (const auto& match : matches) {
        keypoints1.push_back(match.first);
        keypoints2.push_back(match.second);
    }
    cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);

    cv::Mat img_matches(img1.rows, img1.cols + img2.cols, CV_8UC3);
    img1_color.copyTo(img_matches(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2_color.copyTo(img_matches(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    // Draw matches
    for (const auto& match : matches) {
        const KeyPoint& kp1 = match.first;
        const KeyPoint& kp2 = match.second;

        cv::Point2f pt1 = kp1.pt;
        cv::Point2f pt2 = kp2.pt + cv::Point2f((float)img1.cols, 0); // Shift second point for concatenated image

        cv::line(img_matches, pt1, pt2, cv::Scalar(0, 255, 0), 1);
        cv::circle(img_matches, pt1, 3, cv::Scalar(0, 255, 255), -1);
        cv::circle(img_matches, pt2, 3, cv::Scalar(0, 255, 255), -1);
    }

    cv::imshow("Matches", img_matches);
    cv::waitKey(1);
}
