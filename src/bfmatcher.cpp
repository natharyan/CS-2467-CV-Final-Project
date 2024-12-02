#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include "bfmatcher.hpp"

using namespace std;

//Hamming distance on the uchar descriptors
int computeHammingDistance(const cv::Mat& desc1, const cv::Mat& desc2, int idx1, int idx2) {
    int distance = 0;
    for (int i = 0; i < desc1.cols; ++i) {
        uchar val1 = desc1.at<uchar>(idx1, i);
        uchar val2 = desc2.at<uchar>(idx2, i);
        distance += std::bitset<8>(val1 ^ val2).count(); //bitwise xor and get the number of mismatched bits
    }
    return distance;
}

//get the keypoint matches
vector<pair<cv::KeyPoint, cv::KeyPoint>> getMatches_Keypoints(const cv::Mat& descriptors1, const cv::Mat& descriptors2, const vector<cv::KeyPoint> keypoints1, const vector<cv::KeyPoint> keypoints2, const float ratio = 0.75) {
    vector<pair<cv::KeyPoint, cv::KeyPoint>> matches;
    vector<bool> matched_descriptors2(descriptors2.rows, false); //track matched descriptors from descriptors2

    for (int i = 0; i < descriptors1.rows; ++i) {
        int best_idx = -1, second_best_idx = -1;
        int best_dist = numeric_limits<int>::max();
        int second_best_dist = numeric_limits<int>::max();

        //find the best and second-best matches
        for (int j = 0; j < descriptors2.rows; ++j) {
            if (matched_descriptors2[j]) continue; //skip already matched descriptors

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

        //apply Lowe's ratio test
        if (best_idx != -1 && best_dist < ratio * second_best_dist) {
            matches.push_back({keypoints1[i], keypoints2[best_idx]});
            matched_descriptors2[best_idx] = true;
        }
    }

    return matches;
}


void plotMatches(const cv::Mat& img1, const cv::Mat& img2, const vector<cv::KeyPoint>& keypoints1, const vector<cv::KeyPoint>& keypoints2, const vector<pair<cv::KeyPoint, cv::KeyPoint>>& matches) {
    cv::Mat img1_color, img2_color;
    cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);

    cv::Mat img_matches(img1.rows, img1.cols + img2.cols, CV_8UC3);
    img1_color.copyTo(img_matches(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2_color.copyTo(img_matches(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    for (const auto& match : matches) {

        cv::Point pt1 = cv::Point(match.first.pt.x, match.first.pt.y);;
        cv::Point pt2 = cv::Point(match.second.pt.x + img1.cols, match.second.pt.y);

        cv::line(img_matches, pt1, pt2, cv::Scalar(0, 255, 0), 1);
        cv::circle(img_matches, pt1, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(img_matches, pt2, 3, cv::Scalar(255, 0, 0), -1);
    }

    cv::imshow("Matched Keypoints", img_matches);
    cv::waitKey(1);
    cv::destroyWindow("Matched Keypoints");
}