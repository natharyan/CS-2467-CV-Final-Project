#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

// Convert integer descriptor to binary
vector<string> desctoBinary(const cv::Mat& descRow) {
    vector<string> bin_desc;
    for (int i = 0; i < descRow.cols; i++) {
        int val = descRow.at<uchar>(0, i); // Access each element in the row
        string bin_val = "";
        // Convert int to binary
        while (val > 0) {
            bin_val += to_string(val % 2);
            val /= 2;
        }
        // Pad with zeros to ensure 8-bit length
        while (bin_val.size() < 8) {
            bin_val += '0';
        }
        // Reverse the string
        reverse(bin_val.begin(), bin_val.end());
        bin_desc.push_back(bin_val);
    }
    return bin_desc;
}

// Compute Hamming distance for vector containing multiple binary strings
int computeHammingDistance(vector<string> descBin1, vector<string> descBin2){
    int hamming_dist = 0;
    int loop_range = min(descBin1.size(), descBin2.size());
    for(int i = 0; i < loop_range; i++){
        int cur_indx_hamming_distance = 0;
        for(int j = 0; j < 8; j++){
            if(descBin1[i][j] != descBin2[i][j]){
                cur_indx_hamming_distance += 1;
            }
        }
        hamming_dist += cur_indx_hamming_distance;
    }
    return hamming_dist;
}

// Get the keypoint matches
vector<pair<int, int>> getMatches_Keypoints(const cv::Mat& descriptors1, const cv::Mat& descriptors2) {
    vector<pair<int, int>> matches;

    for (int i = 0; i < descriptors1.rows; i++) {
        int best_match_index = -1;
        int best_hamming_dist = INT_MAX;

        // Get binary descriptor for the current row in descriptors1
        vector<string> bin_desc1 = desctoBinary(descriptors1.row(i));

        for (int j = 0; j < descriptors2.rows; j++) {
            // Get binary descriptor for the current row in descriptors2
            vector<string> bin_desc2 = desctoBinary(descriptors2.row(j));

            // Compute Hamming distance
            int hamming_dist = computeHammingDistance(bin_desc1, bin_desc2);
            if (hamming_dist < best_hamming_dist) {
                best_hamming_dist = hamming_dist;
                best_match_index = j;
            }
        }

        if (best_match_index != -1) {
            //Use Lowe's ratio test to check if the match is a good one
            // cout << "best hamming distance: " << best_hamming_dist << endl;
            if(best_hamming_dist < 45){
                matches.push_back({i, best_match_index});
            }
        }
    }
    return matches;
}


void plotMatches(cv::Mat img1, cv::Mat img2, vector<cv::KeyPoint> keypoints1, vector<cv::KeyPoint> keypoints2, vector<pair<int, int>> matches) {
    // Convert grayscale images to BGR for colored visualization
    cv::Mat img1_color, img2_color;
    cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);
    
    // Create the output image with appropriate size
    cv::Mat img_matches(img1.rows, img1.cols + img2.cols, CV_8UC3);
    
    // Copy the images into the output image
    img1_color.copyTo(img_matches(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2_color.copyTo(img_matches(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    for(const auto& match : matches) {
        int idx1 = match.first;
        int idx2 = match.second;

        cv::Point pt1 = keypoints1[idx1].pt;
        cv::Point pt2 = cv::Point(keypoints2[idx2].pt.x + img1.cols, keypoints2[idx2].pt.y);

        cv::line(img_matches, pt1, pt2, cv::Scalar(0, 255, 0), 1);
        cv::circle(img_matches, pt1, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(img_matches, pt2, 3, cv::Scalar(255, 0, 0), -1);
    }

    cv::imshow("Matched Keypoints", img_matches);
    cv::waitKey(0);
}

int main(){
    // Get ORB keypoints and descriptors
    cv::Mat img1 = cv::imread("../dataset/Book Statue/WhatsApp Image 2024-11-25 at 19.01.18 (1).jpeg", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("../dataset/Book Statue/WhatsApp Image 2024-11-25 at 19.01.18 (2).jpeg", cv::IMREAD_GRAYSCALE);

    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    cout << "Keypoints1 size: " << keypoints1.size() << ", Descriptors1 size: " << descriptors1.rows << endl;
    cout << "Keypoints2 size: " << keypoints2.size() << ", Descriptors2 size: " << descriptors2.rows << endl;

    // Compute matches
    vector<pair<int, int>> matches = getMatches_Keypoints(descriptors1, descriptors2);

    cout << "Number of matches: " << matches.size() << endl;

    // Plot matches
    plotMatches(img1, img2, keypoints1, keypoints2, matches);

    return 0;
}
