#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include "bfmatcher.hpp"
namespace fs = std::filesystem;

using namespace std;

bool epipolar_contraint(cv::Mat fundamental_matrix, cv::Point2d keypoint1, cv::Point2d keypoint2, double epsilon = 0.01){
    cv::Mat keypoint1_mat = (cv::Mat_<double>(3,1) << keypoint1.x, keypoint1.y, 1);//define homogenous coordinate
    cv::Mat keypoint2_mat = (cv::Mat_<double>(3,1) << keypoint2.x, keypoint2.y, 1);//define homogenous cooridnate

    cv::Mat epipolar_line_2 = fundamental_matrix * keypoint1_mat;
    cout << "epipolar line: " << epipolar_line_2 << endl;

    double error = keypoint2_mat.dot(epipolar_line_2);

    // normalize the error
    error = error / sqrt(pow(epipolar_line_2.at<double>(0),2) + pow(epipolar_line_2.at<double>(1),2));

    if (abs(error) < epsilon){
        return true;
    }else{
        return false;
    }
}

int main(){
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    string img_path = "../dataset/maingate_statue/";
    vector<string> images;

    for (const auto & entry : fs::directory_iterator(img_path))
        images.push_back(entry.path());

    pair<string,string> initial_img_pair = {};
    int max_inliers = 0;

    for(int i = 0; i < images.size()-1; i++){
        cv::Mat img1 = cv::imread(images[i], cv::IMREAD_GRAYSCALE);
        for(int j = i+1; j < images.size(); j++){
            int num_inliers = 0;
            cv::Mat img2 = cv::imread(images[j], cv::IMREAD_GRAYSCALE);
            cout << images[i] << endl;
            cout << images[j] << endl;
            orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
            orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
            vector<pair<cv::KeyPoint, cv::KeyPoint>> matches = getMatches_Keypoints(descriptors1, descriptors2, keypoints1, keypoints2, 0.75);
            cout << "number of matches: " << matches.size() << endl;
            // plotMatches(img1, img2, keypoints1, keypoints2, matches);

            vector<cv::Point2f> points1, points2;

            for(auto match : matches){
                points1.push_back(match.first.pt);
                points2.push_back(match.second.pt);
            }

            // TODO: implement fundamental matrix using ransac
            if (points1.size() >= 8 && points2.size() >= 8) { //minimum for RANSAC
                cv::Mat fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC);
                cout << "fundamental Matrix: " << endl << fundamental_matrix << endl;
                for(int k = 0; k < points1.size(); k++){
                    bool epipolar_constraint_satisfied = epipolar_contraint(fundamental_matrix, points1[k], points2[k]);
                    if(epipolar_constraint_satisfied){
                        num_inliers += 1;
                    }
                }
                if(num_inliers > 0){
                    cout << "number of inliers: " << num_inliers << endl;
                }
            } else {
                // cout << "insufficient points for fundamental matrix estimation." << endl;
            }
            if(num_inliers > max_inliers){
                max_inliers = num_inliers;
                initial_img_pair = {images[i], images[j]};
            }
            cout << endl << endl;
        }
    }
    cout << "initial image pair: " << initial_img_pair.first << " " << initial_img_pair.second << endl;
    cout << "max inliers: " << max_inliers << endl;
    return 0;

}