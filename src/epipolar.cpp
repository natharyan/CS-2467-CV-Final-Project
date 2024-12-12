// implementations for the epipolar constraint, finding the initial image pair, inliers, epipolar lines, 
// and plotting the epipolar lines and inliers
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include "bfmatcher.hpp"
#include "epipolar.hpp"

using namespace std;

// get the epipolar lines using the fundamental matrix and the matched keypoints from the first image in the pair selected
cv::Mat epipolar_line(cv::Mat fundamental_matrix, cv::Point2d keypoint1){
    cv::Mat keypoint1_mat = (cv::Mat_<double>(3,1) << keypoint1.x, keypoint1.y, 1);//define homogenous coordinate
    cv::Mat epipolar_line = fundamental_matrix * keypoint1_mat;
    return epipolar_line;
}

// get inliers based on the epipolar constraint
vector<bool> getInliers(cv::Mat fundamental_matrix, vector<cv::Point2f> &points1, vector<cv::Point2f> &points2, double epsilon){
    vector<bool> inliers;
    for(int i = 0; i < points1.size(); i++){
        cv::Mat keypoint1_mat = (cv::Mat_<double>(3,1) << points1[i].x, points1[i].y, 1);//define homogenous coordinate
        cv::Mat keypoint2_mat = (cv::Mat_<double>(3,1) << points2[i].x, points2[i].y, 1);//define homogenous cooridnate

        cv::Mat epipolar_line_2 = epipolar_line(fundamental_matrix, points1[i]);

        double error = keypoint2_mat.dot(epipolar_line_2);

        // normalize the error
        error = error / sqrt(pow(epipolar_line_2.at<double>(0),2) + pow(epipolar_line_2.at<double>(1),2));

        if (abs(error) < epsilon){
            inliers.push_back(true);
        }else{
            inliers.push_back(false);
        }
    }
    return inliers;
}

// returns true for points that are within epsilon error from zero for the epipolar constraint
bool epipolar_contraint(cv::Mat fundamental_matrix, cv::Point2d keypoint1, cv::Point2d keypoint2, double epsilon){
    cv::Mat keypoint1_mat = (cv::Mat_<double>(3,1) << keypoint1.x, keypoint1.y, 1);//define homogenous coordinate
    cv::Mat keypoint2_mat = (cv::Mat_<double>(3,1) << keypoint2.x, keypoint2.y, 1);//define homogenous cooridnate
    cv::Mat epipolar_line_2 = epipolar_line(fundamental_matrix, keypoint1);

    double error = keypoint2_mat.dot(epipolar_line_2);

    // normalize the error
    error = error / sqrt(pow(epipolar_line_2.at<double>(0),2) + pow(epipolar_line_2.at<double>(1),2));

    if (abs(error) < epsilon){
        // cout << "epipolar line: " << epipolar_line_2 << endl;
        return true;
    }else{
        return false;
    }
}

// finds the initial image pair from a directory of images
pair<string,string> initial_image_pair(vector<string> images){
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    pair<string,string> initial_img_pair = {};
    int max_inliers = 0;
    cv::Mat descriptors1, descriptors2;
    vector<cv::KeyPoint> keypoints1, keypoints2;
    
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
                // cout << "fundamental Matrix: " << endl << fundamental_matrix << endl;
                for(int k = 0; k < points1.size(); k++){
                    bool epipolar_constraint_satisfied = epipolar_contraint(fundamental_matrix, points1[k], points2[k]);
                    if(epipolar_constraint_satisfied){
                        num_inliers += 1;
                    }
                }
                if(num_inliers > 0){
                    // cout << "number of inliers: " << num_inliers << endl;
                }
            } else {
                // cout << "insufficient points for fundamental matrix estimation." << endl;
            }
            if(num_inliers > max_inliers){
                max_inliers = num_inliers;
                initial_img_pair = {images[i], images[j]};
            }
            // cout << endl << endl;
        }
    }
    return initial_img_pair;
}

// plot the inliers and the epipolar lines for all keypoints from the first image that have been matched with keypoints from the second image using the bfmatcher implementation
void plotEpipolarLinesAndInliers(cv::Mat &img1, cv::Mat &img2, vector<cv::Point2f> &points1, vector<cv::Point2f> &points2, cv::Mat &fundamental_matrix, vector<bool> &inliers){
    cv::Mat combined_img(img1.rows, img1.cols + img2.cols, img1.type());
    img1.copyTo(combined_img(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(combined_img(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    for(size_t i = 0; i < points1.size(); i++){
        cv::Point2f point1 = points1[i];
        cv::Point2f point2 = points2[i];

        cv::Mat epipolar_line_2 = epipolar_line(fundamental_matrix, point1);

        double a = epipolar_line_2.at<double>(0);
        double b = epipolar_line_2.at<double>(1);
        double c = epipolar_line_2.at<double>(2);
        cv::Point2f line_start(0, -c/b);
        cv::Point2f line_end(img2.cols, -(a*img2.cols + c)/b);
        cv::line(combined_img, line_start + cv::Point2f(img1.cols, 0), line_end + cv::Point2f(img1.cols, 0), cv::Scalar(0, 255, 0), 2);
        if(inliers[i]){
            cv::circle(combined_img, point1, 50, cv::Scalar(255, 0, 0), -1);
            cv::circle(combined_img, point2 + cv::Point2f(img1.cols, 0), 50, cv::Scalar(255, 0, 0), -1);
        }
    }
    cv::imshow("Epipolar Lines and Inliers", combined_img);
    cv::waitKey(0);
}