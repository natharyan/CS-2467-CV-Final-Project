#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <fstream>
#include "bfmatcher.hpp"
#include "epipolar.hpp"

namespace fs = std::filesystem;

using namespace std;


// We’ll decompose the essential matrix (computed from the fundamental matrix) to obtain the rotation and translation between the initial image pair.


// get the essential matrix from the fundamental matrix

cv::Mat getEssentialMatrix(cv::Mat fundamental_matrix, cv::Mat K){
    cv::Mat E = K.t() * fundamental_matrix * K;
    return E;
}

// decompose the essential matrix to get the rotation matrix and translation vector between the initial image pair

pair<cv::Mat, cv::Mat> RotationAndTranslation(cv::Mat essential_matrix){
    cv::Mat W, U, Vt;

    // decompose the essential matrix to get the rotation matrix and translation vector
    // essential matrix = U * diag(s, s, 0) * Vt where the diagonal matrix contains the singular values from the essential matrix
    cv::SVD svd(essential_matrix, cv::SVD::FULL_UV);

    W = (cv::Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
    U = svd.u;
    Vt = svd.vt;

    // generate candidate rotation matrices and the translation vector
    cv::Mat R1 = U * W * Vt;
    cv::Mat R2 = U * W.t() * Vt;
    // get the third column of the U matrix as the translation vector (up to scale)
    cv::Mat t = U.col(2);

    return {R1, t};
}

int main(){
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cv::Mat descriptors1, descriptors2;
    vector<cv::KeyPoint> keypoints1, keypoints2;
    pair<string,string> initial_image_pair_paths;
    bool FLAG_initial_image_pair = true; // set to true if the initial image pair is provided

    // check if initial_image_pair.txt exists read the initial image pair from the file
    if(FLAG_initial_image_pair){ 
        if(fs::exists("static/initial_image_pair.txt")){
            ifstream initial_image_pair_file("static/initial_image_pair.txt");
            string img1_path, img2_path;
            getline(initial_image_pair_file, img1_path);
            getline(initial_image_pair_file, img2_path);
            initial_image_pair_file.close();
            cout << "initial image pair: " << img1_path << " " << img2_path << endl;
            initial_image_pair_paths = {img1_path, img2_path};
        }else{
            cout << "initial_image_pair.txt does not exist." << endl;
            FLAG_initial_image_pair = false;
        }
    }else{
        string img_path = "dataset/maingate_statue/";
        vector<string> images;

        for (const auto & entry : fs::directory_iterator(img_path))
            images.push_back(entry.path());

        cout << "Getting the initial image pair..." << endl;
        initial_image_pair_paths = initial_image_pair(images);
        cout << "initial image pair: " << initial_image_pair_paths.first << " " << initial_image_pair_paths.second << endl;
        // write the initial image pair to a file
        ofstream initial_image_pair_file("static/initial_image_pair.txt");
        initial_image_pair_file << initial_image_pair_paths.first << endl;
        initial_image_pair_file << initial_image_pair_paths.second << endl;
        initial_image_pair_file.close();
        cout << "initial image pair: " << initial_image_pair_paths.first << " " << initial_image_pair_paths.second << endl;
    }

    cv::Mat img1 = cv::imread(initial_image_pair_paths.first, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(initial_image_pair_paths.second, cv::IMREAD_GRAYSCALE);
    orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
    // use our BFMatcher to return the matched keypoints
    vector<pair<cv::KeyPoint, cv::KeyPoint>> matches = getMatches_Keypoints(descriptors1, descriptors2, keypoints1, keypoints2, 0.75);
    vector<cv::Point2f> points1, points2;

    for(auto match : matches){
        points1.push_back(match.first.pt);
        points2.push_back(match.second.pt);
    }
    
    // plot the epipolar lines and inliers for the initial image pair
    int num_inliers = 0;
    cv::Mat fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC);
    cout << "fundamental Matrix: " << endl << fundamental_matrix << endl;
    for(int k = 0; k < points1.size(); k++){
        bool epipolar_constraint_satisfied = epipolar_contraint(fundamental_matrix, points1[k], points2[k]);
        if(epipolar_constraint_satisfied){
            num_inliers += 1;
        }
    }
    if(num_inliers > 0){
        cout << "max number of inliers: " << num_inliers << endl;
    }

    vector<bool> inliers_mask = getInliers(fundamental_matrix, points1, points2);

    plotEpipolarLinesAndInliers(img1, img2, points1, points2, fundamental_matrix, inliers_mask);


    // get the camera matrix from file 


    return 0;

}