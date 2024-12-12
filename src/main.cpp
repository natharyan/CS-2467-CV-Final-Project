#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <filesystem>
#include <vector>
#include <fstream>
#include "bfmatcher.hpp"
#include "epipolar.hpp"
#include "orb.hpp"


namespace fs = std::filesystem;

using namespace std;


// get the essential matrix from the fundamental matrix
cv::Mat getEssentialMatrix(cv::Mat fundamental_matrix, cv::Mat K){
    cv::Mat E = K.t() * fundamental_matrix * K;
    return E;
}

// decompose the essential matrix to get the rotation matrix and translation vector between the initial image pair

vector<pair<cv::Mat, cv::Mat>> RotationAndTranslation(cv::Mat essential_matrix){
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
    vector<pair<cv::Mat, cv::Mat>> candidates = {{R1, t}, {R2, t}, {R1, -t}, {R2, -t}};
    return candidates;
}


// get the camera intrinsics from calibration file
pair<cv::Mat,cv::Mat> getCameraIntrinsics(){
    ifstream calibration_file("src/helpers/calibration.txt");
    if (!calibration_file.is_open()) {
        cerr << "Failed to open calibration file!" << endl;
        throw runtime_error("Calibration file not found");
    }

    cv::Mat K, distortion_coefficients;
    string line;
    vector<double> K_data, dist_data;

    // Read camera matrix
    if (getline(calibration_file, line)) {
        stringstream ss(line);
        double val;
        while(ss >> val){
            K_data.push_back(val);
        }
    }

    // Read distortion coefficients
    if (getline(calibration_file, line)) {
        stringstream ss(line);
        double val;
        while(ss >> val){
            dist_data.push_back(val);
        }
    }

    // Verify data
    if (K_data.size() != 9) {
        cerr << "Invalid number of values for camera matrix. Got " << K_data.size() << endl;
        throw runtime_error("Incorrect camera matrix format");
    }

    if (dist_data.size() != 5) {
        cerr << "Invalid number of distortion coefficients. Got " << dist_data.size() << endl;
        throw runtime_error("Incorrect distortion coefficients format");
    }

    // Create matrices
    K = (cv::Mat_<double>(3, 3) << 
        K_data[0], K_data[1], K_data[2], 
        K_data[3], K_data[4], K_data[5], 
        K_data[6], K_data[7], K_data[8]);

    distortion_coefficients = (cv::Mat_<double>(1, 5) << 
        dist_data[0], dist_data[1], dist_data[2], 
        dist_data[3], dist_data[4]);

    return {K, distortion_coefficients};
}



int main(){

    string imgpath = "../dataset/Book Statue/WhatsApp Image 2024-11-25 at 19.01.18 (1).jpeg";
    cv::Mat baseImage;
    baseImage = createBaseImage(imgpath);
    cout << "SIFT running..." << endl;
    cv::imshow("Base Image", baseImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
    // calling FAST9 
    cout << "running" << endl;
    vector<cv::Point> initialKeypoints = FAST9(baseImage, 20);
    cout << "Number of keypoints detected by FAST9: " << initialKeypoints.size() << endl;

    // harris corner response 
    vector<KeypointWithResponse> keypointsWithResponses;
    for (const cv::Point& kp : initialKeypoints) {
        double response = harrisResponse(baseImage, kp);
        // Retain only the keypoints with a Harris response greater than 0.01
        if (response > 0.01) { 
            keypointsWithResponses.push_back({kp, response});
        }
    }

    // Sort the keypoints based on Harris response
    sort(keypointsWithResponses.begin(), keypointsWithResponses.end(), [](const KeypointWithResponse& a, const KeypointWithResponse& b) {
        return a.harrisResponse > b.harrisResponse;
    });

    // Retain only the top N keypoints
    const int maxKeypoints = 500;
    if (keypointsWithResponses.size() > maxKeypoints) {
        keypointsWithResponses.resize(maxKeypoints);
    }

    cout << "Number of keypoints after Harris filtering: " << keypointsWithResponses.size() << endl;

    // Extract the keypoints and compute orientations
    vector<cv::Point> filteredKeypoints;
    vector<double> orientations;
    for (const auto& kpWithResponse : keypointsWithResponses) {
        filteredKeypoints.push_back(kpWithResponse.point);
        orientations.push_back(orientationAssignment(baseImage, kpWithResponse.point));
    }
    cout << filteredKeypoints << endl; 

    vector<cv::KeyPoint> cvKeypoints;
    for (const auto& point : filteredKeypoints) {
        cv::KeyPoint kp(point.x, point.y, 31); // Set patch size (31 as example)
        cvKeypoints.push_back(kp);
    }

    // Compute rBRIEF descriptors
    vector<cv::Mat> descriptors = rBRIEF(baseImage, filteredKeypoints, 31);
    cout << "Descriptors computed using rBRIEF." << endl;

    // Visualize the keypoints
    cv::Mat displayImage;
    cvtColor(baseImage, displayImage, cv::COLOR_GRAY2BGR);  

    // Display keypoints on the image
    cv::Mat imgWithKeypoints;
    drawKeypoints(baseImage, cvKeypoints, imgWithKeypoints, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);

    // Show the image with keypoints
    cv::imshow("Keypoints", imgWithKeypoints);
    cv::waitKey(0);

    // Print the descriptors
    cout << "Descriptors (first 5 descriptors):" << endl;
    for (size_t i = 0; i < min(descriptors.size(), (size_t)5); ++i) {
        cout << "Descriptor " << i << ": " << descriptors[i] << endl;
    }
    // cout << descriptors.size() << endl;
    // cout << "done" << endl;

    cv::imshow("Filtered Keypoints with Orientations", displayImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cout << "ORB pipeline completed successfully." << endl;
    return 0;
    
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

    // get the camera intrinsics from calibration file
    // TODO: fix code for getting the camera intrinsics from calinration.txt (function: getCameraIntrinsics)
    // pair<cv::Mat,cv::Mat> intrinsics = getCameraIntrinsics();
    // cv::Mat K = intrinsics.first;
    // cv::Mat distortion_coefficients = intrinsics.second;
    // cout << "Camera intrinsics: " << endl << K << endl;
    // cout << "Distortion coefficients: " << endl << distortion_coefficients << endl;

    //K = [[3.04690978e+03 0.00000000e+00 1.58798925e+03] [0.00000000e+00 3.04241561e+03 1.52143054e+03] [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
    // distortion_coefficients = [[ 3.92176156e-02 -4.71862125e-01  1.37646274e-03  4.51593168e-04 1.81876525e+00]]
    cv::Mat K = (cv::Mat_<double>(3, 3) << 3.04690978e+03, 0.00000000e+00, 1.58798925e+03, 0.00000000e+00, 3.04241561e+03, 1.52143054e+03, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00);
    cv::Mat distortion_coefficients = (cv::Mat_<double>(1, 5) << 3.92176156e-02, -4.71862125e-01, 1.37646274e-03, 4.51593168e-04, 1.81876525e+00);

    // undistort and normalize points
    vector<cv::Point2f> normalized_points1, normalized_points2;
    cv::undistortPoints(points1, normalized_points1, K, distortion_coefficients);
    cv::undistortPoints(points2, normalized_points2, K, distortion_coefficients);
    // points1 = normalized_points1;
    // points2 = normalized_points2;
    cout << endl;
    // plot the epipolar lines and inliers for the initial image pair
    int num_inliers = 0;
    // TODO: implement fundamental matrix using ransac from scratch
    cv::Mat fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC);
    cout << "fundamental Matrix: " << endl << fundamental_matrix << endl;
    for(int k = 0; k < points1.size(); k++){
        bool epipolar_constraint_satisfied = epipolar_contraint(fundamental_matrix, points1[k], points2[k]);
        if(epipolar_constraint_satisfied){
            num_inliers += 1;
        }
    }
    cout << endl;
    if(num_inliers > 0){
        cout << "max number of inliers: " << num_inliers << endl;
    }
    cout << endl;

    vector<bool> inliers_mask = getInliers(fundamental_matrix, points1, points2);

    plotEpipolarLinesAndInliers(img1, img2, points1, points2, fundamental_matrix, inliers_mask);

    cv::Mat E = getEssentialMatrix(fundamental_matrix, K);
    vector<pair<cv::Mat,cv::Mat>> rotation_translationCandidates = RotationAndTranslation(E);

    // use cheirality check to get the correct rotation and translation
    cv::Mat R, t;
    int counter = 0;
    for (pair<cv::Mat, cv::Mat> R_t : rotation_translationCandidates) {
        // normalize the translation vector
        R_t.second /= cv::norm(R_t.second);

        // construct projection matrices
        cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F); // P1 = K * [I | 0]
        cv::Mat Rt_combined;
        cv::hconcat(R_t.first, R_t.second, Rt_combined); // Combine R and t
        cv::Mat P2 = K * Rt_combined; // P2 = K * [R | t]

        // triangulate points
        vector<cv::Point3d> points3d;
        cv::Mat points4d;
        // TODO: implement triangulation from scratch
        cv::triangulatePoints(P1, P2, normalized_points1, normalized_points2, points4d);

        for (int i = 0; i < points4d.cols; i++) {
            cv::Mat point4d = points4d.col(i);
            point4d /= point4d.at<double>(3, 0); // Normalize homogeneous coordinates
            points3d.push_back(cv::Point3d(point4d.at<double>(0, 0), point4d.at<double>(1, 0), point4d.at<double>(2, 0)));
        }

        // check cheirality condition
        bool cheirality = true;
        for (cv::Point3d point3d : points3d) {
            if (point3d.z < -1e-6) { // Allow small tolerance for numerical errors
                cheirality = false;
                break;
            }
        }
        if (cheirality) {
            R = R_t.first;
            t = R_t.second;
            cout << "Cheirality check passed for candidate " << counter << endl;
            break;
        }
        counter++;
    }

    cout << "Rotation matrix: " << endl << R << endl;
    cout << "Translation vector: " << endl << t << endl;

    return 0;
}
// TODO: Add normalization and undistortion before plotting the epipolar lines and inliers
// TODO: implement undistortion from scratch?