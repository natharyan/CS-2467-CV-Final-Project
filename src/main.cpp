#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <filesystem>
#include <vector>
#include <fstream>
#include "bfmatcher.hpp"
#include "epipolar.hpp"
#include "ransac.hpp"
#include "triangulation.hpp"

namespace fs = filesystem;

using namespace std;

// get the essential matrix from the fundamental matrix
cv::Mat getEssentialMatrix(cv::Mat fundamental_matrix, cv::Mat K){
    cv::Mat E = K.t() * fundamental_matrix * K;
    return E;
}

// decompose the essential matrix to get the rotation matrix and translation vector between the initial image pair
vector<pair<cv::Mat, cv::Mat>> RotationAndTranslation(cv::Mat essential_matrix) {
    cv::Mat W, U, Vt;

    // Decompose the essential matrix to get the rotation matrix and translation vector
    cv::SVD svd(essential_matrix, cv::SVD::FULL_UV);

    W = (cv::Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
    U = svd.u;
    Vt = svd.vt;

    // Ensure the determinants of U and Vt are positive
    if (cv::determinant(U) < 0) {
        U *= -1;
    }
    if (cv::determinant(Vt) < 0) {
        Vt *= -1;
    }

    // Generate candidate rotation matrices and the translation vector
    cv::Mat R1 = U * W * Vt;
    cv::Mat R2 = U * W.t() * Vt;

    // Ensure rotation matrices have determinant +1
    if (cv::determinant(R1) < 0) {
        R1 *= -1;
    }
    if (cv::determinant(R2) < 0) {
        R2 *= -1;
    }

    // Get the third column of the U matrix as the translation vector (up to scale)
    cv::Mat t = U.col(2);
    cout << "t: " << t << endl;

    cout << "R1 determinant: " << cv::determinant(R1) << endl;
    cout << "R2 determinant: " << cv::determinant(R2) << endl;

    // Return all four combinations of R and t
    vector<pair<cv::Mat, cv::Mat>> candidates = {
        {R1, t}, {R2, t}, {R1, -t}, {R2, -t}
    };
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

// normalize the points
// normalize the points with centroid and scale
// std::tuple<cv::Point2f, double, std::vector<cv::Point2f>> normalizePoints(const std::vector<cv::Point2f>& points) {
//     cv::Point2f centroid(0, 0);
//     for (const auto& point : points) {
//         centroid.x += point.x;
//         centroid.y += point.y;
//     }
//     centroid.x /= points.size();
//     centroid.y /= points.size();

//     double rms_dist = 0;
//     for (const auto& point : points) {
//         rms_dist += cv::norm(point - centroid);
//     }
//     rms_dist = sqrt(rms_dist / points.size());

//     double scale = sqrt(2.0) / rms_dist;

//     std::vector<cv::Point2f> normalized_points;
//     for (const auto& point : points) {
//         normalized_points.emplace_back((point - centroid) * scale);
//     }

//     return {centroid, scale, normalized_points};
// }

// normalization with intrinsice matrix
tuple<cv::Point2f, double, std::vector<cv::Point2f>> normalizePoints(const vector<cv::Point2f>& points, const cv::Mat& K) {
    // Ensure K is a 3x3 matrix
    if (K.rows != 3 || K.cols != 3) {
        throw std::invalid_argument("Intrinsic matrix K must be 3x3");
    }

    // Extract intrinsic parameters
    double fx = K.at<double>(0, 0); // Focal length in x
    double fy = K.at<double>(1, 1); // Focal length in y
    double u0 = K.at<double>(0, 2); // Principal point x
    double v0 = K.at<double>(1, 2); // Principal point y

    vector<cv::Point2f> normalizedPoints;
    cv::Point2f centroid(0, 0);
    double scaleFactor = 0;

    // Normalize each point
    for (const auto& pt : points) {
        double x_norm = (pt.x - u0) / fx;
        double y_norm = (pt.y - v0) / fy;

        cv::Point2f norm_pt(x_norm, y_norm);
        normalizedPoints.push_back(norm_pt);

        // Accumulate centroid
        centroid.x += norm_pt.x;
        centroid.y += norm_pt.y;
    }

    // Compute centroid
    centroid.x /= points.size();
    centroid.y /= points.size();

    // Compute average scale (distance from centroid to points)
    double totalScale = 0.0;
    for (const auto& norm_pt : normalizedPoints) {
        totalScale += std::sqrt(std::pow(norm_pt.x - centroid.x, 2) + std::pow(norm_pt.y - centroid.y, 2));
    }

    scaleFactor = totalScale / points.size();

    return {centroid, scaleFactor, normalizedPoints};
}

// denormalize the points with the centroid and scale
// std::vector<cv::Point3f> denormalize3DPoints(
//     const std::vector<cv::Point3f>& points,
//     const cv::Point2f& centroid, double scale) {

//     std::vector<cv::Point3f> denormalized_points;
//     for (const auto& point : points) {
//         denormalized_points.emplace_back(
//             point.x / scale + centroid.x,
//             point.y / scale + centroid.y,
//             point.z * scale); // Scale depth appropriately
//     }
//     return denormalized_points;
// }

// denormalization with intrinsic matrix
vector<cv::Point3f> denormalize3DPoints(const vector<cv::Point3f>& normalizedPoints, const cv::Mat& K) {
    // Ensure K is a 3x3 matrix
    if (K.rows != 3 || K.cols != 3) {
        throw std::invalid_argument("Intrinsic matrix K must be 3x3");
    }

    // Extract intrinsic parameters
    double fx = K.at<double>(0, 0); // Focal length in x
    double fy = K.at<double>(1, 1); // Focal length in y
    double u0 = K.at<double>(0, 2); // Principal point x
    double v0 = K.at<double>(1, 2); // Principal point y

    std::vector<cv::Point3f> denormalizedPoints;

    for (const auto& pt : normalizedPoints) {
        double x_denorm = fx * pt.x + u0;
        double y_denorm = fy * pt.y + v0;
        double z_denorm = pt.z; // z-coordinate remains unchanged

        denormalizedPoints.emplace_back(x_denorm, y_denorm, z_denorm);
    }

    return denormalizedPoints;
}

tuple<vector<cv::Point3f>, vector<cv::KeyPoint>, cv::Mat> incrementalAddition(const string& cur_imgpath, const string& prev_imgpath, const cv::Mat& K, const cv::Mat& distortion_coefficients, vector<cv::KeyPoint>& keypoints_prev, cv::Mat& descriptors_prev, vector<cv::Point3f>& points3d_prev) {
    // Read the new image
    cv::Mat cur_img = cv::imread(cur_imgpath, cv::IMREAD_GRAYSCALE);
    cv::Mat prev_img = cv::imread(prev_imgpath, cv::IMREAD_GRAYSCALE);
    // Detect and compute keypoints and descriptors for the new image
    vector<cv::KeyPoint> keypoints_curr;
    cv::Mat descriptors_curr;
    // cv::Ptr<cv::ORB> orb = cv::ORB::create();
    // orb->detectAndCompute(img_undistorted, cv::noArray(), keypoints_curr, descriptors_curr);
    pair<vector<cv::KeyPoint>,cv::Mat> keypoints_descriptors = runORB(cur_imgpath);
    keypoints_curr = keypoints_descriptors.first;
    descriptors_curr = keypoints_descriptors.second;
    
    // Match features between the previous and current images
    vector<pair<cv::KeyPoint, cv::KeyPoint>> matches = getMatches_Keypoints(descriptors_prev, descriptors_curr, keypoints_prev, keypoints_curr, 0.75);
    cout << "Number of matches: " << matches.size() << endl;
    vector<cv::Point2f> points_prev, points_curr;
    for (auto match : matches) {
        points_prev.push_back(match.first.pt);
        points_curr.push_back(match.second.pt);
    }

    // Plot the matches
    cout << "Plotting matches..." << endl;
    cout << "Number of points_prev: " << points_prev.size() << endl;
    cout << "Number of points_curr: " << points_curr.size() << endl;
    cout << "Image 1 size: " << prev_img.size() << endl;
    cout << "Image 2 size: " << cur_img.size() << endl;
    if(points_prev.size() == 0 || points_curr.size() == 0){
        cout << "No matches found." << endl;
        return make_tuple(points3d_prev, keypoints_prev, descriptors_prev);
    }
    plotMatches(prev_img, cur_img, matches);
    cout << "Matches plotted." << endl;
    // Estimate the fundamental matrix using RANSAC
    cv::Mat fundamental_matrix = cv::findFundamentalMat(points_prev, points_curr, cv::FM_RANSAC);
    // fundamental_matrix.convertTo(fundamental_matrix, CV_64F);
    // int num_inliers = 0;
    // int maxIterations = 1000;
    // double threshold = 0.01;

    // vector<pair<Eigen::Vector2d, Eigen::Vector2d>> eigen_matches;
    // for (const auto& match : matches) {
    //     Eigen::Vector2d pt1(match.first.pt.x, match.first.pt.y);
    //     Eigen::Vector2d pt2(match.second.pt.x, match.second.pt.y);
    //     eigen_matches.emplace_back(pt1, pt2);
    // }
    // pair<MatrixXd, vector<bool>> F_and_inliers = ransacFundamentalMatrix(eigen_matches, maxIterations, threshold);
    // MatrixXd F = F_and_inliers.first;
    // vector<bool> inliers = F_and_inliers.second;
    // cv::Mat fundamental_matrix(F.rows(), F.cols(), CV_64F);
    cout << "Fundamental matrix: " << fundamental_matrix << endl;
    cout << "Size: " << fundamental_matrix.size() << endl;
    // all elements are zero in the fundamental matrix
    if (!fundamental_matrix.empty() && fundamental_matrix.rows >= 3 && fundamental_matrix.cols >= 3) {
        fundamental_matrix = fundamental_matrix(cv::Range(0, 3), cv::Range(0, 3));
    } else {
        cout << "Error: Fundamental matrix has invalid dimensions or is empty." << endl;
        return make_tuple(points3d_prev, keypoints_prev, descriptors_prev);
    }
    // Compute the essential matrix
    cv::Mat E = getEssentialMatrix(fundamental_matrix, K);
    cout << "Essential matrix: " << endl << E << endl;
    // Decompose the essential matrix to get rotation and translation candidates
    vector<pair<cv::Mat, cv::Mat>> rotation_translationCandidates = RotationAndTranslation(E);

    // Normalize points
    vector<cv::Point2f> norm_points_prev, norm_points_curr;
    cv::Point2f centroid_prev, centroid_curr;
    double scale_prev, scale_curr;
    tie(centroid_prev, scale_prev, norm_points_prev) = normalizePoints(points_prev, K);
    tie(centroid_curr, scale_curr, norm_points_curr) = normalizePoints(points_curr, K);
    // tie(centroid_curr, scale_curr, norm_points_curr) = normalizePoints(points_curr);

    // Use cheirality check to get the correct rotation and translation
    cv::Mat R, t;
    vector<cv::Point3f> points3d_curr;
    for (pair<cv::Mat, cv::Mat> R_t : rotation_translationCandidates) {
        cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F); // P1 = K * [I | 0]
        cv::Mat Rt_combined;
        cv::hconcat(R_t.first, R_t.second, Rt_combined); // Combine R and t
        cv::Mat P2 = K * Rt_combined; // P2 = K * [R | t]

        // Triangulate points
        points3d_curr = triangulation(P1, P2, norm_points_prev, norm_points_curr);

        // Check cheirality condition
        bool cheirality = true;
        for (cv::Point3d point3d : points3d_curr) {
            if (point3d.z < 0) {
                cheirality = false;
                break;
            }
        }
        if (cheirality) {
            R = R_t.first;
            t = R_t.second;
            break;
        }
    }

    // Denormalize the 3D points
    // points3d_curr = denormalize3DPoints(points3d_curr, centroid_prev, scale_prev);
    points3d_curr = denormalize3DPoints(points3d_curr, K);
    // Append the new 3D points to the previous 3D points
    points3d_prev.insert(points3d_prev.end(), points3d_curr.begin(), points3d_curr.end());

    // Update the previous keypoints and descriptors
    // keypoints_prev = keypoints_curr;
    // descriptors_prev = descriptors_curr;

    return make_tuple(points3d_prev, keypoints_curr, descriptors_curr);
}

int main(){
    cv::Mat descriptors1, descriptors2;
    vector<cv::KeyPoint> keypoints1, keypoints2;
    pair<string,string> initial_image_pair_paths;
    bool FLAG_initial_image_pair = false; // set to true if the initial image pair is provided

    // get the camera intrinsics from calibration file
    // TODO: fix code for getting the camera intrinsics from calinration.txt (function: getCameraIntrinsics)
    // pair<cv::Mat,cv::Mat> intrinsics = getCameraIntrinsics();
    // cv::Mat K = intrinsics.first;
    // cv::Mat distortion_coefficients = intrinsics.second;
    // cout << "Camera intrinsics: " << endl << K << endl;
    // cout << "Distortion coefficients: " << endl << distortion_coefficients << endl;

    // K =   [[3.04690978e+03 0.00000000e+00 1.58798925e+03] [0.00000000e+00 3.04241561e+03 1.52143054e+03] [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
    // distortion coefficients:  [[ 3.92176156e-02 -4.71862125e-01  1.37646274e-03  4.51593168e-04 1.81876525e+00]]
    cv::Mat K = (cv::Mat_<double>(3, 3) << 3.04690978e+03, 0.00000000e+00, 1.58798925e+03, 0.00000000e+00, 3.04241561e+03, 1.52143054e+03, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00);
    cv::Mat distortion_coefficients = (cv::Mat_<double>(1, 5) << 3.92176156e-02, -4.71862125e-01, 1.37646274e-03, 4.51593168e-04, 1.81876525e+00);

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
        string img_path = "dataset/Book Statue";
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
    cv::Mat img2 = cv::imread(initial_image_pair_paths.first, cv::IMREAD_GRAYSCALE);
    // undistort the images
    cv::Mat img1_undistorted, img2_undistorted;
    // cv::undistort(img1, img1_undistorted, K, distortion_coefficients);
    // cv::undistort(img2, img2_undistorted, K, distortion_coefficients);
    // TODO: getting rotation matrix determinant as 1 when I'm using images without undistortion
    // img1_undistorted = img1;
    // img2_undistorted = img2;
    // detect and compute the keypoints and descriptors
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    // orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    // orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    // TODO: uncomment this for Manya's orb implementation
    pair<vector<cv::KeyPoint>,cv::Mat> keypoints_descriptors1 = runORB(initial_image_pair_paths.first);
    keypoints1 = keypoints_descriptors1.first;
    descriptors1 = keypoints_descriptors1.second;

    pair<vector<cv::KeyPoint>,cv::Mat> keypoints_descriptors2 = runORB(initial_image_pair_paths.second);
    keypoints2 = keypoints_descriptors2.first;
    descriptors2 = keypoints_descriptors2.second;

    // use our BFMatcher to return the matched keypoints
    vector<pair<cv::KeyPoint, cv::KeyPoint>> matches = getMatches_Keypoints(descriptors1, descriptors2, keypoints1, keypoints2, 0.75);
    vector<cv::Point2f> points1, points2;

    for(auto match : matches){
        points1.push_back(match.first.pt);
        points2.push_back(match.second.pt);
    }

    vector<cv::KeyPoint> matches1, matches2;
    for(auto match : matches){
        matches1.push_back(match.first);
        matches2.push_back(match.second);
    }

    // plot the matches
    plotMatches(img1,img2,matches);

    // cout << "number of matches: " << matches.size() << endl;

    cout << endl;
    // plot the epipolar lines and inliers for the initial image pair
    int num_inliers = 0;
    int maxIterations = 1000;
    double threshold = 0.01;

    vector<pair<Eigen::Vector2d, Eigen::Vector2d>> eigen_matches;
    for (const auto& match : matches) {
        Eigen::Vector2d pt1(match.first.pt.x, match.first.pt.y);
        Eigen::Vector2d pt2(match.second.pt.x, match.second.pt.y);
        eigen_matches.emplace_back(pt1, pt2);
    }
    pair<MatrixXd, vector<bool>> F_and_inliers = ransacFundamentalMatrix(eigen_matches, maxIterations, threshold);
    cv::Mat fundamental_matrix_opencv = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC);
    // cout << "fundamental Matrix: " << endl << fundamental_matrix << endl;
    // check if fundamental matrix is empty
    // cout << "fundamental matrix: " << fundamental_matrix << endl;
    // convert F (MatrixXd) to cv::Mat
    MatrixXd F = F_and_inliers.first;
    vector<bool> inliers = F_and_inliers.second;
    cv::Mat fundamental_matrix(F.rows(), F.cols(), CV_64F); // Create a cv::Mat of appropriate size and type

    // Copy data from Eigen matrix to cv::Mat
    for (int i = 0; i < F.rows(); ++i) {
        for (int j = 0; j < F.cols(); ++j) {
            fundamental_matrix.at<double>(i, j) = F(i, j);
        }
    }
    
    // cout << "fundamental matrix(opencv): " << endl << fundamental_matrix_opencv << endl;
    // cout << "fundamental matrix: " << endl << fundamental_matrix << endl;
    for(int k = 0; k < points1.size(); k++){
        bool epipolar_constraint_satisfied = epipolar_contraint(fundamental_matrix_opencv, points1[k], points2[k]);
        if(epipolar_constraint_satisfied){
            num_inliers += 1;
        }
    }
    cout << endl;
    if(num_inliers > 0){
        cout << "max number of inliers: " << num_inliers << endl;
    }
    cout << endl;

    vector<bool> inliers_mask = getInliers(fundamental_matrix_opencv, points1, points2);

    plotEpipolarLinesAndInliers(img1, img2, points1, points2, fundamental_matrix_opencv, inliers_mask);

    cv::Mat E = getEssentialMatrix(fundamental_matrix_opencv, K);
    cout << "Essential matrix: " << endl << E << endl;
    vector<pair<cv::Mat,cv::Mat>> rotation_translationCandidates = RotationAndTranslation(E);

    // Normalize points to image coordinates using K
    vector<cv::Point2f> norm_points1, norm_points2;
    cv::Point2f centroid1, centroid2;
    double scale1, scale2;
    // tie(centroid1, scale1, norm_points1) = normalizePoints(points1);
    // tie(centroid2, scale2, norm_points2) = normalizePoints(points2);
    tie(centroid1, scale1, norm_points1) = normalizePoints(points1, K);
    tie(centroid2, scale2, norm_points2) = normalizePoints(points2, K);
    
    // norm_points1 = normalizePoints(points1, K);
    // norm_points2 = normalizePoints(points2, K);
    // for (const auto& pt : points1) {
    //     float x = (pt.x - K.at<double>(0, 2)) / K.at<double>(0, 0); // Normalize x
    //     float y = (pt.y - K.at<double>(1, 2)) / K.at<double>(1, 1); // Normalize y
    //     norm_points1.emplace_back(x, y);
    // }

    // for (const auto& pt : points2) {
    //     float x = (pt.x - K.at<double>(0, 2)) / K.at<double>(0, 0); // Normalize x
    //     float y = (pt.y - K.at<double>(1, 2)) / K.at<double>(1, 1); // Normalize y
    //     norm_points2.emplace_back(x, y);
    // }

    // use cheirality check to get the correct rotation and translation
    cv::Mat R, t;
    vector<cv::Point3f> points3d;
    cv::Mat points4d;
    cv::Mat normalized_points4d;

    // coordinate wise max difference in normalized points
    double max_diff = 0;
    for (int i = 0; i < norm_points1.size(); i++) {
        double diff = cv::norm(norm_points1[i] - norm_points2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    cout << "Max difference in normalized points: " << max_diff << endl;

    cout << "Matched points: " << endl;
    for (int i = 0; i < points1.size(); i++) {
        cout << "Point " << i << ": " << points1[i] << " " << points1[i] << endl;
    }

    int counter = 0;
    for (pair<cv::Mat, cv::Mat> R_t : rotation_translationCandidates) {
        // normalize the translation vector 
        // TODO: check if this is necessary
        // R_t.second /= cv::norm(R_t.second);
        // cout << "Rotation matrix candidate " << counter << ": " << endl << R_t.first << endl;
        // cout << "Translation vector candidate " << counter << ": " << endl << R_t.second << endl;
        // construct projection matrices
        cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F); // P1 = K * [I | 0]
        cv::Mat Rt_combined;
        cv::hconcat(R_t.first, R_t.second, Rt_combined); // Combine R and t
        // cout << "Rt_combined: " << endl << Rt_combined << endl;
        cv::Mat P2 = K * Rt_combined; // P2 = K * [R | t]
       
        // cv::Mat RT = cv::Mat::zeros(3, 4, R.type()); // Create a 3x4 matrix
        // R.copyTo(RT(cv::Rect(0, 0, 3, 3)));         // Copy R into the first 3 columns
        // T.copyTo(RT(cv::Rect(3, 0, 1, 3)));         // Copy T into the last column
        // cv::Mat P = K * RT;                         // Compute the projection matrix
        // triangulate points
        points3d = triangulation(P1, P2, norm_points1, norm_points2);
    
        // check cheirality condition
        bool cheirality = true;
        for (cv::Point3d point3d : points3d) {
            if (point3d.z < 0) {
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
    cout << "Translation vector: " << endl << t*cv::norm(t) << endl;
    cout << "Translation vector norm: " << cv::norm(t) << endl;

    cout << "3d points: " << points3d.size() << endl;
    // 3d points
    for (int i = 0; i < points3d.size(); i++) {
        cout << "3D point " << i << ": " << points3d[i] << endl;
    }

    // denormalize the 3d points
    points3d = denormalize3DPoints(points3d, K);
    // points3d = denormalize3DPoints(points3d, centroid1, scale1);
    cout << "Denormalized 3D points: " << endl;
    for (int i = 0; i < points3d.size(); i++) {
            cout << "3D point: " << points3d[i] << endl;
    }

    // Iterate through remaining images and add them incrementally
    cout << "reached here" << endl; 
    string img_path = "dataset/Book Statue";
    vector<string> images;
    for (const auto & entry : fs::directory_iterator(img_path))
        images.push_back(entry.path());

    // Remove initial image pair from the list
    images.erase(remove(images.begin(), images.end(), initial_image_pair_paths.first), images.end());
    images.erase(remove(images.begin(), images.end(), initial_image_pair_paths.second), images.end());

    cout << "almost there" << endl; 
    cout << "Number of 3D points: " << points3d.size() << endl;
    string prev_path = initial_image_pair_paths.second;
    string cur_path;
    while(images.size() > 0){
        cur_path = images.back();
        cout << "Processing image: " << cur_path << endl;
        tie(points3d, keypoints1, descriptors1) = incrementalAddition(cur_path, prev_path, K, distortion_coefficients, keypoints1, descriptors1, points3d);
        prev_path = cur_path;
        images.pop_back();
        cout << "Number of 3D points: " << points3d.size() << endl;
    }

    cout << "really there?" << endl; 

    for (int i = 0; i < points3d.size(); i++) {
        cout << "3D point: " << points3d[i] << endl;
    }

    // Create Viz3d window
    cv::viz::Viz3d window("3D Points Visualization");
    window.setWindowSize(cv::Size(3024, 4032));
    window.setBackgroundColor(cv::viz::Color::black());

    // Find the global bounding box of the point cloud
    cv::Point3d min_point(std::numeric_limits<double>::max(), 
                          std::numeric_limits<double>::max(), 
                          std::numeric_limits<double>::max());
    cv::Point3d max_point(std::numeric_limits<double>::lowest(), 
                          std::numeric_limits<double>::lowest(), 
                          std::numeric_limits<double>::lowest());

    for (const auto& pt : points3d) {
        min_point.x = min(min_point.x, static_cast<double>(pt.x));
        min_point.y = min(min_point.y, static_cast<double>(pt.y));
        min_point.z = min(min_point.z, static_cast<double>(pt.z));
        max_point.x = max(max_point.x, static_cast<double>(pt.x));
        max_point.y = max(max_point.y, static_cast<double>(pt.y));
        max_point.z = max(max_point.z, static_cast<double>(pt.z));
    }

    // Calculate the diagonal length of the bounding box
    double diagonal_length = std::sqrt(
        std::pow(max_point.x - min_point.x, 2) +
        std::pow(max_point.y - min_point.y, 2) +
        std::pow(max_point.z - min_point.z, 2)
    );

    // Calculate center of the bounding box
    cv::Point3d center(
        (min_point.x + max_point.x) / 2.0,
        (min_point.y + max_point.y) / 2.0,
        (min_point.z + max_point.z) / 2.0
    );

    // Create point cloud widget with single color
    cv::viz::WCloud cloud_widget(points3d, cv::viz::Color::white());
    cloud_widget.setRenderingProperty(cv::viz::POINT_SIZE, 5.0);

    // Show the point cloud widget
    window.showWidget("point_cloud", cloud_widget);

    // Compute camera distance with a multiplier to ensure all points are visible
    double camera_distance = diagonal_length * 2.0;

    // Set camera pose to encompass all points
    cv::Affine3d camera_pose = cv::viz::makeCameraPose(
        cv::Vec3d(center.x, center.y, center.z + camera_distance),  // Camera position
        cv::Vec3d(center.x, center.y, center.z),  // Look at center
        cv::Vec3d(0, 1, 0)  // Up vector
    );
    window.setViewerPose(camera_pose);
    cv::viz::writeCloud("point_cloud.ply", points3d);
    window.spin();

    return 0;
}