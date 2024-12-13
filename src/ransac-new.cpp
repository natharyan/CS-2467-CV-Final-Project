#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::JacobiSVD;
using Eigen::ComputeFullU;
using Eigen::ComputeFullV;

// Helper function: Compute the fundamental matrix using the 8-point algorithm

void normalizePoints(const std::vector<Eigen::Vector2d>& points, std::vector<Eigen::Vector2d>& normalizedPoints, MatrixXd& T) {
    // Compute centroid
    Eigen::Vector2d centroid(0, 0);
    for (const auto& p : points) {
        centroid += p;
    }
    centroid /= points.size(); // Compute v0 (vertex or reference centroid)

    // Compute mean distance (f0)
    double meanDist = 0.0;
    for (const auto& p : points) {
        meanDist += (p - centroid).norm(); // Sum distances from centroid
    }
    meanDist /= points.size(); // Compute f0 (mean distance -> feature or fitted centroid)

    // Compute transformation matrix T (scaling and translation)
    double scale = 1.0 / meanDist;
    T << scale, 0, -scale * centroid(0),
         0, scale, -scale * centroid(1),
         0, 0, 1;

    // Normalize points using the transformation matrix
    normalizedPoints.clear();
    std::vector<Eigen::Vector2d> transformedPoints;
    for (const auto& p : points) {
        Eigen::Vector3d normalizedP = T * Eigen::Vector3d(p(0), p(1), 1.0);
        normalizedP /= normalizedP(2);
        transformedPoints.emplace_back(normalizedP(0), normalizedP(1));
    }

    // Now, ensure the normalized points are scaled to [-1, 1]
    double minX = std::numeric_limits<double>::max(), maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max(), maxY = std::numeric_limits<double>::lowest();

    // Find the bounding box of the transformed points
    for (const auto& p : transformedPoints) {
        minX = std::min(minX, p(0));
        maxX = std::max(maxX, p(0));
        minY = std::min(minY, p(1));
        maxY = std::max(maxY, p(1));
    }

    // Scale the points to fit within [-1, 1]
    double scaleX = 2.0 / (maxX - minX);
    double scaleY = 2.0 / (maxY - minY);
    double offsetX = -(maxX + minX) / (maxX - minX);
    double offsetY = -(maxY + minY) / (maxY - minY);

    for (const auto& p : transformedPoints) {
        Eigen::Vector2d scaledP;
        scaledP(0) = scaleX * (p(0) + offsetX); // Map x to [-1, 1]
        scaledP(1) = scaleY * (p(1) + offsetY); // Map y to [-1, 1]
        normalizedPoints.push_back(scaledP);
    }
}

MatrixXd computeFundamentalMatrix(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& matches) {
    std::vector<Eigen::Vector2d> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(match.first);
        points2.push_back(match.second);
    }

    std::vector<Eigen::Vector2d> normPoints1, normPoints2;
    MatrixXd T1(3, 3), T2(3, 3);
    normalizePoints(points1, normPoints1, T1);
    normalizePoints(points2, normPoints2, T2);

    MatrixXd A(matches.size(), 9);
    for (size_t i = 0; i < matches.size(); ++i) {
        double x1 = normPoints1[i](0);
        double y1 = normPoints1[i](1);
        double x2 = normPoints2[i](0);
        double y2 = normPoints2[i](1);

        A(i, 0) = x1 * x2;
        A(i, 1) = x1 * y2;
        A(i, 2) = x1;
        A(i, 3) = y1 * x2;
        A(i, 4) = y1 * y2;
        A(i, 5) = y1;
        A(i, 6) = x2;
        A(i, 7) = y2;
        A(i, 8) = 1.0;
    }

    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    VectorXd f = svd.matrixV().col(8);

    MatrixXd F(3, 3);
    F << f(0), f(1), f(2),
         f(3), f(4), f(5),
         f(6), f(7), f(8);

    JacobiSVD<MatrixXd> svdF(F, ComputeFullU | ComputeFullV);
    VectorXd singularValues = svdF.singularValues();
    singularValues(2) = 0; // Enforce rank 2
    F = svdF.matrixU() * singularValues.asDiagonal() * svdF.matrixV().transpose();

    // Denormalize the fundamental matrix
    F = T2.transpose() * F * T1;

    return F;
}

// RANSAC to estimate the fundamental matrix
std::pair<MatrixXd, std::vector<bool>> ransacFundamentalMatrix(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& matches, int maxIterations, double threshold) {
    int bestInlierCount = 0;
    MatrixXd bestF;
    std::vector<bool> bestInliers(matches.size(), false);
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int iter = 0; iter < maxIterations; ++iter) {
        std::uniform_int_distribution<> dis(0, matches.size() - 1);

        // Randomly sample 8 matches
        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> sample;
        for (int i = 0; i < 8; ++i) {
            sample.push_back(matches[dis(gen)]);
        }

        MatrixXd F = computeFundamentalMatrix(sample);

        // Count inliers
        int inlierCount = 0;
        std::vector<bool> inliers(matches.size(), false);
        for (size_t i = 0; i < matches.size(); ++i) {
            const auto& match = matches[i];
            Eigen::Vector3d p1(match.first(0), match.first(1), 1.0);
            Eigen::Vector3d p2(match.second(0), match.second(1), 1.0);
            
            double error = std::abs(p2.transpose() * F * p1);
            if (error < threshold) {
                ++inlierCount;
                inliers[i] = true;
            }
        }

        if (inlierCount > bestInlierCount) {
            bestInlierCount = inlierCount;
            bestF = F;
            bestInliers = inliers;
        }
    }

    return {bestF, bestInliers};
}


// int main() {
//     // Define matches as pairs of Eigen::Vector2d
//     std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> matches = {
//         {{100, 150}, {110, 160}},
//         {{200, 250}, {210, 260}},
//         {{300, 350}, {310, 360}},
//         {{400, 450}, {410, 460}},
//         {{500, 550}, {510, 560}},
//         {{600, 650}, {610, 660}},
//         {{700, 750}, {710, 760}},
//         {{800, 850}, {810, 860}},
//         {{900, 950}, {910, 960}},
//         {{1000, 1050}, {1010, 1060}}
//     };
    
    
//     int num_points = 13;

//     std::vector<cv::Point2f> pts1(num_points);
//     std::vector<cv::Point2f> pts2(num_points);

//     cv::RNG rng(0);

//     for (int i = 0; i < 8; i++) {
//         pts1[i] = cv::Point2f(rng.uniform(0.f, 1000.f), rng.uniform(0.f, 1000.f));
//         pts2[i] = pts1[i] + cv::Point2f(rng.uniform(-5.f, 5.f), rng.uniform(-5.f, 5.f)); // Adding some noise
//     }

//     // Add 5 outliers (random points far from the inlier region)
//     for (int i = 8; i < num_points; i++) {
//         pts1[i] = cv::Point2f(rng.uniform(500.f, 1000.f), rng.uniform(500.f, 1000.f));
//         pts2[i] = cv::Point2f(rng.uniform(500.f, 1000.f), rng.uniform(500.f, 1000.f));
//     }


//     std::vector<uchar> inlierss;
//     cv::Mat fundamentalMatrix = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3, 0.99, inlierss);

//     // Show which points are inliers (1) and outliers (0)
//     std::cout << "Inliers Mask: ";
//     for (size_t i = 0; i < inlierss.size(); i++) {
//         std::cout << static_cast<int>(inlierss[i]) << " ";
//     }
//     std::cout << std::endl;

//     int maxIterations = 2000;
//     double threshold = 0.01; // Adjusted to a more realistic value for RANSAC
//     double confidence = 0.99; // Confidence level for RANSAC

//     // Convert matches to cv::Mat
//     for (const auto& match : matches) {
//         pts1.emplace_back(static_cast<float>(match.first(0)), static_cast<float>(match.first(1)));
//         pts2.emplace_back(static_cast<float>(match.second(0)), static_cast<float>(match.second(1)));
//     }

//     std::vector<Eigen::Vector2d> normPoints1, normPoints2;
//     MatrixXd T1(3, 3), T2(3, 3); // Initialize T1 and T2 as 3x3 matrices

//     std::vector<Eigen::Vector2d> eigenPts1, eigenPts2;
//     for (const auto& pt : pts1) {
//         eigenPts1.emplace_back(pt.x, pt.y);
//     }
//     for (const auto& pt : pts2) {
//         eigenPts2.emplace_back(pt.x, pt.y);
//     }

//     normalizePoints(eigenPts1, normPoints1, T1);
//     normalizePoints(eigenPts2, normPoints2, T2);

//     // print normPoints1 
//     std::cout << "Normalized Points 1:\n";
//     for (const auto& p : normPoints1) {
//         std::cout << "(" << p(0) << ", " << p(1) << ")\n";
//     }

//     // Print matches for debugging
//     std::cout << "Matches:\n";
//     for (size_t i = 0; i < matches.size(); ++i) {
//         std::cout << "Match " << i << ": (" << matches[i].first(0) << ", " << matches[i].first(1) << ") -> ("
//                   << matches[i].second(0) << ", " << matches[i].second(1) << ")\n";
//     }


//     // Find the fundamental matrix using RANSAC
//     // cv::Mat fundamentalMatrix = cv::findFundamentalMat(points3, points4, cv::FM_RANSAC, threshold, confidence, maxIterations);

//     std::cout << "Fundamental Matrix (OpenCV):\n" << fundamentalMatrix << std::endl;
//     // Print the fundamental matrix
//     if (!fundamentalMatrix.empty()) {
//         std::cout << "Fundamental Matrix (OpenCV):\n" << fundamentalMatrix << std::endl;
//     } // else {
//     //     std::cout << "Could not compute the fundamental matrix (OpenCV)." << std::endl;
//     // }

//     // Find the fundamental matrix using custom RANSAC
//     auto [F, inliers] = ransacFundamentalMatrix(matches, maxIterations, threshold);

//     if (F.size() != 0) {
//         std::cout << "Fundamental Matrix (Custom RANSAC):\n" << F << std::endl;
//         std::cout << "Inliers Mask (Custom RANSAC): ";
//         for (bool inlier : inliers) {
//             std::cout << inlier << " ";
//         }
//         std::cout << std::endl;
//     } else {
//         std::cout << "Could not compute the fundamental matrix (Custom RANSAC)." << std::endl;
//     }

//     return 0;
// }