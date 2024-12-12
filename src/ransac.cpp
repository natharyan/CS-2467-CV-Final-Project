#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::JacobiSVD;
using Eigen::ComputeFullU;
using Eigen::ComputeFullV;

// Helper function: Compute the fundamental matrix using the 8-point algorithm
MatrixXd computeFundamentalMatrix(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& matches) {
    MatrixXd A(matches.size(), 9);
    for (size_t i = 0; i < matches.size(); ++i) {
        double x1 = matches[i].first(0);
        double y1 = matches[i].first(1);
        double x2 = matches[i].second(0);
        double y2 = matches[i].second(1);

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

    return F;
}

// RANSAC to estimate the fundamental matrix
MatrixXd ransacFundamentalMatrix(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& matches, int maxIterations, double threshold) {
    int bestInlierCount = 0;
    MatrixXd bestF;
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
        for (const auto& match : matches) {
            Eigen::Vector3d p1(match.first(0), match.first(1), 1.0);
            Eigen::Vector3d p2(match.second(0), match.second(1), 1.0);
            
            double error = std::abs(p2.transpose() * F * p1);
            if (error < threshold) {
                ++inlierCount;
            }
        }

        if (inlierCount > bestInlierCount) {
            bestInlierCount = inlierCount;
            bestF = F;
        }
    }

    return bestF;
}

int main() {
    // Example matches: pairs of corresponding points in two images
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> matches = {
        {{100, 150}, {110, 160}},
        {{200, 250}, {210, 260}},
        {{300, 350}, {310, 360}},
        {{400, 450}, {410, 460}},
        {{500, 550}, {510, 560}},
        {{600, 650}, {610, 660}},
        {{700, 750}, {710, 760}},
        {{800, 850}, {810, 860}},
        {{900, 950}, {910, 960}},
        {{1000, 1050}, {1010, 1060}}
    };

    int maxIterations = 1000;
    double threshold = 0.01;

    MatrixXd F = ransacFundamentalMatrix(matches, maxIterations, threshold);

    std::cout << "Estimated Fundamental Matrix:\n" << F << std::endl;

    return 0;
}
