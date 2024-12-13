#include "ransac.hpp"
#include <cmath>
#include <ctime>
#include <unordered_set>

using namespace Eigen;
using namespace std;

struct Match {
    double x1, y1, x2, y2;
};

pair<MatrixXd, vector<bool>> ransacFundamentalMatrix( const vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& matches, int maxIterations, double threshold) {
    const int sampleSize = 8;
    const int numMatches = matches.size();
    MatrixXd bestF(3, 3);
    vector<bool> bestInliers(numMatches, false);
    int maxInliers = 0;
    // unsigned int seed = static_cast<unsigned int>(time(nullptr));
    unsigned int seed = 1734056127;
    srand(seed);
    cout << "seed: " << seed << endl;
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Randomly select 8 points
        unordered_set<int> indices;
        while (indices.size() < sampleSize) {
            indices.insert(rand() % numMatches);
        }

        // Build the 8-point matrix
        MatrixXd A(sampleSize, 9);
        int row = 0;
        for (int idx : indices) {
            const auto& match = matches[idx];
            A.row(row++) << match.first.x() * match.second.x(), match.first.y() * match.second.x(), match.second.x(),
                            match.first.x() * match.second.y(), match.first.y() * match.second.y(), match.second.y(),
                            match.first.x(), match.first.y(), 1.0;
        }

        // Compute SVD of A
        JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
        VectorXd f = svd.matrixV().col(8);

        // Reshape f into F
        MatrixXd F(3, 3);
        F << f(0), f(1), f(2),
             f(3), f(4), f(5),
             f(6), f(7), f(8);

        // Enforce rank 2 constraint
        JacobiSVD<MatrixXd> svdF(F, ComputeFullU | ComputeFullV);
        VectorXd s = svdF.singularValues();
        s(2) = 0.0; // Set the smallest singular value to zero
        F = svdF.matrixU() * s.asDiagonal() * svdF.matrixV().transpose();

        // Count inliers
        vector<bool> inliers(numMatches, false);
        int inlierCount = 0;

        for (int i = 0; i < numMatches; ++i) {
            const auto& match = matches[i];
            Vector3d x1(match.first.x(), match.first.y(), 1.0);
            Vector3d x2(match.second.x(), match.second.y(), 1.0);
            double error = x2.transpose() * F * x1;
            double dist = abs(error) / sqrt(pow(F(0, 0) * match.first.x() + F(0, 1) * match.first.y() + F(0, 2), 2) +
                                            pow(F(1, 0) * match.first.x() + F(1, 1) * match.first.y() + F(1, 2), 2));
            if (dist < threshold) {
                inliers[i] = true;
                inlierCount++;
            }
        }

        // Update bestF if more inliers are found
        if (inlierCount > maxInliers) {
            maxInliers = inlierCount;
            bestF = F;
            bestInliers = inliers;
        }
    }

    return {bestF, bestInliers};
}