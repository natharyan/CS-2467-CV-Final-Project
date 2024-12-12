#ifndef RANSAC_H
#define RANSAC_H

#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::JacobiSVD;
using Eigen::ComputeFullU;
using Eigen::ComputeFullV;

using namespace std;

MatrixXd computeFundamentalMatrix(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& matches); 

MatrixXd ransacFundamentalMatrix(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& matches, int maxIterations, double threshold); 

#endif // RANSAC_HPP