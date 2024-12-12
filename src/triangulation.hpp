#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <climits> 
#include <cmath>
#include "epipolar.hpp"

using namespace std;

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <climits> 
#include <cmath>
#include "epipolar.hpp"
using namespace std;

// reference: https://staff.fnwi.uva.nl/r.vandenboomgaard/ComputerVision/LectureNotes/CV/StereoVision/triangulation.html

cv::Mat computeProjection(const cv::Mat& K, const cv::Mat& R, const cv::Mat& T);

std::vector<cv::Point3f> triangulation(const cv::Mat& P1, const cv::Mat& P2, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2);

#endif // TRIANGULATION_H