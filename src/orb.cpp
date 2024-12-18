#include <iostream> 
#include <vector>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
// #include "orb.hpp"

using namespace cv;
using namespace std; 

// Bit patterns from opencv/modules/features2d/src/orb.cpp for faster computation


// create base image 
cv::Mat createBaseImage(const string& imagePath)
{
    //create baseimage
    cv::Mat baseImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    //file not found
    if (baseImage.empty())
    {
        cerr << "Error: Could not open or find the image!" << endl;
        exit(EXIT_FAILURE);
    }
    return baseImage;
}

// FAST to detect keypoints

vector<cv::KeyPoint> FAST9(const Mat& image, int threshold){
    vector<cv::KeyPoint> keypoints;
    // Convert image to grayscale if it's not already
    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image;
    }
    
    int rows = gray.rows;
    int cols = gray.cols;
    
    // Loop over all pixels in the image (excluding borders)
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            int intensity = gray.at<uchar>(i, j);
            int countBright = 0;
            int countDark = 0;

            // Check 9 surrounding pixels (3 to the left, 3 to the right, 2 above, and 1 below)
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    // Skip the center pixel
                    if (dx == 0 && dy == 0) continue;  
                    int nx = i + dx;
                    int ny = j + dy;

                    // Skip boundary pixels
                    if (nx < 0 || ny < 0 || nx >= rows || ny >= cols) continue;  
                    int neighborIntens = gray.at<uchar>(nx, ny);

                    if (neighborIntens > intensity + threshold) {
                        countBright++;
                    } else if (neighborIntens < intensity - threshold) {
                        countDark++;
                    }
                }
            }

            // If 3 or more pixels are brighter or darker than the center, mark it as a corner
            if (countBright >= 3 || countDark >= 3) {
                // Store the keypoint (x, y)
                keypoints.push_back(cv::KeyPoint(j, i, 1.0));  
            }
        }
    }
    
    return keypoints;
} 

// Orientation assignment 

double orientationAssignment(const Mat& image, const KeyPoint& keypoint, int patchSize = 7) {
    // Initialize moments
    double m_10 = 0.0;
    double m_01 = 0.0;
    double m_00 = 0.0;

    int radius = patchSize / 2;
    int rows = image.rows;
    int cols = image.cols;

    // Loop over the patch around the keypoint to compute moments
    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
            int x = keypoint.pt.x + dx;
            int y = keypoint.pt.y + dy;

            if (x >= 0 && x < cols && y >= 0 && y < rows) {
                // Pixel intensity as weight
                int weight = image.at<uchar>(y, x);  
                // Updating moments
                m_10 += dx * weight;
                m_01 += dy * weight;
                m_00 += weight;
            }
        }
    }

    // Compute the orientation using atan2 of the moments
    if (m_00 != 0) {  
        return atan2(m_01, m_10); 
    }
    return 0.0;  
}

// Harris 
double harrisResponse(const Mat& gray, const cv::KeyPoint& keypoint, int blockSize = 3, double k = 0.04) {
    int radius = blockSize / 2;
    double sumIx2 = 0.0, sumIy2 = 0.0, sumIxy = 0.0;

    // Optional: Gaussian weighting
    double sigma = blockSize / 6.0;
    double gaussianNorm = 0.0;

    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
            int x = keypoint.pt.x + dx;
            int y = keypoint.pt.y + dy;

            if (x >= 1 && x < gray.cols - 1 && y >= 1 && y < gray.rows - 1) {
                // Floating-point gradients for more precision
                double Ix = static_cast<double>(gray.at<uchar>(y, x + 1) - gray.at<uchar>(y, x - 1));
                double Iy = static_cast<double>(gray.at<uchar>(y + 1, x) - gray.at<uchar>(y - 1, x));

                // Optional Gaussian weighting
                double gaussian = exp(-(dx*dx + dy*dy) / (2 * sigma * sigma));
                gaussianNorm += gaussian;

                sumIx2 += gaussian * Ix * Ix;
                sumIy2 += gaussian * Iy * Iy;
                sumIxy += gaussian * Ix * Iy;
            }
        }
    }

    // Normalize by Gaussian weights if used
    if (gaussianNorm > 0) {
        sumIx2 /= gaussianNorm;
        sumIy2 /= gaussianNorm;
        sumIxy /= gaussianNorm;
    }

    // Compute determinant and trace of the covariance matrix M
    double detM = (sumIx2 * sumIy2) - (sumIxy * sumIxy);
    double traceM = sumIx2 + sumIy2;
    
    return detM - k * (traceM * traceM);
}

// rBRIEF 
cv::Mat rBRIEF(const Mat& image, vector<cv::KeyPoint>& keypoints, int patchSize = 31) {

    Ptr<ORB> orb = ORB::create(500);
    Mat descriptors;
    

    Ptr<xfeatures2d::BriefDescriptorExtractor> briefExtractor = xfeatures2d::BriefDescriptorExtractor::create(32);

    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    vector<cv::KeyPoint> cvKeypoints;
    for (const auto& point : keypoints) {
        KeyPoint kp(point.pt.x, point.pt.y, patchSize, point.angle); // Use the orientation
        cvKeypoints.push_back(kp);
    }

    // opencv check 
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    Mat descriptorsMat;
    briefExtractor->compute(gray, cvKeypoints, descriptorsMat);

    if (descriptorsMat.empty()) {
        cerr << "Error: No descriptors were computed." << endl;
    }

    return descriptors;
}

struct KeypointWithResponse {
    cv::KeyPoint point;
    double harrisResponse;
};

// int main(){
//     string imgpath = "../dataset/Book Statue/WhatsApp Image 2024-11-25 at 19.01.18 (1).jpeg";
//     cv::Mat baseImage;
//     baseImage = createBaseImage(imgpath);
//     cout << "SIFT running..." << endl;
//     // ...existing code...
//     vector<cv::KeyPoint> initialKeypoints = FAST9(baseImage, 20);
//     cout << "Number of keypoints detected by FAST9: " << initialKeypoints.size() << endl;

//     vector<KeypointWithResponse> keypointsWithResponses;
//     for (const auto& kp : initialKeypoints) {
//         double response = harrisResponse(baseImage, kp);
//         if (response > 0.01) { 
//             keypointsWithResponses.push_back({kp, response});
//         }
//     }

//     cout << keypointsWithResponses.size() << endl; 

//     sort(keypointsWithResponses.begin(), keypointsWithResponses.end(), [](const KeypointWithResponse& a, const KeypointWithResponse& b) {
//         return a.harrisResponse < b.harrisResponse;
//     });

//     const int maxKeypoints =1000;
//     if (keypointsWithResponses.size() > maxKeypoints) {
//         keypointsWithResponses.resize(maxKeypoints);
//     }

//     cout << "Number of keypoints after Harris filtering: " << keypointsWithResponses.size() << endl;

//     vector<cv::KeyPoint> filteredKeypoints;
//     vector<double> orientations;
//     for (const auto& kpWithResponse : keypointsWithResponses) {
//         cv::KeyPoint kp = kpWithResponse.point;
//         kp.angle = orientationAssignment(baseImage, kp);
//         filteredKeypoints.push_back(kp);
//     }
//     cout << filteredKeypoints.size() << endl; 

//     cv::Mat descriptors = rBRIEF(baseImage, filteredKeypoints, 31);
//     cout << "Descriptors computed using rBRIEF." << endl;

//     cv::Mat displayImage;
//     cvtColor(baseImage, displayImage, COLOR_GRAY2BGR);  

//     Mat imgWithKeypoints;
//     drawKeypoints(baseImage, filteredKeypoints, imgWithKeypoints, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);

//     imshow("Keypoints", imgWithKeypoints);
//     waitKey(0);

//     cout << "Descriptors (first 5 descriptors):" << endl;
//     for (int i = 0; i < min(descriptors.rows, 5); ++i) {
//         cout << "Descriptor " << i << ": " << descriptors.row(i) << endl;
//     }

//     imshow("Filtered Keypoints with Orientations", displayImage);
//     waitKey(0);
//     destroyAllWindows();

//     cout << "ORB pipeline completed successfully." << endl;
//     return 0;
// }