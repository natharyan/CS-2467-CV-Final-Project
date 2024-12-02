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

vector<Point> FAST9(const Mat& image, int threshold){
    vector<Point> keypoints;
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
                keypoints.push_back(Point(j, i));  
            }
        }
    }
    
    return keypoints;
} 

// Orientation assignment 

double orientationAssignment(const Mat& image, const Point& keypoint, int patchSize = 7) {
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
            int x = keypoint.x + dx;
            int y = keypoint.y + dy;

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

// Harris Corner Response 
double harrisResponse(const Mat& gray, const Point& keypoint, int blockSize = 3, double k = 0.04) {
    int radius = blockSize / 2;
    
    double sumIx2 = 0.0, sumIy2 = 0.0, sumIxy = 0.0;

    // Iterate over a block around the keypoint
    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
            int x = keypoint.x + dx;
            int y = keypoint.y + dy;

            if (x >= 0 && x < gray.cols && y >= 0 && y < gray.rows) {
                // Gradient calculation 
                int Ix = gray.at<uchar>(y, x + 1) - gray.at<uchar>(y, x - 1);  
                int Iy = gray.at<uchar>(y + 1, x) - gray.at<uchar>(y - 1, x);  
                sumIx2 += Ix * Ix;
                sumIy2 += Iy * Iy;
                sumIxy += Ix * Iy;
            }
        }
    }

    // Compute determinant and trace of M
    double detM = (sumIx2 * sumIy2) - (sumIxy * sumIxy);
    double traceM = sumIx2 + sumIy2;
    return detM - k * (traceM * traceM);
}

// rBRIEF 
vector<Mat> rBRIEF(const Mat& image, const vector<Point>& keypoints, int patchSize = 31) {
    vector<Mat> descriptors;

    // Create a binary pattern generator (random sampling of pixel pairs)
    // The size (32 bytes = 256 bits) is set in the constructor
    Ptr<xfeatures2d::BriefDescriptorExtractor> briefExtractor = xfeatures2d::BriefDescriptorExtractor::create(32);

    // Convert the image to a format suitable for keypoint extraction (e.g., grayscale)
    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // Detect keypoints using FAST (or you can pass pre-detected keypoints)
    vector<KeyPoint> cvKeypoints;
    for (const auto& point : keypoints) {
        KeyPoint kp(point.x, point.y, patchSize); // Assuming a patch size, can be changed
        cvKeypoints.push_back(kp);
    }

    // Extract BRIEF descriptors for the keypoints
    Mat descriptorsMat;
    briefExtractor->compute(gray, cvKeypoints, descriptorsMat);

    // Check if the descriptors were successfully computed
    if (!descriptorsMat.empty()) {
        // Store the descriptors in a vector
        for (int i = 0; i < descriptorsMat.rows; ++i) {
            descriptors.push_back(descriptorsMat.row(i));
        }
    } else {
        cerr << "Error: No descriptors were computed." << endl;
    }

    return descriptors;
}


struct KeypointWithResponse {
    Point point;
    double harrisResponse;
};

int main(){
    string imgpath = "/Users/manyagarggg/Desktop/Screenshot 2024-11-24 at 3.35.31 PM.png";
    cv::Mat baseImage;
    baseImage = createBaseImage(imgpath);
    cout << "SIFT running..." << endl;
    cv::imshow("Base Image", baseImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
    // calling FAST9 
    cout << "running" << endl;
    vector<Point> initialKeypoints = FAST9(baseImage, 20);
    cout << "Number of keypoints detected by FAST9: " << initialKeypoints.size() << endl;

    // harris corner response 
    vector<KeypointWithResponse> keypointsWithResponses;
    for (const Point& kp : initialKeypoints) {
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
    vector<Point> filteredKeypoints;
    vector<double> orientations;
    for (const auto& kpWithResponse : keypointsWithResponses) {
        filteredKeypoints.push_back(kpWithResponse.point);
        orientations.push_back(orientationAssignment(baseImage, kpWithResponse.point));
    }

    vector<KeyPoint> cvKeypoints;
    for (const auto& point : filteredKeypoints) {
        KeyPoint kp(point.x, point.y, 31); // Set patch size (31 as example)
        cvKeypoints.push_back(kp);
    }

    // Compute rBRIEF descriptors
    vector<Mat> descriptors = rBRIEF(baseImage, filteredKeypoints, 31);
    cout << "Descriptors computed using rBRIEF." << endl;

    // Visualize the keypoints
    cv::Mat displayImage;
    cvtColor(baseImage, displayImage, COLOR_GRAY2BGR);  

    // Step 5: Display keypoints on the image
    Mat imgWithKeypoints;
    drawKeypoints(baseImage, cvKeypoints, imgWithKeypoints, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);

    // Show the image with keypoints
    imshow("Keypoints", imgWithKeypoints);
    waitKey(0);

    // Print the descriptors
    cout << "Descriptors (first 5 descriptors):" << endl;
    for (size_t i = 0; i < min(descriptors.size(), (size_t)5); ++i) {
        cout << "Descriptor " << i << ": " << descriptors[i] << endl;
    }
    // cout << descriptors.size() << endl;
    // cout << "done" << endl;

    imshow("Filtered Keypoints with Orientations", displayImage);
    waitKey(0);
    destroyAllWindows();

    cout << "ORB pipeline completed successfully." << endl;
    return 0;
}
