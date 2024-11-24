#include <iostream> 
#include <vector>
#include <opencv2/opencv.hpp>
// #include "orb.hpp"

using namespace cv;
using namespace std; 

//  aryan's code 
// cv::Mat createBaseImage(const string& imagePath)
// {
//     //create baseimage
//     cv::Mat baseImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
//     //file not found
//     if (baseImage.empty())
//     {
//         cerr << "Error: Could not open or find the image!" << endl;
//         exit(EXIT_FAILURE);
//     }
//     return baseImage;
// }


// TODO: optimize where possible 

// Orientation Assignment 
// Calculating the orientation of each keypoint using the intensity centroid method. This ensures that the features are rotation invariant. 

double calculateOrientationByIntensityCentroid(const Mat& image, const Point& keypoint, int patchSize = 7) {
    // Initialize moments
    double m_10 = 0.0;
    double m_01 = 0.0;
    double m_00 = 0.0;
    // Patch radius
    int radius = patchSize / 2;

    // Get image dimensions
    int rows = image.rows;
    int cols = image.cols;

    // Loop over the patch around the keypoint to compute moments
    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
            int x = keypoint.x + dx;
            int y = keypoint.y + dy;

            if (x >= 0 && x < cols && y >= 0 && y < rows) {
                int weight = image.at<uchar>(y, x);  // Pixel intensity as weight
                // Update moments
                m_10 += dx * weight;
                m_01 += dy * weight;
                m_00 += weight;
            }
        }
    }

    // Compute the orientation using atan2 of the moments
    if (m_00 != 0) {  // To avoid division by zero if m_00 is too small
        return atan2(m_01, m_10);  // Angle in radians
    }
    return 0.0;  // Return 0 if m_00 is 0 (no valid orientation)
}

// FAST 
// TODO: check if checking for all 9 pixels is ok or if you're supposed to only check 3-4 like normal FAST 
vector<Point> detectFAST9Corners(const Mat& image, int threshold) {
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
                    if (dx == 0 && dy == 0) continue;  // Skip the center pixel
                    int nx = i + dx;
                    int ny = j + dy;

                    if (nx < 0 || ny < 0 || nx >= rows || ny >= cols) continue;  // Skip boundary pixels
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
                keypoints.push_back(Point(j, i));  // Store the keypoint (x, y)
            }
            // TODO: After this - CALCULATE THE ORIENTATION OF THE KEYPOINT (how to best do this - see - if here or at the end with ORB)
        }
    }
    
    return keypoints;
}


// Harris Corner Response   
// Calculate the Harris corner response for each pixel in the image. This is used to detect corners in the image. 
// TOOD: might need to change the const keypoint based on the FAST code + check for invalid keypoint/image size maybe 
double harrisResponse(const Mat& gray, const Point& keypoint, int blockSize = 3, double k = 0.04) {
    int radius = blockSize / 2;
    
    double sumIx2 = 0.0, sumIy2 = 0.0, sumIxy = 0.0;

    // Iterate over a block around the keypoint
    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
            int x = keypoint.x + dx;
            int y = keypoint.y + dy;

            if (x >= 0 && x < gray.cols && y >= 0 && y < gray.rows) {
                int Ix = gray.at<uchar>(y, x + 1) - gray.at<uchar>(y, x - 1);  // Approximate gradient in x
                int Iy = gray.at<uchar>(y + 1, x) - gray.at<uchar>(y - 1, x);  // Approximate gradient in y
                sumIx2 += Ix * Ix;
                sumIy2 += Iy * Iy;
                sumIxy += Ix * Iy;
            }
        }
    }

    // Compute determinant and trace of M
    double detM = (sumIx2 * sumIy2) - (sumIxy * sumIxy);
    double traceM = sumIx2 + sumIy2;

    // Harris corner response formula
    return detM - k * (traceM * traceM);
}

// BRIEF 
// TODO: 31 as patch size was used in the paper, see reason + other references 
vector<uchar> BRIEF(const Mat& image, const Point& keypoint, int patchSize = 31) {
    // Extract a patch around the keypoint
    int radius = patchSize / 2;
    // rectangle patch 
    Rect patchRect(keypoint.x - radius, keypoint.y - radius, patchSize, patchSize);

    // Ensure the patch is within bounds
    if (patchRect.x < 0 || patchRect.y < 0 || patchRect.x + patchRect.width >= image.cols || patchRect.y + patchRect.height >= image.rows) {
        return vector<uchar>();  // TODO: Return empty descriptor if patch is out of bounds - check this 
    }

    Mat patch = image(patchRect);
    vector<uchar> descriptor;

    // // TODO: Generate binary descriptor by comparing pairs of pixels 
    // // assuming they are x1,y1,x2,y2

    //     // Ensure the coordinates are within the patch size
    //     something like - if (x1 < patchSize && y1 < patchSize && x2 < patchSize && y2 < patchSize) {
    //         // Compare pixel intensities at the two positions

    //         // Store the result of the comparison as a bit in the descriptor
    //         descriptor.push_back((pixel1 < pixel2) ? 0 : 1);  // If pixel1 < pixel2, store 0, else 1
    //     }
    // }

    return descriptor;
}

// TODO: 
// Steerable BRIEF
// for rotation invariance 

// PCA
// example: 
// Input data matrix as a cv::Mat
// PCA pca(data, Mat(), PCA::DATA_AS_ROW, 2); // Reduce to 2 components
// Mat reduced = pca.project(data); 

// check 
// int main(){
//     cout << "Hello World!" <<endl; 
//     return 0; 
// }