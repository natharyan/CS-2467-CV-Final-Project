#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

using namespace std;

//extrema detection
//create base image
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

//get the number of octaves
int getNumOctaves(const cv::Mat& baseImage, const int num_intervals = 3){
    return abs(int(round(log2(min(baseImage.rows, baseImage.cols)))) - 2); // -2 heuristic added as per the opencv documentation
}

//generate gaussian kernels
vector<float> genGaussianSigmas(const double sigma=1.3, const int num_intervals=3)
{
    vector<float> kernel_sigmas;
    int num_img_per_octave = num_intervals + 3;
    float k = pow(2.0, 1.0 / num_intervals);
    kernel_sigmas.push_back(sigma);
    kernel_sigmas.push_back((pow((sigma*k),2) - pow(sigma, 2)));
    //generate list of gaussian kernels
    for (int i = 0; i < num_img_per_octave-2; i++)
    {
        kernel_sigmas.push_back(pow(k, i) * kernel_sigmas.back());
    }
    return kernel_sigmas;
}

//build gaussian pyramid
vector<vector<cv::Mat>> buildGausPyramid(cv::Mat baseImage, const int num_octaves, const vector<float>& sigmas, const int num_intervals=3)
{ 
    vector<vector<cv::Mat>> gausPyramid;
    for (int i = 0; i < num_octaves; i++)
    {
        vector<cv::Mat> cur_octave;
        cur_octave.push_back(baseImage);
        for (int j = 0; j < num_intervals+2; j++)
        {
            cv::Mat gausImage;
            //Size(0,0) means that the size of the kernel is computed from the sigmaX and sigmaY
            cv::GaussianBlur(baseImage, gausImage, cv::Size(0, 0), sigmas[j], sigmas[j]);
            cur_octave.push_back(gausImage);
        }
        gausPyramid.push_back(cur_octave);
        //3rd from the last image of the current octave is the base image for the next octave
        cv::resize(cur_octave[num_intervals-3], baseImage, cv::Size(cur_octave[num_intervals-3].cols/2, cur_octave[num_intervals-3].rows/2), 0, 0, cv::INTER_NEAREST);
    }
    return gausPyramid;
}

//build Difference of Gaussian Pyramid
vector<vector<cv::Mat>> buildDoGPyramid(const vector<vector<cv::Mat>> gaussPyramid, const int num_intervals = 3){
    vector<vector<cv::Mat>> DoGPyramid;
    for(int i = 0; i < gaussPyramid.size(); i++){
        vector<cv::Mat> cur_octave_DoG;
        vector<cv::Mat> cur_octave = gaussPyramid[i];
        for(int j = 0; j < num_intervals+1; j++){
            cv::Mat DoGImage = cur_octave[j+1] - cur_octave[j];
            cur_octave_DoG.push_back(DoGImage);
        }
        DoGPyramid.push_back(cur_octave_DoG);
    }
    return DoGPyramid;
}

//Central difference for gradient computation
cv::Mat computeGradient(const cv::Mat& patchCube) {
    cv::Mat grad(1, 3, CV_32F);
    grad.at<float>(0) = (patchCube.at<float>(2, 1, 1) - patchCube.at<float>(0, 1, 1)) * 0.5; // dx
    grad.at<float>(1) = (patchCube.at<float>(1, 2, 1) - patchCube.at<float>(1, 0, 1)) * 0.5; // dy
    grad.at<float>(2) = (patchCube.at<float>(1, 1, 2) - patchCube.at<float>(1, 1, 0)) * 0.5; // ds
    return grad;
}

//Hessian computation
cv::Mat computeHessian(const cv::Mat& patchCube) {
    cv::Mat hessian = cv::Mat::zeros(3, 3, CV_32F);
    float val = patchCube.at<float>(1, 1, 1);

    // Second derivatives
    hessian.at<float>(0, 0) = patchCube.at<float>(2, 1, 1) - 2 * val + patchCube.at<float>(0, 1, 1); // dxx
    hessian.at<float>(1, 1) = patchCube.at<float>(1, 2, 1) - 2 * val + patchCube.at<float>(1, 0, 1); // dyy
    hessian.at<float>(2, 2) = patchCube.at<float>(1, 1, 2) - 2 * val + patchCube.at<float>(1, 1, 0); // dss

    // Mixed derivatives
    hessian.at<float>(0, 1) = hessian.at<float>(1, 0) = 0.25f * ((patchCube.at<float>(2, 2, 1) - patchCube.at<float>(2, 0, 1)) -
                                                                 (patchCube.at<float>(0, 2, 1) - patchCube.at<float>(0, 0, 1)));
    hessian.at<float>(0, 2) = hessian.at<float>(2, 0) = 0.25f * ((patchCube.at<float>(2, 1, 2) - patchCube.at<float>(2, 1, 0)) -
                                                                 (patchCube.at<float>(0, 1, 2) - patchCube.at<float>(0, 1, 0)));
    hessian.at<float>(1, 2) = hessian.at<float>(2, 1) = 0.25f * ((patchCube.at<float>(1, 2, 2) - patchCube.at<float>(1, 2, 0)) -
                                                                 (patchCube.at<float>(1, 0, 2) - patchCube.at<float>(1, 0, 0)));

    return hessian;
}

//Check for extrema
bool isExtremum(const cv::Mat& patchCube, float threshold) {
    float centerValue = patchCube.at<float>(1, 1, 1);
    if (std::fabs(centerValue) < threshold) return false;

    bool isMax = true, isMin = true;
    for (int z = 0; z < 3 && (isMax || isMin); z++) {
        for (int y = 0; y < 3 && (isMax || isMin); y++) {
            for (int x = 0; x < 3 && (isMax || isMin); x++) {
                if (z == 1 && y == 1 && x == 1) continue; // Skip center
                float value = patchCube.at<float>(z, y, x);
                isMax &= (centerValue > value);
                isMin &= (centerValue < value);
            }
        }
    }
    return isMax || isMin;
}

//Refining extrema
cv::Point3f refineExtrema(const cv::Mat& patchCube, int maxIterations, float contrastThreshold) {
    cv::Point3f refined(-1, -1, -1); // Default: invalid
    cv::Mat gradient, hessian;
    cv::Mat offset;

    for (int iter = 0; iter < maxIterations; ++iter) {
        gradient = computeGradient(patchCube);
        hessian = computeHessian(patchCube);

        offset = -hessian.inv(cv::DECOMP_SVD) * gradient.t();
        if (cv::norm(offset) < 0.5) break;

        //If offset is too large or invalid, return failure
        if (cv::norm(offset) > 1.0f) return refined;
    }

    //Check contrast after interpolation
    float interpolatedContrast = patchCube.at<float>(1, 1, 1) + 0.5f * gradient.dot(offset);
    if (std::abs(interpolatedContrast) < contrastThreshold) return refined;

    //Successful refinement
    refined.x = offset.at<float>(0);
    refined.y = offset.at<float>(1);
    refined.z = offset.at<float>(2);
    return refined;
}

//Scale-space extrema detection
vector<cv::KeyPoint> scaleSpaceExtrema(
    const vector<vector<cv::Mat>>& DoGPyramid, int numIntervals, float contrastThreshold, int imageBorder) {
    vector<cv::KeyPoint> keypoints;
    float prelimContrastThreshold = 0.5f * (contrastThreshold / numIntervals) / 255.0f;

    for (int octaveIdx = 0; octaveIdx < DoGPyramid.size(); ++octaveIdx) {
        const auto& octave = DoGPyramid[octaveIdx];
        for (int imgIdx = 1; imgIdx < octave.size() - 1; ++imgIdx) {
            for (int y = imageBorder; y < octave[imgIdx].rows - imageBorder; ++y) {
                for (int x = imageBorder; x < octave[imgIdx].cols - imageBorder; ++x) {
                    if (x - 1 < 0 || x + 1 >= octave[imgIdx].cols || y - 1 < 0 || y + 1 >= octave[imgIdx].rows) {
                        continue;
                    }

                    cv::Mat patchCube = octave[imgIdx](cv::Rect(x - 1, y - 1, 3, 3)).clone();
                    if (patchCube.rows != 3 || patchCube.cols != 3 || patchCube.type() != CV_32F) {
                        continue;
                    }

                    if (isExtremum(patchCube, prelimContrastThreshold)) {
                        cv::Point3f refined = refineExtrema(patchCube, 10, contrastThreshold);
                        if (refined.x != -1) {
                            cv::KeyPoint kp;
                            kp.pt = cv::Point2f(x + refined.x, y + refined.y);
                            kp.octave = octaveIdx;
                            kp.response = patchCube.at<float>(1, 1);
                            keypoints.push_back(kp);
                        }
                    }
                }
            }
        }
    }
    return keypoints;
}


int main()
{
    string imgpath = "../dataset/nurmahal/nuramal.jpg";
    cv::Mat baseImage;
    baseImage = createBaseImage(imgpath);
    cout << "SIFT running..." << endl;
    cv::imshow("Base Image", baseImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
    //get the number of octaves
    int num_octaves = getNumOctaves(baseImage);
    cout << "Number of octaves: " << num_octaves << endl;
    //generate gaussian kernels
    vector<float> sigmas = genGaussianSigmas();
    //build gaussian pyramid
    vector<vector<cv::Mat>> gaussPyramid = buildGausPyramid(baseImage, num_octaves, sigmas,3);
    cout << "Number of images in each octave of GausPyramid: " << gaussPyramid[0].size() << endl;
    
    //build Difference of Gaussian Pyramid
    vector<vector<cv::Mat>> DoGPyramid = buildDoGPyramid(gaussPyramid, 3);

    cout << "Number of images in each octave of DoGPyramid: " << DoGPyramid[0].size() << endl;

    //scale-space extrema detection
    vector<cv::KeyPoint> keypoints = scaleSpaceExtrema(DoGPyramid, 3, 0.04, 1);
    cout << "Number of keypoints: " << keypoints.size() << endl;
    
    cv::Mat keypointImage;
    cv::drawKeypoints(baseImage, keypoints, keypointImage);
    cv::imshow("Keypoints", keypointImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cout << "SIFT completed!" << endl;
    return 0;
}
