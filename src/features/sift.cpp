#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>

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



int main()
{
    //how can i get the base image in a variable
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
    //build Difference of Gaussian Pyramid
    vector<vector<cv::Mat>> DoGPyramid = buildDoGPyramid(gaussPyramid, 3);
    cout << "SIFT completed!" << endl;
    return 0;
}
