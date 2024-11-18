#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <pair>
#include <cmath>

using namespace std;

// conv int descriptor to binary
vector<string> desctoBinary(vector<int> descSingle){
    vector<string> bin_desc;
    for(int i = 0; i < descSingle.size(); i++){
        int val = descSingle[i];
        string bin_val = "";
        // convert int to binary
        while(val > 0){
            bin_val += to_string(val % 2) + '0';
            val /= 2;
        }
        // pad with zeros
        while(bin_val.size() < 8){
            bin_val += '0';
        }
        // reverse the string
        reverse(bin_val.begin(), bin_val.end());
        bin_desc.push_back(bin_val);
    }
    return bin_desc;
}

// get hamming distance
int calHammingDistance(vector<string> descBin1, vector<string> descBin2){
    int hamming_dist = 0;
    int loop_range;
    if(descBin1.size() != descBin2.size()){
        loop_range = min(descBin1.size(),descBin2.size());
    }else{
        loop_range = descBin1.size();
    }
    for(int i = 0; i < loop_range; i++){
        int cur_indx_hamming_distance = 0;
        for(int j = 0; j < 8; j++){
            if(descBin1[i][j] != descBin2[i][j]){
                cur_indx_hamming_distance += 1;
            }
        }
        hamming_dist += cur_indx_hamming_distance;
    }
    return hamming_dist;
}

// get feature matches - and corresponding keypoint matches, with threshold on the hamming distance
vector<pair(int,int)> getMatches_Keypoints()


int main(){
    // get orb keypoints and descriptors

    cv::Mat img1 = cv::imread("../dataset/nurmahal/Nurmahal_Sarai_Mughal_Heritage_Punjab_India.JPG",cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("../dataset/nurmahal/Srai_Nurmahal_Jalandhar,Punjab,India_02.jpg",cv::IMREAD_GRAYSCALE);

    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    // print keypoints and decsriptors

    cout << "keypoints1: ";
    for(int i = 0; i < 5 && i < keypoints1.size(); i++){
        cout << keypoints1[i].pt << " ";
    }
    cout << endl;

    cout << "descriptors1: ";
    for(int i = 0; i < 5 && i < descriptors1.rows; i++){
        cout << descriptors1.row(i) << endl;
    }

    cout << endl;

    cout << keypoints1.size() << " " << descriptors1.rows << endl;

    vector<int> descp_first = descriptors1.row(0);
    cout << "bin_descriptor1[0]: " << desctoBinary(descp_first) << endl;
    // get the matches

}