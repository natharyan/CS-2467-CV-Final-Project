#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

struct Descriptor {
    vector<uint8_t> values; // Feature descriptor values (from SIFT or ORB)
};

// Hamming distance between two descriptors
float hammingDistance(const Descriptor &d1, const Descriptor &d2) {
    // Ensure the descriptors have the same size
    if (d1.values.size() != d2.values.size()) {
        return -1;
    }

    // Compute the Hamming distance
    float distance = 0;

    return distance;
}

// Brute force matcher
void bruteForceMatcher(){

}