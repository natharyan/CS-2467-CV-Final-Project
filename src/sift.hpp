#ifndef SIFT_HEADER
#define SIFT_HEADER

#include <array>

using namespace std;

template <typename Tp, size_t InnerSize, size_t OuterSize>

class Sift
{
    public:
    void createBaseImag();
    void computerNumOctaves();
    void generateGaussianKernels();
    void buildGaussianPyramid();
    void buildDoGPyramid();
    void scqleSpaceExtrema();
    void refineExtrema();
    void interpolateContrast();
    void interpolateStep();
    array<array<Tp, InnerSize>, OuterSize> hessian3D();
    bool isExtremum();
};

// check normalized cross correlation at some point - for different intensity images.

#endif
