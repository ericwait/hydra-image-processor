#pragma once
#include "Vec.h"
#include "CudaDeviceStats.h"
#include "ImageContainer.h"


#ifdef IMAGE_PROCESSOR_DLL
#ifdef IMAGE_PROCESSOR_INTERNAL
#define IMAGE_PROCESSOR_API __declspec(dllexport)
#else
#define IMAGE_PROCESSOR_API __declspec(dllimport)
#endif // IMAGE_PROCESSOR_INTERNAL
#else
#define IMAGE_PROCESSOR_API
#endif // IMAGE_PROCESSOR_DLL

IMAGE_PROCESSOR_API void clearDevice();

IMAGE_PROCESSOR_API int deviceCount();
IMAGE_PROCESSOR_API int deviceStats(DevStats** stats);
IMAGE_PROCESSOR_API int memoryStats(std::size_t** stats);


/// Example wrapper header calls 
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<bool> imageIn, ImageView<bool> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);

//IMAGE_PROCESSOR_API void fooFilter(const ImageView<char> imageIn, ImageView<char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<short> imageIn, ImageView<short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<int> imageIn, ImageView<int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<float> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<double> imageIn, ImageView<double> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);

IMAGE_PROCESSOR_API void closure(const ImageView<bool> imageIn, ImageView<bool> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void closure(const ImageView<char> imageIn, ImageView<char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void closure(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void closure(const ImageView<short> imageIn, ImageView<short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void closure(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void closure(const ImageView<int> imageIn, ImageView<int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void closure(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void closure(const ImageView<float> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void closure(const ImageView<double> imageIn, ImageView<double> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);

IMAGE_PROCESSOR_API void elementWiseDifference(const ImageView<bool> image1In, ImageView<bool> image2In, ImageView<bool> imageOut, int device = -1);
IMAGE_PROCESSOR_API void elementWiseDifference(const ImageView<char> image1In, ImageView<char> image2In, ImageView<char> imageOut, int device = -1);
IMAGE_PROCESSOR_API void elementWiseDifference(const ImageView<unsigned char> image1In, ImageView<unsigned char> image2In, ImageView<unsigned char> imageOut, int device = -1);
IMAGE_PROCESSOR_API void elementWiseDifference(const ImageView<short> image1In, ImageView<short> image2In, ImageView<short> imageOut, int device = -1);
IMAGE_PROCESSOR_API void elementWiseDifference(const ImageView<unsigned short> image1In, ImageView<unsigned short> image2In, ImageView<unsigned short> imageOut, int device = -1);
IMAGE_PROCESSOR_API void elementWiseDifference(const ImageView<int> image1In, ImageView<int> image2In, ImageView<int> imageOut, int device = -1);
IMAGE_PROCESSOR_API void elementWiseDifference(const ImageView<unsigned int> image1In, ImageView<unsigned int> image2In, ImageView<unsigned int> imageOut, int device = -1);
IMAGE_PROCESSOR_API void elementWiseDifference(const ImageView<float> image1In, ImageView<float> image2In, ImageView<float> imageOut, int device = -1);
IMAGE_PROCESSOR_API void elementWiseDifference(const ImageView<double> image1In, ImageView<double> image2In, ImageView<double> imageOut, int device = -1);

IMAGE_PROCESSOR_API void entropyFilter(const ImageView<bool> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int device = -1);
IMAGE_PROCESSOR_API void entropyFilter(const ImageView<char> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int device = -1);
IMAGE_PROCESSOR_API void entropyFilter(const ImageView<unsigned char> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int device = -1);
IMAGE_PROCESSOR_API void entropyFilter(const ImageView<short> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int device = -1);
IMAGE_PROCESSOR_API void entropyFilter(const ImageView<unsigned short> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int device = -1);
IMAGE_PROCESSOR_API void entropyFilter(const ImageView<int> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int device = -1);
IMAGE_PROCESSOR_API void entropyFilter(const ImageView<unsigned int> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int device = -1);
IMAGE_PROCESSOR_API void entropyFilter(const ImageView<float> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int device = -1);
IMAGE_PROCESSOR_API void entropyFilter(const ImageView<double> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int device = -1);

IMAGE_PROCESSOR_API void gaussian(const ImageView<bool> imageIn, ImageView<bool> imageOut, Vec<double> sigmas, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void gaussian(const ImageView<char> imageIn, ImageView<char> imageOut, Vec<double> sigmas, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void gaussian(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, Vec<double> sigmas, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void gaussian(const ImageView<short> imageIn, ImageView<short> imageOut, Vec<double> sigmas, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void gaussian(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, Vec<double> sigmas, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void gaussian(const ImageView<int> imageIn, ImageView<int> imageOut, Vec<double> sigmas, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void gaussian(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, Vec<double> sigmas, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void gaussian(const ImageView<float> imageIn, ImageView<float> imageOut, Vec<double> sigmas, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void gaussian(const ImageView<double> imageIn, ImageView<double> imageOut, Vec<double> sigmas, int numIterations = 1, int device = -1);

IMAGE_PROCESSOR_API void getMinMax(const bool* imageIn, std::size_t numElements, bool& minVal, bool& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const char* imageIn, std::size_t numElements, char& minVal, char& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const unsigned char* imageIn, std::size_t numElements, unsigned char& minVal, unsigned char& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const short* imageIn, std::size_t numElements, short& minVal, short& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const unsigned short* imageIn, std::size_t numElements, unsigned short& minVal, unsigned short& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const int* imageIn, std::size_t numElements, int& minVal, int& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const unsigned int* imageIn, std::size_t numElements, unsigned int& minVal, unsigned int& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const float* imageIn, std::size_t numElements, float& minVal, float& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const double* imageIn, std::size_t numElements, double& minVal, double& maxVal, int device = 0);

IMAGE_PROCESSOR_API void highPassFilter(const ImageView<bool> imageIn, ImageView<bool> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void highPassFilter(const ImageView<char> imageIn, ImageView<char> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void highPassFilter(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void highPassFilter(const ImageView<short> imageIn, ImageView<short> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void highPassFilter(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void highPassFilter(const ImageView<int> imageIn, ImageView<int> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void highPassFilter(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void highPassFilter(const ImageView<float> imageIn, ImageView<float> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void highPassFilter(const ImageView<double> imageIn, ImageView<double> imageOut, Vec<double> sigmas, int device = -1);

IMAGE_PROCESSOR_API void identityFilter(const ImageView<bool> imageIn, ImageView<bool> imageOut, int device = -1);
IMAGE_PROCESSOR_API void identityFilter(const ImageView<char> imageIn, ImageView<char> imageOut, int device = -1);
IMAGE_PROCESSOR_API void identityFilter(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, int device = -1);
IMAGE_PROCESSOR_API void identityFilter(const ImageView<short> imageIn, ImageView<short> imageOut, int device = -1);
IMAGE_PROCESSOR_API void identityFilter(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, int device = -1);
IMAGE_PROCESSOR_API void identityFilter(const ImageView<int> imageIn, ImageView<int> imageOut, int device = -1);
IMAGE_PROCESSOR_API void identityFilter(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, int device = -1);
IMAGE_PROCESSOR_API void identityFilter(const ImageView<float> imageIn, ImageView<float> imageOut, int device = -1);
IMAGE_PROCESSOR_API void identityFilter(const ImageView<double> imageIn, ImageView<double> imageOut, int device = -1);

IMAGE_PROCESSOR_API void LoG(const ImageView<bool> imageIn, ImageView<float> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void LoG(const ImageView<char> imageIn, ImageView<float> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void LoG(const ImageView<unsigned char> imageIn, ImageView<float> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void LoG(const ImageView<short> imageIn, ImageView<float> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void LoG(const ImageView<unsigned short> imageIn, ImageView<float> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void LoG(const ImageView<int> imageIn, ImageView<float> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void LoG(const ImageView<unsigned int> imageIn, ImageView<float> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void LoG(const ImageView<float> imageIn, ImageView<float> imageOut, Vec<double> sigmas, int device = -1);
IMAGE_PROCESSOR_API void LoG(const ImageView<double> imageIn, ImageView<float> imageOut, Vec<double> sigmas, int device = -1);

IMAGE_PROCESSOR_API void maxFilter(const ImageView<bool> imageIn, ImageView<bool> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void maxFilter(const ImageView<char> imageIn, ImageView<char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void maxFilter(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void maxFilter(const ImageView<short> imageIn, ImageView<short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void maxFilter(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void maxFilter(const ImageView<int> imageIn, ImageView<int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void maxFilter(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void maxFilter(const ImageView<float> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void maxFilter(const ImageView<double> imageIn, ImageView<double> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);

IMAGE_PROCESSOR_API void meanFilter(const ImageView<bool> imageIn, ImageView<bool> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void meanFilter(const ImageView<char> imageIn, ImageView<char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void meanFilter(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void meanFilter(const ImageView<short> imageIn, ImageView<short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void meanFilter(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void meanFilter(const ImageView<int> imageIn, ImageView<int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void meanFilter(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void meanFilter(const ImageView<float> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void meanFilter(const ImageView<double> imageIn, ImageView<double> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);

IMAGE_PROCESSOR_API void medianFilter(const ImageView<bool> imageIn, ImageView<bool> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void medianFilter(const ImageView<char> imageIn, ImageView<char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void medianFilter(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void medianFilter(const ImageView<short> imageIn, ImageView<short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void medianFilter(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void medianFilter(const ImageView<int> imageIn, ImageView<int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void medianFilter(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void medianFilter(const ImageView<float> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void medianFilter(const ImageView<double> imageIn, ImageView<double> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);

IMAGE_PROCESSOR_API void minFilter(const ImageView<bool> imageIn, ImageView<bool> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void minFilter(const ImageView<char> imageIn, ImageView<char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void minFilter(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void minFilter(const ImageView<short> imageIn, ImageView<short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void minFilter(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void minFilter(const ImageView<int> imageIn, ImageView<int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void minFilter(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void minFilter(const ImageView<float> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void minFilter(const ImageView<double> imageIn, ImageView<double> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);

IMAGE_PROCESSOR_API void minMax(const ImageView<bool> imageIn, bool& minVal, bool& maxVal, int device = -1);
IMAGE_PROCESSOR_API void minMax(const ImageView<char> imageIn, char& minVal, char& maxVal, int device = -1);
IMAGE_PROCESSOR_API void minMax(const ImageView<unsigned char> imageIn, unsigned char& minVal, unsigned char& maxVal, int device = -1);
IMAGE_PROCESSOR_API void minMax(const ImageView<short> imageIn, short& minVal, short& maxVal, int device = -1);
IMAGE_PROCESSOR_API void minMax(const ImageView<unsigned short> imageIn, unsigned short& minVal, unsigned short& maxVal, int device = -1);
IMAGE_PROCESSOR_API void minMax(const ImageView<int> imageIn, int& minVal, int& maxVal, int device = -1);
IMAGE_PROCESSOR_API void minMax(const ImageView<unsigned int> imageIn, unsigned int& minVal, unsigned int& maxVal, int device = -1);
IMAGE_PROCESSOR_API void minMax(const ImageView<float> imageIn, float& minVal, float& maxVal, int device = -1);
IMAGE_PROCESSOR_API void minMax(const ImageView<double> imageIn, double& minVal, double& maxVal, int device = -1);

IMAGE_PROCESSOR_API void multiplySum(const ImageView<bool> imageIn, ImageView<bool> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void multiplySum(const ImageView<char> imageIn, ImageView<char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void multiplySum(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void multiplySum(const ImageView<short> imageIn, ImageView<short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void multiplySum(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void multiplySum(const ImageView<int> imageIn, ImageView<int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void multiplySum(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void multiplySum(const ImageView<float> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void multiplySum(const ImageView<double> imageIn, ImageView<double> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);

IMAGE_PROCESSOR_API void opener(const ImageView<bool> imageIn, ImageView<bool> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void opener(const ImageView<char> imageIn, ImageView<char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void opener(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void opener(const ImageView<short> imageIn, ImageView<short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void opener(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void opener(const ImageView<int> imageIn, ImageView<int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void opener(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void opener(const ImageView<float> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void opener(const ImageView<double> imageIn, ImageView<double> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);

IMAGE_PROCESSOR_API void stdFilter(const ImageView<bool> imageIn, ImageView<bool> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void stdFilter(const ImageView<char> imageIn, ImageView<char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void stdFilter(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void stdFilter(const ImageView<short> imageIn, ImageView<short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void stdFilter(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void stdFilter(const ImageView<int> imageIn, ImageView<int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void stdFilter(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void stdFilter(const ImageView<float> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void stdFilter(const ImageView<double> imageIn, ImageView<double> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);

IMAGE_PROCESSOR_API void sum(const ImageView<bool> imageIn, std::size_t& valOut, int device = -1);
IMAGE_PROCESSOR_API void sum(const ImageView<char> imageIn, long long& valOut, int device = -1);
IMAGE_PROCESSOR_API void sum(const ImageView<unsigned char> imageIn, std::size_t& valOut, int device = -1);
IMAGE_PROCESSOR_API void sum(const ImageView<short> imageIn, long long& valOut, int device = -1);
IMAGE_PROCESSOR_API void sum(const ImageView<unsigned short> imageIn, std::size_t& valOut, int device = -1);
IMAGE_PROCESSOR_API void sum(const ImageView<int> imageIn, long long& valOut, int device = -1);
IMAGE_PROCESSOR_API void sum(const ImageView<unsigned int> imageIn, std::size_t& valOut, int device = -1);
IMAGE_PROCESSOR_API void sum(const ImageView<float> imageIn, double& valOut, int device = -1);
IMAGE_PROCESSOR_API void sum(const ImageView<double> imageIn, double& valOut, int device = -1);


IMAGE_PROCESSOR_API void varFilter(const ImageView<bool> imageIn, ImageView<bool> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void varFilter(const ImageView<char> imageIn, ImageView<char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void varFilter(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void varFilter(const ImageView<short> imageIn, ImageView<short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void varFilter(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void varFilter(const ImageView<int> imageIn, ImageView<int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void varFilter(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void varFilter(const ImageView<float> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
IMAGE_PROCESSOR_API void varFilter(const ImageView<double> imageIn, ImageView<double> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);

IMAGE_PROCESSOR_API void wienerFilter(ImageView<bool> imageIn, ImageView<bool> imageOut, ImageView<float> kernel, double noiseVariance = -1.0, int device = -1);
IMAGE_PROCESSOR_API void wienerFilter(ImageView<char> imageIn, ImageView<char> imageOut, ImageView<float> kernel, double noiseVariance = -1.0, int device = -1);
IMAGE_PROCESSOR_API void wienerFilter(ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, ImageView<float> kernel, double noiseVariance = -1.0, int device = -1);
IMAGE_PROCESSOR_API void wienerFilter(ImageView<short> imageIn, ImageView<short> imageOut, ImageView<float> kernel, double noiseVariance = -1.0, int device = -1);
IMAGE_PROCESSOR_API void wienerFilter(ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, ImageView<float> kernel, double noiseVariance = -1.0, int device = -1);
IMAGE_PROCESSOR_API void wienerFilter(ImageView<int> imageIn, ImageView<int> imageOut, ImageView<float> kernel, double noiseVariance = -1.0, int device = -1);
IMAGE_PROCESSOR_API void wienerFilter(ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, ImageView<float> kernel, double noiseVariance = -1.0, int device = -1);
IMAGE_PROCESSOR_API void wienerFilter(ImageView<float> imageIn, ImageView<float> imageOut, ImageView<float> kernel, double noiseVariance = -1.0, int device = -1);
IMAGE_PROCESSOR_API void wienerFilter(ImageView<double> imageIn, ImageView<double> imageOut, ImageView<float> kernel, double noiseVariance = -1.0, int device = -1);

