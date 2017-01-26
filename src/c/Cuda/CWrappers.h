#pragma once
#include "Vec.h"
#include "CudaDeviceStats.h"
#include <limits>

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

IMAGE_PROCESSOR_API unsigned char* addConstant(const unsigned char* imageIn, Vec<size_t> dims, double additive, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* addConstant(const unsigned short* imageIn, Vec<size_t> dims, double additive, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* addConstant(const short* imageIn, Vec<size_t> dims, double additive, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* addConstant(const unsigned int* imageIn, Vec<size_t> dims, double additive, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* addConstant(const int* imageIn, Vec<size_t> dims, double additive, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* addConstant(const float* imageIn, Vec<size_t> dims, double additive, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* addConstant(const double* imageIn, Vec<size_t> dims, double additive, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API unsigned char* addImageWith(const unsigned char* imageIn1, const unsigned char* imageIn2, Vec<size_t> dims, double additive, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* addImageWith(const unsigned short* imageIn1, const unsigned short* imageIn2, Vec<size_t> dims, double additive, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* addImageWith(const short* imageIn1, const short* imageIn2, Vec<size_t> dims, double additive, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* addImageWith(const unsigned int* imageIn1, const unsigned int* imageIn2, Vec<size_t> dims, double additive, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* addImageWith(const int* imageIn1, const int* imageIn2, Vec<size_t> dims, double additive, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* addImageWith(const float* imageIn1, const float* imageIn2, Vec<size_t> dims, double additive, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* addImageWith(const double* imageIn1, const double* imageIn2, Vec<size_t> dims, double additive, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API unsigned char* applyPolyTransferFunction(const unsigned char* imageIn, Vec<size_t> dims, double a, double b, double c, unsigned char minValue = std::numeric_limits<unsigned char>::lowest(), unsigned char maxValue = std::numeric_limits<unsigned char>::max(), unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* applyPolyTransferFunction(const unsigned short* imageIn, Vec<size_t> dims, double a, double b, double c, unsigned short minValue = std::numeric_limits<unsigned short>::lowest(), unsigned short maxValue = std::numeric_limits<unsigned short>::max(), unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* applyPolyTransferFunction(const short* imageIn, Vec<size_t> dims, double a, double b, double c, short minValue = std::numeric_limits<short>::lowest(), short maxValue = std::numeric_limits<short>::max(), short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* applyPolyTransferFunction(const unsigned int* imageIn, Vec<size_t> dims, double a, double b, double c, unsigned int minValue = std::numeric_limits<unsigned int>::lowest(), unsigned int maxValue = std::numeric_limits<unsigned int>::max(), unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* applyPolyTransferFunction(const int* imageIn, Vec<size_t> dims, double a, double b, double c, int minValue = std::numeric_limits<int>::lowest(), int maxValue = std::numeric_limits<int>::max(), int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* applyPolyTransferFunction(const float* imageIn, Vec<size_t> dims, double a, double b, double c, float minValue = std::numeric_limits<float>::lowest(), float maxValue = std::numeric_limits<float>::max(), float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* applyPolyTransferFunction(const double* imageIn, Vec<size_t> dims, double a, double b, double c, double minValue = std::numeric_limits<double>::lowest(), double maxValue = std::numeric_limits<double>::max(), double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API unsigned char* contrastEnhancement(const unsigned char* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* contrastEnhancement(const unsigned short* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* contrastEnhancement(const short* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* contrastEnhancement(const unsigned int* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* contrastEnhancement(const int* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* contrastEnhancement(const float* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* contrastEnhancement(const double* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API int deviceCount();
IMAGE_PROCESSOR_API int deviceStats(DevStats** stats);

IMAGE_PROCESSOR_API size_t* histogram(const unsigned char* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned char minVal = std::numeric_limits<unsigned char>::lowest(), unsigned char maxVal = std::numeric_limits<unsigned char>::max(), int device = 0);
IMAGE_PROCESSOR_API size_t* histogram(const unsigned short* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned short minVal = std::numeric_limits<unsigned short>::lowest(), unsigned short maxVal = std::numeric_limits<unsigned short>::max(), int device = 0);
IMAGE_PROCESSOR_API size_t* histogram(const short* imageIn, Vec<size_t> dims, unsigned int arraySize, short minVal = std::numeric_limits<short>::lowest(), short maxVal = std::numeric_limits<short>::max(), int device = 0);
IMAGE_PROCESSOR_API size_t* histogram(const unsigned int* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned int minVal = std::numeric_limits<unsigned int>::lowest(), unsigned int maxVal = std::numeric_limits<unsigned int>::max(), int device = 0);
IMAGE_PROCESSOR_API size_t* histogram(const int* imageIn, Vec<size_t> dims, unsigned int arraySize, int minVal = std::numeric_limits<int>::lowest(), int maxVal = std::numeric_limits<int>::max(), int device = 0);
IMAGE_PROCESSOR_API size_t* histogram(const float* imageIn, Vec<size_t> dims, unsigned int arraySize, float minVal = std::numeric_limits<float>::lowest(), float maxVal = std::numeric_limits<float>::max(), int device = 0);
IMAGE_PROCESSOR_API size_t* histogram(const double* imageIn, Vec<size_t> dims, unsigned int arraySize, double minVal = std::numeric_limits<double>::lowest(), double maxVal = std::numeric_limits<double>::max(), int device = 0);

IMAGE_PROCESSOR_API unsigned char* gaussianFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<float> sigmas, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* gaussianFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<float> sigmas, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* gaussianFilter(const short* imageIn, Vec<size_t> dims, Vec<float> sigmas, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* gaussianFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<float> sigmas, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* gaussianFilter(const int* imageIn, Vec<size_t> dims, Vec<float> sigmas, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* gaussianFilter(const float* imageIn, Vec<size_t> dims, Vec<float> sigmas, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* gaussianFilter(const double* imageIn, Vec<size_t> dims, Vec<float> sigmas, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API void getMinMax(const unsigned char* imageIn, Vec<size_t> dims, unsigned char& minVal, unsigned char& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const unsigned short* imageIn, Vec<size_t> dims, unsigned short& minVal, unsigned short& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const short* imageIn, Vec<size_t> dims, short& minVal, short& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const unsigned int* imageIn, Vec<size_t> dims, unsigned int& minVal, unsigned int& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const int* imageIn, Vec<size_t> dims, int& minVal, int& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const float* imageIn, Vec<size_t> dims, float& minVal, float& maxVal, int device = 0);
IMAGE_PROCESSOR_API void getMinMax(const double* imageIn, Vec<size_t> dims, double& minVal, double& maxVal, int device = 0);

IMAGE_PROCESSOR_API unsigned char* imagePow(const unsigned char* imageIn, Vec<size_t> dims, double power, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* imagePow(const unsigned short* imageIn, Vec<size_t> dims, double power, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* imagePow(const short* imageIn, Vec<size_t> dims, double power, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* imagePow(const unsigned int* imageIn, Vec<size_t> dims, double power, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* imagePow(const int* imageIn, Vec<size_t> dims, double power, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* imagePow(const float* imageIn, Vec<size_t> dims, double power, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* imagePow(const double* imageIn, Vec<size_t> dims, double power, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API unsigned char* linearUnmixing(const unsigned char* imageIn, Vec<size_t> imageDims, size_t numImages, const float* unmixing, Vec<size_t> umixingDims, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* linearUnmixing(const unsigned short* imageIn, Vec<size_t> imageDims, size_t numImages, const float* unmixing, Vec<size_t> umixingDims, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* linearUnmixing(const short* imageIn, Vec<size_t> imageDims, size_t numImages, const float* unmixing, Vec<size_t> umixingDims, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* linearUnmixing(const unsigned int* imageIn, Vec<size_t> imageDims, size_t numImages, const float* unmixing, Vec<size_t> umixingDims, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* linearUnmixing(const int* imageIn, Vec<size_t> imageDims, size_t numImages, const float* unmixing, Vec<size_t> umixingDims, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* linearUnmixing(const float* imageIn, Vec<size_t> imageDims, size_t numImages, const float* unmixing, Vec<size_t> umixingDims, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* linearUnmixing(const double* imageIn, Vec<size_t> imageDims, size_t numImages, const float* unmixing, Vec<size_t> umixingDims, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API float* markovRandomFieldDenoiser(const float* imageIn, Vec<size_t> dims, int maxIterations, float** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API unsigned char* maxFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* maxFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* maxFilter(const short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* maxFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* maxFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* maxFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* maxFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API unsigned char* meanFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* meanFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* meanFilter(const short* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* meanFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* meanFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* meanFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* meanFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API unsigned char* medianFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* medianFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* medianFilter(const short* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* medianFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* medianFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* medianFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* medianFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API int memoryStats(size_t** stats);

IMAGE_PROCESSOR_API unsigned char* minFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* minFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* minFilter(const short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* minFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* minFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* minFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* minFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API bool* morphologicalClosure(const bool* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, bool** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned char* morphologicalClosure(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* morphologicalClosure(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* morphologicalClosure(const short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* morphologicalClosure(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* morphologicalClosure(const int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* morphologicalClosure(const float* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* morphologicalClosure(const double* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API bool* morphologicalOpening(const bool* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, bool** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned char* morphologicalOpening(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* morphologicalOpening(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* morphologicalOpening(const short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* morphologicalOpening(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* morphologicalOpening(const int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* morphologicalOpening(const float* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* morphologicalOpening(const double* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel = NULL, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API unsigned char* multiplyImage(const unsigned char* imageIn, Vec<size_t> dims, double multiplier, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* multiplyImage(const unsigned short* imageIn, Vec<size_t> dims, double multiplier, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* multiplyImage(const short* imageIn, Vec<size_t> dims, double multiplier, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* multiplyImage(const unsigned int* imageIn, Vec<size_t> dims, double multiplier, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* multiplyImage(const int* imageIn, Vec<size_t> dims, double multiplier, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* multiplyImage(const float* imageIn, Vec<size_t> dims, double multiplier, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* multiplyImage(const double* imageIn, Vec<size_t> dims, double multiplier, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API unsigned char* multiplyImageWith(const unsigned char* imageIn1, const unsigned char* imageIn2, Vec<size_t> dims, double factor, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* multiplyImageWith(const unsigned short* imageIn1, const unsigned short* imageIn2, Vec<size_t> dims, double factor, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* multiplyImageWith(const short* imageIn1, const short* imageIn2, Vec<size_t> dims, double factor, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* multiplyImageWith(const unsigned int* imageIn1, const unsigned int* imageIn2, Vec<size_t> dims, double factor, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* multiplyImageWith(const int* imageIn1, const int* imageIn2, Vec<size_t> dims, double factor, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* multiplyImageWith(const float* imageIn1, const float* imageIn2, Vec<size_t> dims, double factor, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* multiplyImageWith(const double* imageIn1, const double* imageIn2, Vec<size_t> dims, double factor, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API double normalizedCovariance(const unsigned char* imageIn1, const unsigned char* imageIn2, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API double normalizedCovariance(const unsigned short* imageIn1, const unsigned short* imageIn2, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API double normalizedCovariance(const short* imageIn1, const short* imageIn2, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API double normalizedCovariance(const unsigned int* imageIn1, const unsigned int* imageIn2, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API double normalizedCovariance(const int* imageIn1, const int* imageIn2, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API double normalizedCovariance(const float* imageIn1, const float* imageIn2, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API double normalizedCovariance(const double* imageIn1, const double* imageIn2, Vec<size_t> dims, int device = 0);

IMAGE_PROCESSOR_API double* normalizeHistogram(const unsigned char* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned char minVal = std::numeric_limits<unsigned char>::lowest(), unsigned char maxVal = std::numeric_limits<unsigned char>::max(), int device = 0);
IMAGE_PROCESSOR_API double* normalizeHistogram(const unsigned short* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned short minVal = std::numeric_limits<unsigned short>::lowest(), unsigned short maxVal = std::numeric_limits<unsigned short>::max(), int device = 0);
IMAGE_PROCESSOR_API double* normalizeHistogram(const short* imageIn, Vec<size_t> dims, unsigned int arraySize, short minVal = std::numeric_limits<short>::lowest(), short maxVal = std::numeric_limits<short>::max(), int device = 0);
IMAGE_PROCESSOR_API double* normalizeHistogram(const unsigned int* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned int minVal = std::numeric_limits<unsigned int>::lowest(), unsigned int maxVal = std::numeric_limits<unsigned int>::max(), int device = 0);
IMAGE_PROCESSOR_API double* normalizeHistogram(const int* imageIn, Vec<size_t> dims, unsigned int arraySize, int minVal = std::numeric_limits<int>::lowest(), int maxVal = std::numeric_limits<int>::max(), int device = 0);
IMAGE_PROCESSOR_API double* normalizeHistogram(const float* imageIn, Vec<size_t> dims, unsigned int arraySize, float minVal = std::numeric_limits<float>::lowest(), float maxVal = std::numeric_limits<float>::max(), int device = 0);
IMAGE_PROCESSOR_API double* normalizeHistogram(const double* imageIn, Vec<size_t> dims, unsigned int arraySize, double minVal = std::numeric_limits<double>::lowest(), double maxVal = std::numeric_limits<double>::max(), int device = 0);

IMAGE_PROCESSOR_API unsigned char* otsuThresholdFilter(const unsigned char* imageIn, Vec<size_t> dims, double alpha = 1.0, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* otsuThresholdFilter(const unsigned short* imageIn, Vec<size_t> dims, double alpha = 1.0, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* otsuThresholdFilter(const short* imageIn, Vec<size_t> dims, double alpha = 1.0, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* otsuThresholdFilter(const unsigned int* imageIn, Vec<size_t> dims, double alpha = 1.0, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* otsuThresholdFilter(const int* imageIn, Vec<size_t> dims, double alpha = 1.0, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* otsuThresholdFilter(const float* imageIn, Vec<size_t> dims, double alpha = 1.0, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* otsuThresholdFilter(const double* imageIn, Vec<size_t> dims, double alpha = 1.0, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API unsigned char otsuThresholdValue(const unsigned char* imageIn, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API unsigned short otsuThresholdValue(const unsigned short* imageIn, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API short otsuThresholdValue(const short* imageIn, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API unsigned int otsuThresholdValue(const unsigned int* imageIn, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API int otsuThresholdValue(const int* imageIn, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API float otsuThresholdValue(const float* imageIn, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API double otsuThresholdValue(const double* imageIn, Vec<size_t> dims, int device = 0);

IMAGE_PROCESSOR_API unsigned char* reduceImage(const unsigned char* imageIn, Vec<size_t> dims, Vec<double> reductions, Vec<size_t>& reducedDims, ReductionMethods method = REDUC_MEAN, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* reduceImage(const unsigned short* imageIn, Vec<size_t> dims, Vec<double> reductions, Vec<size_t>& reducedDims, ReductionMethods method = REDUC_MEAN, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* reduceImage(const short* imageIn, Vec<size_t> dims, Vec<double> reductions, Vec<size_t>& reducedDims, ReductionMethods method = REDUC_MEAN, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* reduceImage(const unsigned int* imageIn, Vec<size_t> dims, Vec<double> reductions, Vec<size_t>& reducedDims, ReductionMethods method = REDUC_MEAN, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* reduceImage(const int* imageIn, Vec<size_t> dims, Vec<double> reductions, Vec<size_t>& reducedDims, ReductionMethods method = REDUC_MEAN, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* reduceImage(const float* imageIn, Vec<size_t> dims, Vec<double> reductions, Vec<size_t>& reducedDims, ReductionMethods method = REDUC_MEAN, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* reduceImage(const double* imageIn, Vec<size_t> dims, Vec<double> reductions, Vec<size_t>& reducedDims, ReductionMethods method = REDUC_MEAN, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API void regionGrowing(const unsigned char* imageIn,Vec<size_t> dims,Vec<size_t> kernelDims,float* kernel,bool* imageMask,double threshold,bool allowConnection=true,int device=0);
IMAGE_PROCESSOR_API void regionGrowing(const unsigned short* imageIn,Vec<size_t> dims,Vec<size_t> kernelDims,float* kernel,bool* imageMask,double threshold,bool allowConnection=true,int device=0);
IMAGE_PROCESSOR_API void regionGrowing(const short* imageIn,Vec<size_t> dims,Vec<size_t> kernelDims,float* kernel,bool* imageMask,double threshold,bool allowConnection=true,int device=0);
IMAGE_PROCESSOR_API void regionGrowing(const unsigned int* imageIn,Vec<size_t> dims,Vec<size_t> kernelDims,float* kernel,bool* imageMask,double threshold,bool allowConnection=true,int device=0);
IMAGE_PROCESSOR_API void regionGrowing(const int* imageIn,Vec<size_t> dims,Vec<size_t> kernelDims,float* kernel,bool* imageMask,double threshold,bool allowConnection=true,int device=0);
IMAGE_PROCESSOR_API void regionGrowing(const float* imageIn,Vec<size_t> dims,Vec<size_t> kernelDims,float* kernel,bool* imageMask,double threshold,bool allowConnection=true,int device=0);
IMAGE_PROCESSOR_API void regionGrowing(const double* imageIn,Vec<size_t> dims,Vec<size_t> kernelDims,float* kernel,bool* imageMask,double threshold,bool allowConnection=true,int device=0);

IMAGE_PROCESSOR_API bool* resize(const bool* imageIn, Vec<size_t> dimsIn, Vec<double> resizeFactors, Vec<size_t>& dimsOut, ReductionMethods method = REDUC_MEAN, bool** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned char* resize(const unsigned char* imageIn, Vec<size_t> dimsIn, Vec<double> resizeFactors, Vec<size_t>& dimsOut, ReductionMethods method = REDUC_MEAN, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* resize(const unsigned short* imageIn, Vec<size_t> dimsIn, Vec<double> resizeFactors, Vec<size_t>& dimsOut, ReductionMethods method = REDUC_MEAN, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* resize(const short* imageIn, Vec<size_t> dimsIn, Vec<double> resizeFactors, Vec<size_t>& dimsOut, ReductionMethods method = REDUC_MEAN, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* resize(const unsigned int* imageIn, Vec<size_t> dimsIn, Vec<double> resizeFactors, Vec<size_t>& dimsOut, ReductionMethods method = REDUC_MEAN, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* resize(const int* imageIn, Vec<size_t> dimsIn, Vec<double> resizeFactors, Vec<size_t>& dimsOut, ReductionMethods method = REDUC_MEAN, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* resize(const float* imageIn, Vec<size_t> dimsIn, Vec<double> resizeFactors, Vec<size_t>& dimsOut, ReductionMethods method = REDUC_MEAN, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* resize(const double* imageIn, Vec<size_t> dimsIn, Vec<double> resizeFactors, Vec<size_t>& dimsOut, ReductionMethods method = REDUC_MEAN, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API size_t sumArray(const unsigned char* imageIn, size_t n, int device = 0);
IMAGE_PROCESSOR_API size_t sumArray(const unsigned short* imageIn, size_t n, int device = 0);
IMAGE_PROCESSOR_API size_t sumArray(const short* imageIn, size_t n, int device = 0);
IMAGE_PROCESSOR_API size_t sumArray(const unsigned int* imageIn, size_t n, int device = 0);
IMAGE_PROCESSOR_API size_t sumArray(const int* imageIn, size_t n, int device = 0);
IMAGE_PROCESSOR_API double sumArray(const float* imageIn, size_t n, int device = 0);
IMAGE_PROCESSOR_API double sumArray(const double* imageIn, size_t n, int device = 0);

IMAGE_PROCESSOR_API unsigned char* segment(const unsigned char* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel = NULL, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* segment(const unsigned short* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel = NULL, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* segment(const short* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel = NULL, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* segment(const unsigned int* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel = NULL, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* segment(const int* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel = NULL, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* segment(const float* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel = NULL, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* segment(const double* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel = NULL, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API unsigned char* stdFilter(const unsigned char* imageIn,Vec<size_t> dims,Vec<size_t> neighborhood,unsigned char** imageOut = NULL,int device = 0);
IMAGE_PROCESSOR_API unsigned short* stdFilter(const unsigned short* imageIn,Vec<size_t> dims,Vec<size_t> neighborhood,unsigned short** imageOut = NULL,int device = 0);
IMAGE_PROCESSOR_API short* stdFilter(const short* imageIn,Vec<size_t> dims,Vec<size_t> neighborhood,short** imageOut = NULL,int device = 0);
IMAGE_PROCESSOR_API unsigned int* stdFilter(const unsigned int* imageIn,Vec<size_t> dims,Vec<size_t> neighborhood,unsigned int** imageOut = NULL,int device = 0);
IMAGE_PROCESSOR_API int* stdFilter(const int* imageIn,Vec<size_t> dims,Vec<size_t> neighborhood,int** imageOut = NULL,int device = 0);
IMAGE_PROCESSOR_API float* stdFilter(const float* imageIn,Vec<size_t> dims,Vec<size_t> neighborhood,float** imageOut = NULL,int device = 0);
IMAGE_PROCESSOR_API double* stdFilter(const double* imageIn,Vec<size_t> dims,Vec<size_t> neighborhood,double** imageOut = NULL,int device = 0);

IMAGE_PROCESSOR_API unsigned char* thresholdFilter(const unsigned char* imageIn, Vec<size_t> dims, unsigned char thresh, unsigned char** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned short* thresholdFilter(const unsigned short* imageIn, Vec<size_t> dims, unsigned short thresh, unsigned short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API short* thresholdFilter(const short* imageIn, Vec<size_t> dims, short thresh, short** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API unsigned int* thresholdFilter(const unsigned int* imageIn, Vec<size_t> dims, unsigned int thresh, unsigned int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API int* thresholdFilter(const int* imageIn, Vec<size_t> dims, int thresh, int** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API float* thresholdFilter(const float* imageIn, Vec<size_t> dims, float thresh, float** imageOut = NULL, int device = 0);
IMAGE_PROCESSOR_API double* thresholdFilter(const double* imageIn, Vec<size_t> dims, double thresh, double** imageOut = NULL, int device = 0);

IMAGE_PROCESSOR_API unsigned char* tileImage(const unsigned char* imageIn,Vec<size_t> dims,Vec<size_t> roiStart,Vec<size_t> roiSize,unsigned char** imageOut=NULL,int device=0);
IMAGE_PROCESSOR_API unsigned short* tileImage(const unsigned short* imageIn,Vec<size_t> dims,Vec<size_t> roiStart,Vec<size_t> roiSize,unsigned short** imageOut=NULL,int device=0);
IMAGE_PROCESSOR_API short* tileImage(const short* imageIn,Vec<size_t> dims,Vec<size_t> roiStart,Vec<size_t> roiSize,short** imageOut=NULL,int device=0);
IMAGE_PROCESSOR_API unsigned int* tileImage(const unsigned int* imageIn,Vec<size_t> dims,Vec<size_t> roiStart,Vec<size_t> roiSize,unsigned int** imageOut=NULL,int device=0);
IMAGE_PROCESSOR_API int* tileImage(const int* imageIn,Vec<size_t> dims,Vec<size_t> roiStart,Vec<size_t> roiSize,int** imageOut=NULL,int device=0);
IMAGE_PROCESSOR_API float* tileImage(const float* imageIn,Vec<size_t> dims,Vec<size_t> roiStart,Vec<size_t> roiSize,float** imageOut=NULL,int device=0);
IMAGE_PROCESSOR_API double* tileImage(const double* imageIn,Vec<size_t> dims,Vec<size_t> roiStart,Vec<size_t> roiSize,double** imageOut=NULL,int device=0);

IMAGE_PROCESSOR_API double variance(const unsigned char* imageIn, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API double variance(const unsigned short* imageIn, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API double variance(const short* imageIn, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API double variance(const unsigned int* imageIn, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API double variance(const int* imageIn, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API double variance(const float* imageIn, Vec<size_t> dims, int device = 0);
IMAGE_PROCESSOR_API double variance(const double* imageIn, Vec<size_t> dims, int device = 0);
