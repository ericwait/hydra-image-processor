#pragma once
#include "Vec.h"
#include <limits>
#include "Defines.h"

unsigned char* addConstant(const unsigned char* imageIn, Vec<size_t> dims, double additive, unsigned char** imageOut=NULL, int device=0);
unsigned short* addConstant(const unsigned short* imageIn, Vec<size_t> dims, double additive, unsigned short** imageOut=NULL, int device=0);
short* addConstant(const short* imageIn, Vec<size_t> dims, double additive, short** imageOut=NULL, int device=0);
unsigned int* addConstant(const unsigned int* imageIn, Vec<size_t> dims, double additive, unsigned int** imageOut=NULL, int device=0);
int* addConstant(const int* imageIn, Vec<size_t> dims, double additive, int** imageOut=NULL, int device=0);
float* addConstant(const float* imageIn, Vec<size_t> dims, double additive, float** imageOut=NULL, int device=0);
double* addConstant(const double* imageIn, Vec<size_t> dims, double additive, double** imageOut=NULL, int device=0);

unsigned char* addImageWith(const unsigned char* imageIn1, const unsigned char* imageIn2, Vec<size_t> dims, double additive, unsigned char** imageOut=NULL, int device=0);
unsigned short* addImageWith(const unsigned short* imageIn1, const unsigned short* imageIn2, Vec<size_t> dims, double additive, unsigned short** imageOut=NULL, int device=0);
short* addImageWith(const short* imageIn1, const short* imageIn2, Vec<size_t> dims, double additive, short** imageOut=NULL, int device=0);
unsigned int* addImageWith(const unsigned int* imageIn1, const unsigned int* imageIn2, Vec<size_t> dims, double additive, unsigned int** imageOut=NULL, int device=0);
int* addImageWith(const int* imageIn1, const int* imageIn2, Vec<size_t> dims, double additive, int** imageOut=NULL, int device=0);
float* addImageWith(const float* imageIn1, const float* imageIn2, Vec<size_t> dims, double additive, float** imageOut=NULL, int device=0);
double* addImageWith(const double* imageIn1, const double* imageIn2, Vec<size_t> dims, double additive, double** imageOut=NULL, int device=0);

unsigned char* applyPolyTransferFunction(const unsigned char* imageIn, Vec<size_t> dims, double a, double b, double c, unsigned char minValue=std::numeric_limits<unsigned char>::lowest(), unsigned char maxValue=std::numeric_limits<unsigned char>::max(), unsigned char** imageOut=NULL, int device=0);
unsigned short* applyPolyTransferFunction(const unsigned short* imageIn, Vec<size_t> dims, double a, double b, double c, unsigned short minValue=std::numeric_limits<unsigned short>::lowest(), unsigned short maxValue=std::numeric_limits<unsigned short>::max(), unsigned short** imageOut=NULL, int device=0);
short* applyPolyTransferFunction(const short* imageIn, Vec<size_t> dims, double a, double b, double c, short minValue=std::numeric_limits<short>::lowest(), short maxValue=std::numeric_limits<short>::max(), short** imageOut=NULL, int device=0);
unsigned int* applyPolyTransferFunction(const unsigned int* imageIn, Vec<size_t> dims, double a, double b, double c, unsigned int minValue=std::numeric_limits<unsigned int>::lowest(),unsigned int maxValue=std::numeric_limits<unsigned int>::max(), unsigned int** imageOut=NULL, int device=0);
int* applyPolyTransferFunction(const int* imageIn, Vec<size_t> dims, double a, double b, double c, int minValue=std::numeric_limits<int>::lowest(), int maxValue=std::numeric_limits<int>::max(), int** imageOut=NULL, int device=0);
float* applyPolyTransferFunction(const float* imageIn, Vec<size_t> dims, double a, double b, double c, float minValue=std::numeric_limits<float>::lowest(), float maxValue=std::numeric_limits<float>::max(), float** imageOut=NULL, int device=0);
double* applyPolyTransferFunction(const double* imageIn, Vec<size_t> dims, double a, double b, double c, double minValue=std::numeric_limits<double>::lowest(), double maxValue=std::numeric_limits<double>::max(),double** imageOut=NULL, int device=0);

unsigned char* contrastEnhancement(const unsigned char* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, unsigned char** imageOut=NULL, int device=0);
unsigned short* contrastEnhancement(const unsigned short* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, unsigned short** imageOut=NULL, int device=0);
short* contrastEnhancement(const short* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, short** imageOut=NULL, int device=0);
unsigned int* contrastEnhancement(const unsigned int* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, unsigned int** imageOut=NULL, int device=0);
int* contrastEnhancement(const int* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, int** imageOut=NULL, int device=0);
float* contrastEnhancement(const float* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, float** imageOut=NULL, int device=0);
double* contrastEnhancement(const double* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, double** imageOut=NULL, int device=0);

size_t* histogram(const unsigned char* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned char minVal=std::numeric_limits<unsigned char>::lowest(), unsigned char maxVal=std::numeric_limits<unsigned char>::max(), int device=0);
size_t* histogram(const unsigned short* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned short minVal=std::numeric_limits<unsigned short>::lowest(), unsigned short maxVal=std::numeric_limits<unsigned short>::max(), int device=0);
size_t* histogram(const short* imageIn, Vec<size_t> dims, unsigned int arraySize, short minVal=std::numeric_limits<short>::lowest(), short maxVal=std::numeric_limits<short>::max(), int device=0);
size_t* histogram(const unsigned int* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned int minVal=std::numeric_limits<unsigned int>::lowest(), unsigned int maxVal=std::numeric_limits<unsigned int>::max(), int device=0);
size_t* histogram(const int* imageIn, Vec<size_t> dims, unsigned int arraySize, int minVal=std::numeric_limits<int>::lowest(), int maxVal=std::numeric_limits<int>::max(), int device=0);
size_t* histogram(const float* imageIn, Vec<size_t> dims, unsigned int arraySize, float minVal=std::numeric_limits<float>::lowest(), float maxVal=std::numeric_limits<float>::max(), int device=0);
size_t* histogram(const double* imageIn, Vec<size_t> dims, unsigned int arraySize, double minVal=std::numeric_limits<double>::lowest(), double maxVal=std::numeric_limits<double>::max(), int device=0);

unsigned char* gaussianFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<float> sigmas, unsigned char** imageOut=NULL, int device=0);
unsigned short* gaussianFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<float> sigmas, unsigned short** imageOut=NULL, int device=0);
short* gaussianFilter(const short* imageIn, Vec<size_t> dims, Vec<float> sigmas, short** imageOut=NULL, int device=0);
unsigned int* gaussianFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<float> sigmas, unsigned int** imageOut=NULL, int device=0);
int* gaussianFilter(const int* imageIn, Vec<size_t> dims, Vec<float> sigmas, int** imageOut=NULL, int device=0);
float* gaussianFilter(const float* imageIn, Vec<size_t> dims, Vec<float> sigmas, float** imageOut=NULL, int device=0);
double* gaussianFilter(const double* imageIn, Vec<size_t> dims, Vec<float> sigmas, double** imageOut=NULL, int device=0);

void getMinMax(const unsigned char* imageIn, Vec<size_t> dims, unsigned char& minVal, unsigned char& maxVal, int device=0);
void getMinMax(const unsigned short* imageIn, Vec<size_t> dims, unsigned short& minVal, unsigned short& maxVal, int device=0);
void getMinMax(const short* imageIn, Vec<size_t> dims, short& minVal, short& maxVal, int device=0);
void getMinMax(const unsigned int* imageIn, Vec<size_t> dims, unsigned int& minVal, unsigned int& maxVal, int device=0);
void getMinMax(const int* imageIn, Vec<size_t> dims, int& minVal, int& maxVal, int device=0);
void getMinMax(const float* imageIn, Vec<size_t> dims, float& minVal, float& maxVal, int device=0);
void getMinMax(const double* imageIn, Vec<size_t> dims, double& minVal, double& maxVal, int device=0);

unsigned char* imagePow(const unsigned char* imageIn, Vec<size_t> dims, double power, unsigned char** imageOut=NULL, int device=0);
unsigned short* imagePow(const unsigned short* imageIn, Vec<size_t> dims, double power, unsigned short** imageOut=NULL, int device=0);
short* imagePow(const short* imageIn, Vec<size_t> dims, double power, short** imageOut=NULL, int device=0);
unsigned int* imagePow(const unsigned int* imageIn, Vec<size_t> dims, double power, unsigned int** imageOut=NULL, int device=0);
int* imagePow(const int* imageIn, Vec<size_t> dims, double power, int** imageOut=NULL, int device=0);
float* imagePow(const float* imageIn, Vec<size_t> dims, double power, float** imageOut=NULL, int device=0);
double* imagePow(const double* imageIn, Vec<size_t> dims, double power, double** imageOut=NULL, int device=0);

unsigned char* maxFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, unsigned char** imageOut=NULL, int device=0);
unsigned short* maxFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, unsigned short** imageOut=NULL, int device=0);
short* maxFilter(const short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, short** imageOut=NULL, int device=0);
unsigned int* maxFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, unsigned int** imageOut=NULL, int device=0);
int* maxFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, int** imageOut=NULL, int device=0);
float* maxFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, float** imageOut=NULL, int device=0);
double* maxFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, double** imageOut=NULL, int device=0);

unsigned char* meanFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned char** imageOut=NULL, int device=0);
unsigned short* meanFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned short** imageOut=NULL, int device=0);
short* meanFilter(const short* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, short** imageOut=NULL, int device=0);
unsigned int* meanFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned int** imageOut=NULL, int device=0);
int* meanFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, int** imageOut=NULL, int device=0);
float* meanFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, float** imageOut=NULL, int device=0);
double* meanFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, double** imageOut=NULL, int device=0);

unsigned char* medianFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned char** imageOut=NULL, int device=0);
unsigned short* medianFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned short** imageOut=NULL, int device=0);
short* medianFilter(const short* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, short** imageOut=NULL, int device=0);
unsigned int* medianFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned int** imageOut=NULL, int device=0);
int* medianFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, int** imageOut=NULL, int device=0);
float* medianFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, float** imageOut=NULL, int device=0);
double* medianFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, double** imageOut=NULL, int device=0);

unsigned char* minFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, unsigned char** imageOut=NULL, int device=0);
unsigned short* minFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, unsigned short** imageOut=NULL, int device=0);
short* minFilter(const short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, short** imageOut=NULL, int device=0);
unsigned int* minFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, unsigned int** imageOut=NULL, int device=0);
int* minFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, int** imageOut=NULL, int device=0);
float* minFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, float** imageOut=NULL, int device=0);
double* minFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, double** imageOut=NULL, int device=0);

unsigned char* morphologicalClosure(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, unsigned char** imageOut=NULL, int device=0);
unsigned short* morphologicalClosure(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, unsigned short** imageOut=NULL, int device=0);
short* morphologicalClosure(const short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, short** imageOut=NULL, int device=0);
unsigned int* morphologicalClosure(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, unsigned int** imageOut=NULL, int device=0);
int* morphologicalClosure(const int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, int** imageOut=NULL, int device=0);
float* morphologicalClosure(const float* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, float** imageOut=NULL, int device=0);
double* morphologicalClosure(const double* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, double** imageOut=NULL, int device=0);

unsigned char* morphologicalOpening(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, unsigned char** imageOut=NULL, int device=0);
unsigned short* morphologicalOpening(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, unsigned short** imageOut=NULL, int device=0);
short* morphologicalOpening(const short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, short** imageOut=NULL, int device=0);
unsigned int* morphologicalOpening(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, unsigned int** imageOut=NULL, int device=0);
int* morphologicalOpening(const int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, int** imageOut=NULL, int device=0);
float* morphologicalOpening(const float* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, float** imageOut=NULL, int device=0);
double* morphologicalOpening(const double* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel=NULL, double** imageOut=NULL, int device=0);

unsigned char* multiplyImage(const unsigned char* imageIn, Vec<size_t> dims, double multiplier, unsigned char** imageOut=NULL, int device=0);
unsigned short* multiplyImage(const unsigned short* imageIn, Vec<size_t> dims, double multiplier, unsigned short** imageOut=NULL, int device=0);
short* multiplyImage(const short* imageIn, Vec<size_t> dims, double multiplier, short** imageOut=NULL, int device=0);
unsigned int* multiplyImage(const unsigned int* imageIn, Vec<size_t> dims, double multiplier, unsigned int** imageOut=NULL, int device=0);
int* multiplyImage(const int* imageIn, Vec<size_t> dims, double multiplier, int** imageOut=NULL, int device=0);
float* multiplyImage(const float* imageIn, Vec<size_t> dims, double multiplier, float** imageOut=NULL, int device=0);
double* multiplyImage(const double* imageIn, Vec<size_t> dims, double multiplier, double** imageOut=NULL, int device=0);

unsigned char* multiplyImageWith(const unsigned char* imageIn1, const unsigned char* imageIn2, Vec<size_t> dims, double factor, unsigned char** imageOut=NULL, int device=0);
unsigned short* multiplyImageWith(const unsigned short* imageIn1, const unsigned short* imageIn2, Vec<size_t> dims, double factor, unsigned short** imageOut=NULL, int device=0);
short* multiplyImageWith(const short* imageIn1, const short* imageIn2, Vec<size_t> dims, double factor, short** imageOut=NULL, int device=0);
unsigned int* multiplyImageWith(const unsigned int* imageIn1, const unsigned int* imageIn2, Vec<size_t> dims, double factor, unsigned int** imageOut=NULL, int device=0);
int* multiplyImageWith(const int* imageIn1, const int* imageIn2, Vec<size_t> dims, double factor, int** imageOut=NULL, int device=0);
float* multiplyImageWith(const float* imageIn1, const float* imageIn2, Vec<size_t> dims, double factor, float** imageOut=NULL, int device=0);
double* multiplyImageWith(const double* imageIn1, const double* imageIn2, Vec<size_t> dims, double factor, double** imageOut=NULL, int device=0);

double normalizedCovariance(const unsigned char* imageIn1, const unsigned char* imageIn2, Vec<size_t> dims, int device=0);
double normalizedCovariance(const unsigned short* imageIn1, const unsigned short* imageIn2, Vec<size_t> dims, int device=0);
double normalizedCovariance(const short* imageIn1, const short* imageIn2, Vec<size_t> dims, int device=0);
double normalizedCovariance(const unsigned int* imageIn1, const unsigned int* imageIn2, Vec<size_t> dims, int device=0);
double normalizedCovariance(const int* imageIn1, const int* imageIn2, Vec<size_t> dims, int device=0);
double normalizedCovariance(const float* imageIn1, const float* imageIn2, Vec<size_t> dims, int device=0);
double normalizedCovariance(const double* imageIn1, const double* imageIn2, Vec<size_t> dims, int device=0);

double* normalizeHistogram(const unsigned char* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned char minVal=std::numeric_limits<unsigned char>::lowest(), unsigned char maxVal=std::numeric_limits<unsigned char>::max(), int device=0);
double* normalizeHistogram(const unsigned short* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned short minVal=std::numeric_limits<unsigned short>::lowest(), unsigned short maxVal=std::numeric_limits<unsigned short>::max(), int device=0);
double* normalizeHistogram(const short* imageIn, Vec<size_t> dims, unsigned int arraySize, short minVal=std::numeric_limits<int>::lowest(), short maxVal=std::numeric_limits<short>::max(), int device=0);
double* normalizeHistogram(const unsigned int* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned int minVal=std::numeric_limits<unsigned int>::lowest(), unsigned int maxVal=std::numeric_limits<unsigned int>::max(), int device=0);
double* normalizeHistogram(const int* imageIn, Vec<size_t> dims, unsigned int arraySize, int minVal=std::numeric_limits<int>::lowest(), int maxVal=std::numeric_limits<int>::max(), int device=0);
double* normalizeHistogram(const float* imageIn, Vec<size_t> dims, unsigned int arraySize, float minVal=std::numeric_limits<float>::lowest(), float maxVal=std::numeric_limits<float>::max(), int device=0);
double* normalizeHistogram(const double* imageIn, Vec<size_t> dims, unsigned int arraySize, double minVal=std::numeric_limits<double>::lowest(), double maxVal=std::numeric_limits<double>::max(), int device=0);

unsigned char* otsuThresholdFilter(const unsigned char* imageIn, Vec<size_t> dims, double alpha=1.0, unsigned char** imageOut=NULL, int device=0);
unsigned short* otsuThresholdFilter(const unsigned short* imageIn, Vec<size_t> dims, double alpha=1.0, unsigned short** imageOut=NULL, int device=0);
short* otsuThresholdFilter(const short* imageIn, Vec<size_t> dims, double alpha=1.0, short** imageOut=NULL, int device=0);
unsigned int* otsuThresholdFilter(const unsigned int* imageIn, Vec<size_t> dims, double alpha=1.0, unsigned int** imageOut=NULL, int device=0);
int* otsuThresholdFilter(const int* imageIn, Vec<size_t> dims, double alpha=1.0, int** imageOut=NULL, int device=0);
float* otsuThresholdFilter(const float* imageIn, Vec<size_t> dims, double alpha=1.0, float** imageOut=NULL, int device=0);
double* otsuThresholdFilter(const double* imageIn, Vec<size_t> dims, double alpha=1.0, double** imageOut=NULL, int device=0);

unsigned char otsuThresholdValue(const unsigned char* imageIn, Vec<size_t> dims, int device=0);
unsigned short otsuThresholdValue(const unsigned short* imageIn, Vec<size_t> dims, int device=0);
short otsuThresholdValue(const short* imageIn, Vec<size_t> dims, int device=0);
unsigned int otsuThresholdValue(const unsigned int* imageIn, Vec<size_t> dims, int device=0);
int otsuThresholdValue(const int* imageIn, Vec<size_t> dims, int device=0);
float otsuThresholdValue(const float* imageIn, Vec<size_t> dims, int device=0);
double otsuThresholdValue(const double* imageIn, Vec<size_t> dims, int device=0);

unsigned char* reduceImage(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method=REDUC_MEAN, unsigned char** imageOut=NULL, int device=0);
unsigned short* reduceImage(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method=REDUC_MEAN, unsigned short** imageOut=NULL, int device=0);
short* reduceImage(const short* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method=REDUC_MEAN, short** imageOut=NULL, int device=0);
unsigned int* reduceImage(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method=REDUC_MEAN, unsigned int** imageOut=NULL, int device=0);
int* reduceImage(const int* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method=REDUC_MEAN, int** imageOut=NULL, int device=0);
float* reduceImage(const float* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method=REDUC_MEAN, float** imageOut=NULL, int device=0);
double* reduceImage(const double* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method=REDUC_MEAN, double** imageOut=NULL, int device=0);

size_t sumArray(const unsigned char* imageIn, size_t n, int device=0);
size_t sumArray(const unsigned short* imageIn, size_t n, int device=0);
size_t sumArray(const short* imageIn, size_t n, int device=0);
size_t sumArray(const unsigned int* imageIn, size_t n, int device=0);
size_t sumArray(const int* imageIn, size_t n, int device=0);
double sumArray(const float* imageIn, size_t n, int device=0);
double sumArray(const double* imageIn, size_t n, int device=0);

unsigned char* thresholdFilter(const unsigned char* imageIn, Vec<size_t> dims, unsigned char thresh, unsigned char** imageOut=NULL, int device=0);
unsigned short* thresholdFilter(const unsigned short* imageIn, Vec<size_t> dims, unsigned short thresh, unsigned short** imageOut=NULL, int device=0);
short* thresholdFilter(const short* imageIn, Vec<size_t> dims, int thresh, short** imageOut=NULL, int device=0);
unsigned int* thresholdFilter(const unsigned int* imageIn, Vec<size_t> dims, unsigned int thresh, unsigned int** imageOut=NULL, int device=0);
int* thresholdFilter(const int* imageIn, Vec<size_t> dims, int thresh, int** imageOut=NULL, int device=0);
float* thresholdFilter(const float* imageIn, Vec<size_t> dims, float thresh, float** imageOut=NULL, int device=0);
double* thresholdFilter(const double* imageIn, Vec<size_t> dims, double thresh, double** imageOut=NULL, int device=0);
