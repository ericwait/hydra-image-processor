#include "CWrappers.cuh"
#include "CudaAdd.cuh"
#include "CudaContrastEnhancement.cuh"
#include "CudaHistogram.cuh"
#include "CudaGaussianFilter.cuh"
#include "CudaGetMinMax.cuh"
#include "CudaMarkovRandomFieldDenoiser.cuh"
#include "CudaMaxFilter.cuh"
#include "CudaMeanFilter.cuh"
#include "CudaMedianFilter.cuh"
#include "CudaMinFilter.cuh"
#include "CudaMorphologicalOperations.cuh"
#include "CudaMultiplyImage.cuh"
#include "CudaNormalizedCovariance.cuh"
#include "CudaPolyTransferFunc.cuh"
#include "CudaPow.cuh"
#include "CudaImageReduction.cuh"
#include "CudaSum.cuh"
#include "CudaSegment.cuh"
#include "CudaThreshold.cuh"
#include "CudaVariance.cuh"

void clearDevice()
{
	cudaDeviceReset();
}

 unsigned char* addConstant(const unsigned char* imageIn, Vec<size_t> dims, double additive, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddConstant(imageIn,dims,additive,imageOut,device);
}

 unsigned short* addConstant(const unsigned short* imageIn, Vec<size_t> dims, double additive, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddConstant(imageIn,dims,additive,imageOut,device);
}

 short* addConstant(const short* imageIn, Vec<size_t> dims, double additive, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddConstant(imageIn,dims,additive,imageOut,device);
}

 unsigned int* addConstant(const unsigned int* imageIn, Vec<size_t> dims, double additive, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddConstant(imageIn,dims,additive,imageOut,device);
}

 int* addConstant(const int* imageIn, Vec<size_t> dims, double additive, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddConstant(imageIn,dims,additive,imageOut,device);
}

 float* addConstant(const float* imageIn, Vec<size_t> dims, double additive, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddConstant(imageIn,dims,additive,imageOut,device);
}

 double* addConstant(const double* imageIn, Vec<size_t> dims, double additive, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddConstant(imageIn,dims,additive,imageOut,device);
}


 unsigned char* applyPolyTransferFunction(const unsigned char* imageIn, Vec<size_t> dims, double a, double b, double c, unsigned char minValue/*=std::numeric_limits<PixelType>::lowest()*/, unsigned char maxValue/*=std::numeric_limits<PixelType>::max()*/, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cApplyPolyTransferFunction(imageIn,dims,a,b,c,minValue,maxValue,imageOut,device);
}

 unsigned short* applyPolyTransferFunction(const unsigned short* imageIn, Vec<size_t> dims, double a, double b, double c, unsigned short minValue/*=std::numeric_limits<PixelType>::lowest()*/, unsigned short maxValue/*=std::numeric_limits<PixelType>::max()*/, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cApplyPolyTransferFunction(imageIn,dims,a,b,c,minValue,maxValue,imageOut,device);
}

 short* applyPolyTransferFunction(const short* imageIn, Vec<size_t> dims, double a, double b, double c, short minValue/*=std::numeric_limits<PixelType>::lowest()*/, short maxValue/*=std::numeric_limits<PixelType>::max()*/, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cApplyPolyTransferFunction(imageIn,dims,a,b,c,minValue,maxValue,imageOut,device);
}

 unsigned int* applyPolyTransferFunction(const unsigned int* imageIn, Vec<size_t> dims, double a, double b, double c, unsigned int minValue/*=std::numeric_limits<PixelType>::lowest()*/, unsigned int maxValue/*=std::numeric_limits<PixelType>::max()*/, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cApplyPolyTransferFunction(imageIn,dims,a,b,c,minValue,maxValue,imageOut,device);
}

 int* applyPolyTransferFunction(const int* imageIn, Vec<size_t> dims, double a, double b, double c, int minValue/*=std::numeric_limits<PixelType>::lowest()*/, int maxValue/*=std::numeric_limits<PixelType>::max()*/, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cApplyPolyTransferFunction(imageIn,dims,a,b,c,minValue,maxValue,imageOut,device);
}

 float* applyPolyTransferFunction(const float* imageIn, Vec<size_t> dims, double a, double b, double c, float minValue/*=std::numeric_limits<PixelType>::lowest()*/, float maxValue/*=std::numeric_limits<PixelType>::max()*/, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cApplyPolyTransferFunction(imageIn,dims,a,b,c,minValue,maxValue,imageOut,device);
}

 double* applyPolyTransferFunction(const double* imageIn, Vec<size_t> dims, double a, double b, double c, double minValue/*=std::numeric_limits<PixelType>::lowest()*/, double maxValue/*=std::numeric_limits<PixelType>::max()*/,double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cApplyPolyTransferFunction(imageIn,dims,a,b,c,minValue,maxValue,imageOut,device);
}


 unsigned char* addImageWith(const unsigned char* imageIn1, const unsigned char* imageIn2, Vec<size_t> dims, double additive, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddImageWith(imageIn1,imageIn2,dims,additive,imageOut,device);
}

 unsigned short* addImageWith(const unsigned short* imageIn1, const unsigned short* imageIn2, Vec<size_t> dims, double additive, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddImageWith(imageIn1,imageIn2,dims,additive,imageOut,device);
}

 short* addImageWith(const short* imageIn1, const short* imageIn2, Vec<size_t> dims, double additive, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddImageWith(imageIn1,imageIn2,dims,additive,imageOut,device);
}

 unsigned int* addImageWith(const unsigned int* imageIn1, const unsigned int* imageIn2, Vec<size_t> dims, double additive, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddImageWith(imageIn1,imageIn2,dims,additive,imageOut,device);
}

 int* addImageWith(const int* imageIn1, const int* imageIn2, Vec<size_t> dims, double additive, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddImageWith(imageIn1,imageIn2,dims,additive,imageOut,device);
}

 float* addImageWith(const float* imageIn1, const float* imageIn2, Vec<size_t> dims, double additive, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddImageWith(imageIn1,imageIn2,dims,additive,imageOut,device);
}

 double* addImageWith(const double* imageIn1, const double* imageIn2, Vec<size_t> dims, double additive, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddImageWith(imageIn1,imageIn2,dims,additive,imageOut,device);
}


 unsigned char* contrastEnhancement(const unsigned char* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cContrastEnhancement(imageIn,dims,sigmas,neighborhood,imageOut,device);
}

 unsigned short* contrastEnhancement(const unsigned short* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cContrastEnhancement(imageIn,dims,sigmas,neighborhood,imageOut,device);
}

 short* contrastEnhancement(const short* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cContrastEnhancement(imageIn,dims,sigmas,neighborhood,imageOut,device);
}

 unsigned int* contrastEnhancement(const unsigned int* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cContrastEnhancement(imageIn,dims,sigmas,neighborhood,imageOut,device);
}

 int* contrastEnhancement(const int* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cContrastEnhancement(imageIn,dims,sigmas,neighborhood,imageOut,device);
}

 float* contrastEnhancement(const float* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cContrastEnhancement(imageIn,dims,sigmas,neighborhood,imageOut,device);
}

 double* contrastEnhancement(const double* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cContrastEnhancement(imageIn,dims,sigmas,neighborhood,imageOut,device);
}


 size_t* histogram(const unsigned char* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned char minVal/*=std::numeric_limits<unsigned char>::lowest()*/, unsigned char maxVal/*=std::numeric_limits<unsigned char>::max()*/, int device/*=0*/)
{
	return cCalculateHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}

 size_t* histogram(const unsigned short* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned short minVal/*=std::numeric_limits<unsigned short>::lowest()*/, unsigned short maxVal/*=std::numeric_limits<unsigned short>::max()*/, int device/*=0*/)
{
	return cCalculateHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}

 size_t* histogram(const short* imageIn, Vec<size_t> dims, unsigned int arraySize, short minVal/*=std::numeric_limits<short>::lowest()*/,short maxVal/*=std::numeric_limits<short>::max()*/, int device/*=0*/)
{
	return cCalculateHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}

 size_t* histogram(const unsigned int* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned int minVal/*=std::numeric_limits<unsigned short>::lowest()*/, unsigned int maxVal/*=std::numeric_limits<unsigned int>::max()*/, int device/*=0*/)
{
	return cCalculateHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}

 size_t* histogram(const int* imageIn, Vec<size_t> dims, unsigned int arraySize, int minVal/*=std::numeric_limits<int>::lowest()*/, int maxVal/*=std::numeric_limits<int>::max()*/, int device/*=0*/)
{
	return cCalculateHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}

 size_t* histogram(const float* imageIn, Vec<size_t> dims, unsigned int arraySize, float minVal/*=std::numeric_limits<float>::lowest()*/, float maxVal/*=std::numeric_limits<float>::max()*/, int device/*=0*/)
{
	return cCalculateHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}

 size_t* histogram(const double* imageIn, Vec<size_t> dims, unsigned int arraySize, double minVal/*=std::numeric_limits<double>::lowest()*/, double maxVal/*=std::numeric_limits<double>::max()*/, int device/*=0*/)
{
	return cCalculateHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}


 unsigned char* gaussianFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<float> sigmas, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cGaussianFilter(imageIn,dims,sigmas,imageOut,device);
}

 unsigned short* gaussianFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<float> sigmas, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cGaussianFilter(imageIn,dims,sigmas,imageOut,device);
}

 short* gaussianFilter(const short* imageIn, Vec<size_t> dims, Vec<float> sigmas, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cGaussianFilter(imageIn,dims,sigmas,imageOut,device);
}

 unsigned int* gaussianFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<float> sigmas, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cGaussianFilter(imageIn,dims,sigmas,imageOut,device);
}

 int* gaussianFilter(const int* imageIn, Vec<size_t> dims, Vec<float> sigmas, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cGaussianFilter(imageIn,dims,sigmas,imageOut,device);
}

 float* gaussianFilter(const float* imageIn, Vec<size_t> dims, Vec<float> sigmas, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cGaussianFilter(imageIn,dims,sigmas,imageOut,device);
}

 double* gaussianFilter(const double* imageIn, Vec<size_t> dims, Vec<float> sigmas, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cGaussianFilter(imageIn,dims,sigmas,imageOut,device);
}


 void getMinMax(const unsigned char* imageIn, Vec<size_t> dims, unsigned char& minVal, unsigned char& maxVal, int device/*=0*/)
{
	cGetMinMax(imageIn,dims,minVal,maxVal,device);
}

 void getMinMax(const unsigned short* imageIn, Vec<size_t> dims, unsigned short& minVal, unsigned short& maxVal, int device/*=0*/)
{
	cGetMinMax(imageIn,dims,minVal,maxVal,device);
}

 void getMinMax(const short* imageIn, Vec<size_t> dims, short& minVal, short& maxVal, int device/*=0*/)
{
	cGetMinMax(imageIn,dims,minVal,maxVal,device);
}

 void getMinMax(const unsigned int* imageIn, Vec<size_t> dims, unsigned int& minVal, unsigned int& maxVal, int device/*=0*/)
{
	cGetMinMax(imageIn,dims,minVal,maxVal,device);
}

 void getMinMax(const int* imageIn, Vec<size_t> dims, int& minVal, int& maxVal, int device/*=0*/)
{
	cGetMinMax(imageIn,dims,minVal,maxVal,device);
}

 void getMinMax(const float* imageIn, Vec<size_t> dims, float& minVal, float& maxVal, int device/*=0*/)
{
	cGetMinMax(imageIn,dims,minVal,maxVal,device);
}

 void getMinMax(const double* imageIn, Vec<size_t> dims, double& minVal, double& maxVal, int device/*=0*/)
{
	cGetMinMax(imageIn,dims,minVal,maxVal,device);
}


 unsigned char* imagePow(const unsigned char* imageIn, Vec<size_t> dims, double additive, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cAddConstant(imageIn,dims,additive,imageOut,device);
}

 unsigned short* imagePow(const unsigned short* imageIn, Vec<size_t> dims, double power, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cImagePow(imageIn,dims,power,imageOut,device);
}

 short* imagePow(const short* imageIn, Vec<size_t> dims, double power, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cImagePow(imageIn,dims,power,imageOut,device);
}

 unsigned int* imagePow(const unsigned int* imageIn, Vec<size_t> dims, double power, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cImagePow(imageIn,dims,power,imageOut,device);
}

 int* imagePow(const int* imageIn, Vec<size_t> dims, double power, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cImagePow(imageIn,dims,power,imageOut,device);
}

 float* imagePow(const float* imageIn, Vec<size_t> dims, double power, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cImagePow(imageIn,dims,power,imageOut,device);
}

 double* imagePow(const double* imageIn, Vec<size_t> dims, double power, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cImagePow(imageIn,dims,power,imageOut,device);
}


float* markovRandomFieldDenoiser(const float* imageIn, Vec<size_t> dims, int maxIterations, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMarkovRandomFieldDenoiser(imageIn,dims,maxIterations,imageOut,device);
}

 unsigned char* maxFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMaxFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 unsigned short* maxFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMaxFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 short* maxFilter(const short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMaxFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 unsigned int* maxFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMaxFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 int* maxFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMaxFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 float* maxFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMaxFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 double* maxFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMaxFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}


 unsigned char* meanFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMeanFilter(imageIn,dims,neighborhood,imageOut,device);
}

 unsigned short* meanFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMeanFilter(imageIn,dims,neighborhood,imageOut,device);
}

 short* meanFilter(const short* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMeanFilter(imageIn,dims,neighborhood,imageOut,device);
}

 unsigned int* meanFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMeanFilter(imageIn,dims,neighborhood,imageOut,device);
}

 int* meanFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMeanFilter(imageIn,dims,neighborhood,imageOut,device);
}

 float* meanFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMeanFilter(imageIn,dims,neighborhood,imageOut,device);
}

 double* meanFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMeanFilter(imageIn,dims,neighborhood,imageOut,device);
}


 unsigned char* medianFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMedianFilter(imageIn,dims,neighborhood,imageOut,device);
}

 unsigned short* medianFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMedianFilter(imageIn,dims,neighborhood,imageOut,device);
}

 short* medianFilter(const short* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMedianFilter(imageIn,dims,neighborhood,imageOut,device);
}

 unsigned int* medianFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMedianFilter(imageIn,dims,neighborhood,imageOut,device);
}

 int* medianFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMedianFilter(imageIn,dims,neighborhood,imageOut,device);
}

 float* medianFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMedianFilter(imageIn,dims,neighborhood,imageOut,device);
}

 double* medianFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMedianFilter(imageIn,dims,neighborhood,imageOut,device);
}


 unsigned char* minFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMinFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 unsigned short* minFilter(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMinFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 short* minFilter(const short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMinFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 unsigned int* minFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMinFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 int* minFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMinFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 float* minFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMinFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 double* minFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMinFilter(imageIn,dims,kernelDims,kernel,imageOut,device);
}


 unsigned char* morphologicalClosure(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalClosure(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 unsigned short* morphologicalClosure(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalClosure(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 short* morphologicalClosure(const short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalClosure(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 unsigned int* morphologicalClosure(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalClosure(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 int* morphologicalClosure(const int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalClosure(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 float* morphologicalClosure(const float* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalClosure(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 double* morphologicalClosure(const double* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalClosure(imageIn,dims,kernelDims,kernel,imageOut,device);
}


 unsigned char* morphologicalOpening(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalOpening(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 unsigned short* morphologicalOpening(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalOpening(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 short* morphologicalOpening(const short* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalOpening(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 unsigned int* morphologicalOpening(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalOpening(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 int* morphologicalOpening(const int* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalOpening(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 float* morphologicalOpening(const float* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalOpening(imageIn,dims,kernelDims,kernel,imageOut,device);
}

 double* morphologicalOpening(const double* imageIn, Vec<size_t> dims, Vec<size_t> kernelDims, float* kernel/*=NULL*/, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMorphologicalOpening(imageIn,dims,kernelDims,kernel,imageOut,device);
}


 unsigned char* multiplyImage(const unsigned char* imageIn, Vec<size_t> dims, double multiplier, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImage(imageIn,dims,multiplier,imageOut,device);
}

 unsigned short* multiplyImage(const unsigned short* imageIn, Vec<size_t> dims, double multiplier, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImage(imageIn,dims,multiplier,imageOut,device);
}

 short* multiplyImage(const short* imageIn, Vec<size_t> dims, double multiplier, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImage(imageIn,dims,multiplier,imageOut,device);
}

 unsigned int* multiplyImage(const unsigned int* imageIn, Vec<size_t> dims, double multiplier, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImage(imageIn,dims,multiplier,imageOut,device);
}

 int* multiplyImage(const int* imageIn, Vec<size_t> dims, double multiplier, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImage(imageIn,dims,multiplier,imageOut,device);
}

 float* multiplyImage(const float* imageIn, Vec<size_t> dims, double multiplier, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImage(imageIn,dims,multiplier,imageOut,device);
}

 double* multiplyImage(const double* imageIn, Vec<size_t> dims, double multiplier, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImage(imageIn,dims,multiplier,imageOut,device);
}


 unsigned char* multiplyImageWith(const unsigned char* imageIn1, const unsigned char* imageIn2, Vec<size_t> dims, double factor, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImageWith(imageIn1,imageIn2,dims,factor,imageOut,device);
}

 unsigned short* multiplyImageWith(const unsigned short* imageIn1, const unsigned short* imageIn2, Vec<size_t> dims, double factor, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImageWith(imageIn1,imageIn2,dims,factor,imageOut,device);
}

 short* multiplyImageWith(const short* imageIn1, const short* imageIn2, Vec<size_t> dims, double factor, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImageWith(imageIn1,imageIn2,dims,factor,imageOut,device);
}

 unsigned int* multiplyImageWith(const unsigned int* imageIn1, const unsigned int* imageIn2, Vec<size_t> dims, double factor, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImageWith(imageIn1,imageIn2,dims,factor,imageOut,device);
}

 int* multiplyImageWith(const int* imageIn1, const int* imageIn2, Vec<size_t> dims, double factor, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImageWith(imageIn1,imageIn2,dims,factor,imageOut,device);
}

 float* multiplyImageWith(const float* imageIn1, const float* imageIn2, Vec<size_t> dims, double factor, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImageWith(imageIn1,imageIn2,dims,factor,imageOut,device);
}

 double* multiplyImageWith(const double* imageIn1, const double* imageIn2, Vec<size_t> dims, double factor, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cMultiplyImageWith(imageIn1,imageIn2,dims,factor,imageOut,device);
}


 double normalizedCovariance(const unsigned char* imageIn1, const unsigned char* imageIn2, Vec<size_t> dims, int device/*=0*/)
{
	return cNormalizedCovariance(imageIn1,imageIn2,dims,device);
}

 double normalizedCovariance(const unsigned short* imageIn1, const unsigned short* imageIn2, Vec<size_t> dims, int device/*=0*/)
{
	return cNormalizedCovariance(imageIn1,imageIn2,dims,device);
}

 double normalizedCovariance(const short* imageIn1, const short* imageIn2, Vec<size_t> dims, int device/*=0*/)
{
	return cNormalizedCovariance(imageIn1,imageIn2,dims,device);
}

 double normalizedCovariance(const unsigned int* imageIn1, const unsigned int* imageIn2, Vec<size_t> dims, int device/*=0*/)
{
	return cNormalizedCovariance(imageIn1,imageIn2,dims,device);
}

 double normalizedCovariance(const int* imageIn1, const int* imageIn2, Vec<size_t> dims, int device/*=0*/)
{
	return cNormalizedCovariance(imageIn1,imageIn2,dims,device);
}

 double normalizedCovariance(const float* imageIn1, const float* imageIn2, Vec<size_t> dims, int device/*=0*/)
{
	return cNormalizedCovariance(imageIn1,imageIn2,dims,device);
}

 double normalizedCovariance(const double* imageIn1, const double* imageIn2, Vec<size_t> dims, int device/*=0*/)
{
	return cNormalizedCovariance(imageIn1,imageIn2,dims,device);
}


 double* normalizeHistogram(const unsigned char* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned char minVal/*=std::numeric_limits<unsigned char>::lowest()*/, unsigned char maxVal/*=std::numeric_limits<unsigned char>::max()*/, int device/*=0*/)
{
	return cNormalizeHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}

 double* normalizeHistogram(const unsigned short* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned short minVal/*=std::numeric_limits<unsigned short>::lowest()*/, unsigned short maxVal/*=std::numeric_limits<unsigned short>::max()*/, int device/*=0*/)
{
	return cNormalizeHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}

 double* normalizeHistogram(const short* imageIn, Vec<size_t> dims, unsigned int arraySize, short minVal/*=std::numeric_limits<short>::lowest()*/, short maxVal/*=std::numeric_limits<short>::max()*/, int device/*=0*/)
{
	return cNormalizeHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}

 double* normalizeHistogram(const unsigned int* imageIn, Vec<size_t> dims, unsigned int arraySize, unsigned int minVal/*=std::numeric_limits<unsigned int>::lowest()*/, unsigned int maxVal/*=std::numeric_limits<unsigned int>::max()*/, int device/*=0*/)
{
	return cNormalizeHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}

 double* normalizeHistogram(const int* imageIn, Vec<size_t> dims, unsigned int arraySize, int minVal/*=std::numeric_limits<int>::lowest()*/, int maxVal/*=std::numeric_limits<int>::max()*/, int device/*=0*/)
{
	return cNormalizeHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}

 double* normalizeHistogram(const float* imageIn, Vec<size_t> dims, unsigned int arraySize, float minVal/*=std::numeric_limits<float>::lowest()*/, float maxVal/*=std::numeric_limits<float>::max()*/, int device/*=0*/)
{
	return cNormalizeHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}

 double* normalizeHistogram(const double* imageIn, Vec<size_t> dims, unsigned int arraySize, double minVal/*=std::numeric_limits<double>::lowest()*/, double maxVal/*=std::numeric_limits<double>::max()*/, int device/*=0*/)
{
	return cNormalizeHistogram(imageIn,dims,arraySize,minVal,maxVal,device);
}


 unsigned char* otsuThresholdFilter(const unsigned char* imageIn, Vec<size_t> dims, double alpha/*=1.0*/, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cOtsuThresholdFilter(imageIn,dims,alpha,imageOut,device);
}

 unsigned short* otsuThresholdFilter(const unsigned short* imageIn, Vec<size_t> dims, double alpha/*=1.0*/, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cOtsuThresholdFilter(imageIn,dims,alpha,imageOut,device);
}

 short* otsuThresholdFilter(const short* imageIn, Vec<size_t> dims, double alpha/*=1.0*/, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cOtsuThresholdFilter(imageIn,dims,alpha,imageOut,device);
}

 unsigned int* otsuThresholdFilter(const unsigned int* imageIn, Vec<size_t> dims, double alpha/*=1.0*/, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cOtsuThresholdFilter(imageIn,dims,alpha,imageOut,device);
}

 int* otsuThresholdFilter(const int* imageIn, Vec<size_t> dims, double alpha/*=1.0*/, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cOtsuThresholdFilter(imageIn,dims,alpha,imageOut,device);
}

 float* otsuThresholdFilter(const float* imageIn, Vec<size_t> dims, double alpha/*=1.0*/, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cOtsuThresholdFilter(imageIn,dims,alpha,imageOut,device);
}

 double* otsuThresholdFilter(const double* imageIn, Vec<size_t> dims, double alpha/*=1.0*/, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cOtsuThresholdFilter(imageIn,dims,alpha,imageOut,device);
}


 unsigned char otsuThresholdValue(const unsigned char* imageIn, Vec<size_t> dims, int device/*=0*/)
{
	return cOtsuThresholdValue(imageIn,dims,device);
}

 unsigned short otsuThresholdValue(const unsigned short* imageIn, Vec<size_t> dims, int device/*=0*/)
{
	return cOtsuThresholdValue(imageIn,dims,device);
}

 short otsuThresholdValue(const short* imageIn, Vec<size_t> dims, int device/*=0*/)
{
	return cOtsuThresholdValue(imageIn,dims,device);
}

 unsigned int otsuThresholdValue(const unsigned int* imageIn, Vec<size_t> dims, int device/*=0*/)
{
	return cOtsuThresholdValue(imageIn,dims,device);
}

 int otsuThresholdValue(const int* imageIn, Vec<size_t> dims, int device/*=0*/)
{
	return cOtsuThresholdValue(imageIn,dims,device);
}

 float otsuThresholdValue(const float* imageIn, Vec<size_t> dims, int device/*=0*/)
{
	return cOtsuThresholdValue(imageIn,dims,device);
}

 double otsuThresholdValue(const double* imageIn, Vec<size_t> dims, int device/*=0*/)
{
	return cOtsuThresholdValue(imageIn,dims,device);
}


 unsigned char* reduceImage(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method/*=MEAN*/, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cReduceImage(imageIn,dims,reductions,reducedDims,method,imageOut,device);
}

 unsigned short* reduceImage(const unsigned short* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method/*=MEAN*/, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cReduceImage(imageIn,dims,reductions,reducedDims,method,imageOut,device);
}

 short* reduceImage(const short* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method/*=MEAN*/, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cReduceImage(imageIn,dims,reductions,reducedDims,method,imageOut,device);
}

 unsigned int* reduceImage(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method/*=MEAN*/, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cReduceImage(imageIn,dims,reductions,reducedDims,method,imageOut,device);
}

 int* reduceImage(const int* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method/*=MEAN*/, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cReduceImage(imageIn,dims,reductions,reducedDims,method,imageOut,device);
}

 float* reduceImage(const float* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method/*=MEAN*/, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cReduceImage(imageIn,dims,reductions,reducedDims,method,imageOut,device);
}

 double* reduceImage(const double* imageIn, Vec<size_t> dims, Vec<size_t> reductions, Vec<size_t>& reducedDims, ReductionMethods method/*=MEAN*/, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cReduceImage(imageIn,dims,reductions,reducedDims,method,imageOut,device);
}


 size_t sumArray(const unsigned char* imageIn, size_t n, int device/*=0*/)
{
	return cSumArray<size_t>(imageIn,n,device);
}

 size_t sumArray(const unsigned short* imageIn, size_t n, int device/*=0*/)
{
	return cSumArray<size_t>(imageIn,n,device);
}

 size_t sumArray(const short* imageIn, size_t n, int device/*=0*/)
{
	return cSumArray<size_t>(imageIn,n,device);
}

 size_t sumArray(const unsigned int* imageIn, size_t n, int device/*=0*/)
{
	return cSumArray<size_t>(imageIn,n,device);
}

 size_t sumArray(const int* imageIn, size_t n, int device/*=0*/)
{
	return cSumArray<size_t>(imageIn,n,device);
}

 double sumArray(const float* imageIn, size_t n, int device/*=0*/)
{
	return cSumArray<double>(imageIn,n,device);
}

 double sumArray(const double* imageIn, size_t n, int device/*=0*/)
{
	return cSumArray<double>(imageIn,n,device);
}


unsigned char* segment(const unsigned char* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
 {
	 return cSegment(imageIn,dims,alpha,kernelDims,kernel,imageOut,device);
 }

unsigned short* segment(const unsigned short* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
 {
	 return cSegment(imageIn,dims,alpha,kernelDims,kernel,imageOut,device);
 }

short* segment(const short* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel/*=NULL*/, short** imageOut/*=NULL*/, int device/*=0*/)
 {
	 return cSegment(imageIn,dims,alpha,kernelDims,kernel,imageOut,device);
 }

unsigned int* segment(const unsigned int* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel/*=NULL*/, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
 {
	 return cSegment(imageIn,dims,alpha,kernelDims,kernel,imageOut,device);
 }

int* segment(const int* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel/*=NULL*/, int** imageOut/*=NULL*/, int device/*=0*/)
 {
	 return cSegment(imageIn,dims,alpha,kernelDims,kernel,imageOut,device);
 }

float* segment(const float* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel/*=NULL*/, float** imageOut/*=NULL*/, int device/*=0*/)
 {
	 return cSegment(imageIn,dims,alpha,kernelDims,kernel,imageOut,device);
 }

double* segment(const double* imageIn, Vec<size_t> dims, double alpha, Vec<size_t> kernelDims, float* kernel/*=NULL*/, double** imageOut/*=NULL*/, int device/*=0*/)
 {
	 return cSegment(imageIn,dims,alpha,kernelDims,kernel,imageOut,device);
 }


 unsigned char* thresholdFilter(const unsigned char* imageIn, Vec<size_t> dims, unsigned char thresh, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return cThresholdFilter(imageIn,dims,thresh,imageOut,device);
}

 unsigned short* thresholdFilter(const unsigned short* imageIn, Vec<size_t> dims, unsigned short thresh, unsigned short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cThresholdFilter(imageIn,dims,thresh,imageOut,device);
}

 short* thresholdFilter(const short* imageIn, Vec<size_t> dims, short thresh, short** imageOut/*=NULL*/, int device/*=0*/)
{
	return cThresholdFilter(imageIn,dims,thresh,imageOut,device);
}

 unsigned int* thresholdFilter(const unsigned int* imageIn, Vec<size_t> dims, unsigned int thresh, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cThresholdFilter(imageIn,dims,thresh,imageOut,device);
}

 int* thresholdFilter(const int* imageIn, Vec<size_t> dims, int thresh, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return cThresholdFilter(imageIn,dims,thresh,imageOut,device);
}

 float* thresholdFilter(const float* imageIn, Vec<size_t> dims, float thresh, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return cThresholdFilter(imageIn,dims,thresh,imageOut,device);
}

 double* thresholdFilter(const double* imageIn, Vec<size_t> dims, double thresh, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return cThresholdFilter(imageIn,dims,thresh,imageOut,device);
}

 double variance(const unsigned char* imageIn, Vec<size_t> dims, int device/*=0*/)
 {
	 return cVariance(imageIn,dims,device,(float*)NULL);
 }

 double variance(const unsigned short* imageIn, Vec<size_t> dims, int device/*=0*/)
 {
	 return cVariance(imageIn,dims,device,(float*)NULL);
 }

 double variance(const short* imageIn, Vec<size_t> dims, int device/*=0*/)
 {
	 return cVariance(imageIn,dims,device,(float*)NULL);
 }

 double variance(const unsigned int* imageIn, Vec<size_t> dims, int device/*=0*/)
 {
	 return cVariance(imageIn,dims,device,(float*)NULL);
 }

 double variance(const int* imageIn, Vec<size_t> dims, int device/*=0*/)
 {
	 return cVariance(imageIn,dims,device,(float*)NULL);
 }

 double variance(const float* imageIn, Vec<size_t> dims, int device/*=0*/)
 {
	 return cVariance(imageIn,dims,device,(float*)NULL);
 }

 double variance(const double* imageIn, Vec<size_t> dims, int device/*=0*/)
 {
	 return cVariance(imageIn,dims,device,(double*)NULL);
 }
