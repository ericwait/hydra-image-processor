#include "CWrappers.cuh"
#include "CudaAdd.cuh"
#include "CudaGaussianFilter.cuh"
#include "CudaMedianFilter.cuh"
#include "CudaMultiplyImage.cuh"
#include "CudaPow.cuh"
#include "CudaSum.cuh"

unsigned char* cAddConstant(const unsigned char* imageIn, Vec<size_t> dims, double additive, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return addConstant(imageIn,dims,additive,imageOut,device);
}

unsigned int* cAddConstant(const unsigned int* imageIn, Vec<size_t> dims, double additive, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return addConstant(imageIn,dims,additive,imageOut,device);
}

int* cAddConstant(const int* imageIn, Vec<size_t> dims, double additive, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return addConstant(imageIn,dims,additive,imageOut,device);
}

float* cAddConstant(const float* imageIn, Vec<size_t> dims, double additive, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return addConstant(imageIn,dims,additive,imageOut,device);
}

double* cAddConstant(const double* imageIn, Vec<size_t> dims, double additive, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return addConstant(imageIn,dims,additive,imageOut,device);
}

unsigned char* cAddImageWith(const unsigned char* imageIn1, const unsigned char* imageIn2, Vec<size_t> dims, double additive,
							 unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return addImageWith(imageIn1,imageIn2,dims,additive,imageOut,device);
}

unsigned int* cAddImageWith(const unsigned int* imageIn1, const unsigned int* imageIn2, Vec<size_t> dims, double additive,
							unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return addImageWith(imageIn1,imageIn2,dims,additive,imageOut,device);
}

int* cAddImageWith(const int* imageIn1, const int* imageIn2, Vec<size_t> dims, double additive, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return addImageWith(imageIn1,imageIn2,dims,additive,imageOut,device);
}

float* cAddImageWith(const float* imageIn1, const float* imageIn2, Vec<size_t> dims, double additive, float** imageOut/*=NULL*/,
					 int device/*=0*/)
{
	return addImageWith(imageIn1,imageIn2,dims,additive,imageOut,device);
}

double* cAddImageWith(const double* imageIn1, const double* imageIn2, Vec<size_t> dims, double additive, double** imageOut/*=NULL*/,
					  int device/*=0*/)
{
	return addImageWith(imageIn1,imageIn2,dims,additive,imageOut,device);
}

unsigned char* cGaussianFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<float> sigmas, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return gaussianFilter(imageIn,dims,sigmas,imageOut,device);
}

unsigned int* cGaussianFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<float> sigmas, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return gaussianFilter(imageIn,dims,sigmas,imageOut,device);
}

int* cGaussianFilter(const int* imageIn, Vec<size_t> dims, Vec<float> sigmas, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return gaussianFilter(imageIn,dims,sigmas,imageOut,device);
}

float* cGaussianFilter(const float* imageIn, Vec<size_t> dims, Vec<float> sigmas, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return gaussianFilter(imageIn,dims,sigmas,imageOut,device);
}

double* cGaussianFilter(const double* imageIn, Vec<size_t> dims, Vec<float> sigmas, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return gaussianFilter(imageIn,dims,sigmas,imageOut,device);
}

unsigned char* cImagePow(const unsigned char* imageIn, Vec<size_t> dims, double additive, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return addConstant(imageIn,dims,additive,imageOut,device);
}

unsigned int* cImagePow(const unsigned int* imageIn, Vec<size_t> dims, double power, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return imagePow(imageIn,dims,power,imageOut,device);
}

int* cImagePow(const int* imageIn, Vec<size_t> dims, double power, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return imagePow(imageIn,dims,power,imageOut,device);
}

float* cImagePow(const float* imageIn, Vec<size_t> dims, double power, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return imagePow(imageIn,dims,power,imageOut,device);
}

double* cImagePow(const double* imageIn, Vec<size_t> dims, double power, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return imagePow(imageIn,dims,power,imageOut,device);
}

unsigned char* cMedianFilter(const unsigned char* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned char** imageOut/*=NULL*/,
							 int device/*=0*/)
{
	return medianFilter(imageIn,dims,neighborhood,imageOut,device);
}

unsigned int* cMedianFilter(const unsigned int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, unsigned int** imageOut/*=NULL*/,
							int device/*=0*/)
{
	return medianFilter(imageIn,dims,neighborhood,imageOut,device);
}

int* cMedianFilter(const int* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return medianFilter(imageIn,dims,neighborhood,imageOut,device);
}

float* cMedianFilter(const float* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return medianFilter(imageIn,dims,neighborhood,imageOut,device);
}

double* cMedianFilter(const double* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return medianFilter(imageIn,dims,neighborhood,imageOut,device);
}

unsigned char* cMultiplyImage(const unsigned char* imageIn, Vec<size_t> dims, double multiplier, unsigned char** imageOut/*=NULL*/, int device/*=0*/)
{
	return multiplyImage(imageIn,dims,multiplier,imageOut,device);
}

unsigned int* cMultiplyImage(const unsigned int* imageIn, Vec<size_t> dims, double multiplier, unsigned int** imageOut/*=NULL*/, int device/*=0*/)
{
	return multiplyImage(imageIn,dims,multiplier,imageOut,device);
}

int* cMultiplyImage(const int* imageIn, Vec<size_t> dims, double multiplier, int** imageOut/*=NULL*/, int device/*=0*/)
{
	return multiplyImage(imageIn,dims,multiplier,imageOut,device);
}

float* cMultiplyImage(const float* imageIn, Vec<size_t> dims, double multiplier, float** imageOut/*=NULL*/, int device/*=0*/)
{
	return multiplyImage(imageIn,dims,multiplier,imageOut,device);
}

double* cMultiplyImage(const double* imageIn, Vec<size_t> dims, double multiplier, double** imageOut/*=NULL*/, int device/*=0*/)
{
	return multiplyImage(imageIn,dims,multiplier,imageOut,device);
}

double cSumArray(const unsigned char* imageIn, size_t n, int device/*=0*/)
{
	return sumArray(imageIn,n,device);
}

double cSumArray(const unsigned int* imageIn, size_t n, int device/*=0*/)
{
	return sumArray(imageIn,n,device);
}

double cSumArray(const int* imageIn, size_t n, int device/*=0*/)
{
	return sumArray(imageIn,n,device);
}

double cSumArray(const float* imageIn, size_t n, int device/*=0*/)
{
	return sumArray(imageIn,n,device);
}

double cSumArray(const double* imageIn, size_t n, int device/*=0*/)
{
	return sumArray(imageIn,n,device);
}
