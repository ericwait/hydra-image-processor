#include "CWrappers.cuh"
#include "CudaAdd.cuh"
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
