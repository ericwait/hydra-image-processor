#pragma once

#include "CudaSum.cuh"
#include "CudaAdd.cuh"
#include "CudaPow.cuh"
#include "CudaMultiplyImage.cuh"

template <class PixelType>
double cNormalizedCovariance(const PixelType* imageIn1, const PixelType* imageIn2, Vec<size_t> dims, int device=0)
{
	double im1Mean = cSumArray<double>(imageIn1,dims.product(),device) / (double)dims.product();
	double im2Mean = cSumArray<double>(imageIn2,dims.product(),device) / (double)dims.product();

	float* im1Sub = cAddConstant<PixelType,float>(imageIn1,dims,-1.0*im1Mean,NULL,device);
	float* im2Sub = cAddConstant<PixelType,float>(imageIn2,dims,-1.0*im2Mean,NULL,device);

	float* im1P = cImagePow<float>(im1Sub,dims,2.0,NULL,device);
	float* im2P = cImagePow<float>(im2Sub,dims,2.0,NULL,device);

	double sigma1 = sqrt(cSumArray<double>(im1P,dims.product(),device)/(double)dims.product());
	double sigma2 = sqrt(cSumArray<double>(im2P,dims.product(),device)/(double)dims.product());

	delete[] im1P;
	delete[] im2P;

	float* imMul = cMultiplyImageWith<float>(im1Sub,im2Sub,dims,1.0,NULL,device);
	double numerator = cSumArray<double>(imMul,dims.product(),device);

	double coVar = numerator/(dims.product()*sigma1*sigma2);

	delete[] im1Sub;
	delete[] im2Sub;
	delete[] imMul;

	return coVar;
}

double cNormalizedCovariance(const double* imageIn1, const double* imageIn2, Vec<size_t> dims, int device=0)
{
	double im1Mean = cSumArray<double>(imageIn1,dims.product(),device) / (double)dims.product();
	double im2Mean = cSumArray<double>(imageIn2,dims.product(),device) / (double)dims.product();

	double* im1Sub = cAddConstant<double,double>(imageIn1,dims,-1.0*im1Mean,NULL,device);
	double* im2Sub = cAddConstant<double,double>(imageIn2,dims,-1.0*im2Mean,NULL,device);

	double* im1P = cImagePow<double>(im1Sub,dims,2.0,NULL,device);
	double* im2P = cImagePow<double>(im2Sub,dims,2.0,NULL,device);

	double sigma1 = sqrt(cSumArray<double>(im1P,dims.product(),device)/(double)dims.product());
	double sigma2 = sqrt(cSumArray<double>(im2P,dims.product(),device)/(double)dims.product());

	delete[] im1P;
	delete[] im2P;

	double* imMul = cMultiplyImageWith<double>(im1Sub,im2Sub,dims,1.0,NULL,device);
	double numerator = cSumArray<double>(imMul,dims.product(),device);

	double coVar = numerator/(dims.product()*sigma1*sigma2);

	delete[] im1Sub;
	delete[] im2Sub;
	delete[] imMul;

	return coVar;
}
