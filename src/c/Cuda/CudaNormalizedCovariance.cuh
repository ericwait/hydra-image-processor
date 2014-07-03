#pragma once

#include "CudaSum.cuh"
#include "CudaAdd.cuh"
#include "CudaPow.cuh"
#include "CudaMultiplyImage.cuh"
#include "CudaVariance.cuh"

template <class PixelType>
double cNormalizedCovariance(const PixelType* imageIn1, const PixelType* imageIn2, Vec<size_t> dims, int device=0)
{
	float* im1Sub = new float[dims.product()];
	float* im2Sub = new float[dims.product()];

	double sigma1 = sqrt(cVariance(imageIn1,dims,device,im1Sub));
	double sigma2 = sqrt(cVariance(imageIn2,dims,device,im2Sub));

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
	double* im1Sub = new double[dims.product()];
	double* im2Sub = new double[dims.product()];

	double sigma1 = sqrt(cVariance(imageIn1,dims,device,im1Sub));
	double sigma2 = sqrt(cVariance(imageIn2,dims,device,im2Sub));

	double* imMul = cMultiplyImageWith<double>(im1Sub,im2Sub,dims,1.0,NULL,device);
	double numerator = cSumArray<double>(imMul,dims.product(),device);

	double coVar = numerator/(dims.product()*sigma1*sigma2);

	delete[] im1Sub;
	delete[] im2Sub;
	delete[] imMul;

	return coVar;
}
