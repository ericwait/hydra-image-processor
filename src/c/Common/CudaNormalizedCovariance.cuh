#pragma once

#include "CudaSum.cuh"
#include "CudaAdd.cuh"
#include "CudaPow.cuh"
#include "CudaMultiplyImage.cuh"

template <class PixelType>
double normalizedCovariance(const PixelType* imageIn1, const PixelType* imageIn2, Vec<size_t> dims, int device=0)
{
	double im1Mean = sumArray(imageIn1,dims.product(),device) / (double)dims.product();
	double im2Mean = sumArray(imageIn2,dims.product(),device) / (double)dims.product();

	float* im1Sub = addConstant<PixelType,float>(imageIn1,dims,-1.0*im1Mean,NULL,device);
	float* im2Sub = addConstant<PixelType,float>(imageIn2,dims,-1.0*im2Mean,NULL,device);

	float* im1P = imagePow<float>(im1Sub,dims,2.0,NULL,device);
	float* im2P = imagePow<float>(im2Sub,dims,2.0,NULL,device);

	double sigma1 = sqrt(sumArray(im1P,dims.product(),device)/(double)dims.product());
	double sigma2 = sqrt(sumArray(im2P,dims.product(),device)/(double)dims.product());

	delete[] im1P;
	delete[] im2P;

	float* imMul = multiplyImageWith<float>(im1Sub,im2Sub,dims,1.0,NULL,device);
	double numerator = sumArray(imMul,dims.product(),device);

	double coVar = numerator/(dims.product()*sigma1*sigma2);

	delete[] im1Sub;
	delete[] im2Sub;
	delete[] imMul;

	return coVar;
}
