#pragma once
#include "Vec.h"
#include <vector>
#include <limits>
#include "ImageChunk.cuh"
#include "CudaDeviceImages.cuh"
#include "CudaUtilities.cuh"

#include "CudaGaussianFilter.cuh"
#include "CudaAdd.cuh"
#include "CudaMedianFilter.cuh"

template <class PixelType>
PixelType* cContrastEnhancement(const PixelType* imageIn, Vec<size_t> dims, Vec<float> sigmas, Vec<size_t> neighborhood,
							   PixelType** imageOut=NULL, int device=0)
{
	PixelType* imGauss = cGaussianFilter<PixelType>(imageIn,dims,sigmas,NULL,device);

	PixelType* imSub = cAddImageWith<PixelType>(imageIn,imGauss,dims,-1.0,NULL,device);

	delete[] imGauss;

	cMedianFilter(imSub,dims,neighborhood,imageOut,device);

	delete[] imSub;
	
	return imageOut;
}
