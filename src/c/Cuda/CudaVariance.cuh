#pragma once
#include "Vec.h"
#include "CudaSum.cuh"
#include "CudaAdd.cuh"
#include "CudaPow.cuh"
#include "CudaConvertType.cuh"

template <class PixelType, class PixelTypeOut>
double cVariance(const PixelType* imageIn, Vec<size_t> dims, int device=0, PixelTypeOut* imageOut=NULL)
{
	double variance = 0.0;

	double imMean = cSumArray<double>(imageIn, dims.product(), device) / (double)dims.product();
	PixelTypeOut* imSub = cAddConstant<PixelType, PixelTypeOut>(imageIn, dims, -imMean, NULL, device);
	PixelTypeOut* imP = cImagePow<PixelTypeOut>(imSub, dims, 2.0, NULL, device);
	variance = cSumArray<double>(imP, dims.product(), device) / (double)dims.product();

	if (imageOut != NULL)
	{
		memcpy(imageOut, imSub, sizeof(PixelTypeOut)*dims.product());
	}

	delete[] imSub;
	delete[] imP;

	return variance;
}
