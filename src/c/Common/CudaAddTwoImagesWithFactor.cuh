#pragma once

#define DEVICE_VEC
#include "Vec.h"

template <class PixelType>
__global__ void cudaAddTwoImagesWithFactor( CudaImageContainer<PixelType> imageIn1, CudaImageContainer<PixelType> imageIn2,
										   CudaImageContainer<PixelType> imageOut, double factor, PixelType minValue, PixelType maxValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDeviceDims())
	{
		double additive = factor*(double)(imageIn2[coordinate]);
		double outValue = (double)(imageIn1[coordinate]) + additive;

		imageOut[coordinate] = (outValue>(double)maxValue) ? (maxValue) : ((outValue<(double)minValue) ? (minValue) : ((PixelType)outValue));
	}
}