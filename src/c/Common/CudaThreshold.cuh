#pragma once

#ifndef DEVICE_VEC
#define DEVICE_VEC
#include "Vec.h"
#endif
#undef DEVICE_VEC

#include "CudaImageContainer.cuh"

template <class PixelType>
__global__ void cudaThreshold( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut, PixelType threshold,
							  PixelType minValue, PixelType maxValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		imageOut[coordinate] = (imageIn[coordinate]>=threshold) ? (maxValue) : (minValue);
	}
}