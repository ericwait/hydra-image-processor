#pragma once

#ifndef DEVICE_VEC
#define DEVICE_VEC
#include "Vec.h"
#endif
#undef DEVICE_VEC

#include "CudaImageContainer.cuh"

template <class PixelType>
__global__ void cudaMask( const CudaImageContainer<PixelType> imageIn1, const CudaImageContainer<PixelType> imageIn2,
						 CudaImageContainer<PixelType> imageOut, PixelType threshold )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDeviceDims())
	{
		DevicePixelType val=0;

		if (imageIn2[coordinate] <= threshold)
			val = imageIn1[coordinate];

		imageOut[coordinate] = val;
	}
}