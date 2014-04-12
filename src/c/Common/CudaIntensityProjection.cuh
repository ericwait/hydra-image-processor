#pragma once

#ifndef DEVICE_VEC
#define DEVICE_VEC
#include "Vec.h"
#endif
#undef DEVICE_VEC

#include "CudaImageContainer.cuh"

template <class PixelType>
__global__ void cudaMaximumIntensityProjection( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims() && coordinate.z==0)
	{
		PixelType maxVal = imageIn[coordinate];
		for (; coordinate.z<imageIn.getDepth(); ++coordinate.z)
		{
			if (maxVal<imageIn[coordinate])
			{
				maxVal = imageIn[coordinate];
			}
		}

		imageOut[coordinate] = maxVal;
	}
}

template <class PixelType>
__global__ void cudaMeanIntensityProjection( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims() && coordinate.z==0)
	{
		double val = imageIn[coordinate];
		for (; coordinate.z<imageIn.getDepth(); ++coordinate.z)
		{
				val = imageIn[coordinate];
		}

		imageOut[coordinate] = val/imageIn.getDepth();
	}
}