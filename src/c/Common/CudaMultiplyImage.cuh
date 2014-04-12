#pragma once
#define DEVICE_VEC
#include "Vec.h"
#include "CudaImageContainer.cuh"

template <class PixelType>
__global__ void cudaMultiplyImageScaler( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut, double factor,
								  PixelType minValue, PixelType maxValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		double outValue = factor*imageIn[coordinate];
		imageOut[coordinate] = (outValue>maxValue) ? (maxValue) : ((outValue<minValue) ? (minValue) : (outValue));
	}
}

template <class PixelType>
__global__ void cudaMultiplyTwoImages(CudaImageContainer<PixelType> imageIn1, CudaImageContainer<PixelType> imageIn2,
									  CudaImageContainer<PixelType> imageOut, double factor, PixelType minValue,
									  PixelType maxValue)
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDeviceDims())
	{
		double outValue = factor * (double)(imageIn1[coordinate]) * (double)(imageIn2[coordinate]);
		imageOut[coordinate] = (outValue>(double)maxValue) ? (maxValue) : ((outValue<(double)minValue) ? (minValue) : (outValue));
	}
}