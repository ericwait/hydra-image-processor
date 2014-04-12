#pragma once
#define DEVICE_VEC
#include "Vec.h"
#include "CudaImageContainer.cuh"

template <class PixelType>
__global__ void cudaPow( CudaImageContainer<PixelType> imageIn1, CudaImageContainer<PixelType> imageOut, double power, PixelType minValue,
						PixelType maxValue)
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDeviceDims())
	{
		double outValue = pow((double)imageIn1[coordinate], power);
		imageOut[coordinate] = (outValue>maxValue) ? (maxValue) : ((outValue<minValue) ? (minValue) : (outValue));
	}
}