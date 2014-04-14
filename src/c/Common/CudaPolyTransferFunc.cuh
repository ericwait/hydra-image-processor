#pragma once

#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC

#include "CudaImageContainer.cuh"

template <class PixelType>
__global__ void cudaPolyTransferFunc( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut, double a, double b,
									 double c, PixelType minPixelValue, PixelType maxPixelValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		double pixVal = (double)imageIn[coordinate] / maxPixelValue;// place value between [0,1]
		double multiplier = a*pixVal*pixVal + b*pixVal + c;
		if (multiplier<0)
			multiplier = 0;
		if (multiplier>1)
			multiplier = 1;

		PixelType newPixelVal = min((double)maxPixelValue,max((double)minPixelValue, multiplier*maxPixelValue));

		imageOut[coordinate] = newPixelVal;
	}
}