#pragma once

#include "CudaImageContainer.cuh"

template <class PixelType>
__global__ void cudaMask( const CudaImageContainer<PixelType> imageIn1, const CudaImageContainer<PixelType> imageIn2,
						 CudaImageContainer<PixelType> imageOut, PixelType threshold )
{
	Vec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDims())
	{
		PixelType val=0;

		if (imageIn2[coordinate] <= threshold)
			val = imageIn1[coordinate];

		imageOut[coordinate] = val;
	}
}