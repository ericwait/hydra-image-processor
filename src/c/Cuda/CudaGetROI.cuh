#pragma once

#include "CudaImageContainer.cuh"

template <class PixelType>
__global__ void cudaGetROI( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut,
						   Vec<size_t> hostStartPos, Vec<size_t> hostNewSize )
{
    cudaSetDevice(device);

	Vec<size_t> newSize = hostNewSize;
	Vec<size_t> startPos = hostStartPos;
	Vec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate>=startPos && coordinate<startPos+newSize && coordinate<imageIn.getDims())
	{
		imageOut[coordinate-startPos] = imageIn[coordinate];
	}
}