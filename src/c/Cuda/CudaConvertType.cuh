#pragma once

#include "Vec.h"

template <class PixelTypeIn, class PixelTypeOut>
__global__ void cudaConvertType( const PixelTypeIn* imageIn, PixelTypeOut* imageOut, size_t imSize, PixelTypeOut minValue,
						PixelTypeOut maxValue)
{
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i<imSize)
	{
		PixelTypeOut outValue = (PixelTypeOut)(imageIn[i]);
		imageOut[i] = (outValue>maxValue) ? (maxValue) : ((outValue<minValue) ? (minValue) : (outValue));
	}
}

// template <class PixelType>
// __global__ void cudaConvertType( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut, PixelType minValue,
// 								PixelType maxValue)
// {
// 	Vec<size_t> coordinate;
// 	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
// 	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
// 	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;
// 
// 	if (coordinate<imageIn.getDims())
// 		imageOut[coordinate] = imageIn[coordinate];
// }
