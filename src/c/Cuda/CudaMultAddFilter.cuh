#pragma once

#include "CudaImageContainer.cuh"
#include "KernelIterator.cuh"
#include "float.h"

#ifndef CUDA_CONST_KERNEL
#define CUDA_CONST_KERNEL
__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
#endif

template <class PixelType>
__global__ void cudaMultAddFilter( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut,
								  Vec<size_t> hostKernelDims, size_t kernelOffset=0, bool normalize=true, float* globalKernel=NULL)
{
	Vec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	const float* convKernel = NULL;
	if(NULL!=globalKernel)
		convKernel = globalKernel;
	else
		convKernel = cudaConstKernel;

	if (coordinate<imageIn.getDims())
	{
		double val = 0;
		double kernFactor = 0;
		int kernHits = 0;

		PixelType localMaxVal = imageIn(coordinate);
		Vec<size_t> kernelDims = hostKernelDims;
		KernelIterator kIt(coordinate, imageIn.getDims(), kernelDims);

		for(; !kIt.end(); ++kIt)
		{
			Vec<size_t> kernIdx(kIt.getKernelIdx());
			float kernVal = cudaConstKernel[kernelDims.linearAddressAt(kernIdx)];

			if(kernVal<=FLT_MIN && kernVal>=FLT_MIN)//float zero
				continue;

			kernFactor += kernVal;
			val += double((imageIn(kIt.getImageCoordinate())) * kernVal);

			++kernHits;
		}

		if (normalize)
		{
			imageOut(coordinate) = (PixelType)(val/kernFactor);
		}
		else
		{
			kernFactor = double(kernHits)/kernelDims.product();
			imageOut(coordinate) = (PixelType)(val/kernFactor);
		}
	}
}