#pragma once

#include "CudaImageContainer.cuh"

#ifndef CUDA_CONST_KERNEL
#define CUDA_CONST_KERNEL
__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
#endif

template <class PixelType>
__global__ void cudaMultAddFilter( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut,
								  Vec<size_t> hostKernelDims, size_t kernelOffset=0, bool normalize=true)
{
	Vec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDims())
	{
		double val = 0;
		double kernFactor = 0;
		int kernHits = 0;

		PixelType localMaxVal = imageIn[coordinate];
		Vec<size_t> kernelDims = hostKernelDims;

		Vec<int> startLimit = Vec<int>(coordinate) - Vec<int>((kernelDims)/2);
		Vec<size_t> endLimit = coordinate + (kernelDims+1)/2;
		Vec<size_t> kernelStart(Vec<int>::max(-startLimit,Vec<int>(0,0,0)));

		startLimit = Vec<int>::max(startLimit,Vec<int>(0,0,0));
		endLimit = Vec<size_t>::min(Vec<size_t>(endLimit),imageIn.getDims());

		Vec<size_t> imageStart(coordinate-(kernelDims/2)+kernelStart);
		Vec<size_t> iterationEnd(endLimit-Vec<size_t>(startLimit));

		Vec<size_t> i(0,0,0);
		for (i.z=0; i.z<iterationEnd.z; ++i.z)
		{
			for (i.y=0; i.y<iterationEnd.y; ++i.y)
			{
				for (i.x=0; i.x<iterationEnd.x; ++i.x)
				{
					double kernVal = double(cudaConstKernel[kernelDims.linearAddressAt(kernelStart+i)+kernelOffset]);

					kernFactor += kernVal;
					val += double((imageIn[imageStart+i]) * kernVal);

					++kernHits;
				}
			}
		}

		if (normalize)
		{
			imageOut[coordinate] = (PixelType)(val/kernFactor);
		}
		else
		{
			kernFactor = double(kernHits)/kernelDims.product();
			imageOut[coordinate] = (PixelType)(val/kernFactor);
		}
	}
}