#pragma once

#ifndef DEVICE_VEC
#define DEVICE_VEC
#include "Vec.h"
#endif
#undef DEVICE_VEC

#include "CudaImageContainer.cuh"

template <class PixelType>
__device__ PixelType* SubDivide(PixelType* pB, PixelType* pE)
{
	PixelType* pPivot = --pE;
	const PixelType pivot = *pPivot;

	while (pB < pE)
	{
		if (*pB > pivot)
		{
			--pE;
			PixelType temp = *pB;
			*pB = *pE;
			*pE = temp;
		} else
			++pB;
	}

	PixelType temp = *pPivot;
	*pPivot = *pE;
	*pE = temp;

	return pE;
}

template <class PixelType>
__device__ void SelectElement(PixelType* pB, PixelType* pE, size_t k)
{
	while (true)
	{
		PixelType* pPivot = SubDivide(pB, pE);
		size_t n = pPivot - pB;

		if (n == k)
			break;

		if (n > k)
			pE = pPivot;
		else
		{
			pB = pPivot + 1;
			k -= (n + 1);
		}
	}
}

template <class PixelType>
__device__ PixelType cudaFindMedian(PixelType* vals, int numVals)
{
	SelectElement(vals,vals+numVals, numVals/2);
	return vals[numVals/2];
}

template <class PixelType>
__global__ void cudaMedianFilter( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut,
								 Vec<size_t> hostKernelDims )
{
	extern __shared__ DevicePixelType vals[];
	DeviceVec<size_t> kernelDims = hostKernelDims;
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
	offset *=  kernelDims.product();

	if (coordinate<imageIn.getDeviceDims())
	{
		int kernelVolume = 0;
		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> halfKernal = kernelDims/2;

		DeviceVec<size_t> curCoordIm = coordinate - halfKernal;
		curCoordIm.z = (coordinate.z<halfKernal.z) ? 0 : coordinate.z-halfKernal.z;
		for (; curCoordIm.z<=coordinate.z+halfKernal.z && curCoordIm.z<imageIn.getDeviceDims().z; ++curCoordIm.z)
		{
			curCoordIm.y = (coordinate.y<halfKernal.y) ? 0 : coordinate.y-halfKernal.y/2;
			for (; curCoordIm.y<=coordinate.y+halfKernal.y && curCoordIm.y<imageIn.getDeviceDims().y; ++curCoordIm.y)
			{
				curCoordIm.x = (coordinate.x<halfKernal.x) ? 0 : coordinate.x-halfKernal.x/2;
				for (; curCoordIm.x<=coordinate.x+halfKernal.x && curCoordIm.x<imageIn.getDeviceDims().x; ++curCoordIm.x)
				{
					vals[kernelVolume+offset] = imageIn[curCoordIm];
					++kernelVolume;
				}
			}
		}

		imageOut[coordinate] = cudaFindMedian(vals+offset,kernelVolume);
	}
	__syncthreads();
}