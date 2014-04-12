#pragma once
#define DEVICE_VEC
#include "Vec.h"
#include "CudaImageContainer.cuh"

template <class PixelType>
__global__ void cudaMaxFilter( CudaImageContainer<PixelType> imageIn, CudaImageContainer<PixelType> imageOut, Vec<size_t> hostKernelDims,
							  PixelType minVal, PixelType maxVal)
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		double localMaxVal = imageIn[coordinate];
		DeviceVec<size_t> kernelDims = hostKernelDims;

		DeviceVec<int> startLimit = DeviceVec<int>(coordinate) - DeviceVec<int>((kernelDims)/2);
		DeviceVec<size_t> endLimit = coordinate + (kernelDims+1)/2;
		DeviceVec<size_t> kernelStart(DeviceVec<int>::max(-startLimit,DeviceVec<int>(0,0,0)));

		startLimit = DeviceVec<int>::max(startLimit,DeviceVec<int>(0,0,0));
		endLimit = DeviceVec<size_t>::min(DeviceVec<size_t>(endLimit),imageIn.getDeviceDims());

		DeviceVec<size_t> imageStart(coordinate-(kernelDims/2)+kernelStart);
		DeviceVec<size_t> iterationEnd(endLimit-DeviceVec<size_t>(startLimit));

		DeviceVec<size_t> i(0,0,0);
		for (i.z=0; i.z<iterationEnd.z; ++i.z)
		{
			for (i.y=0; i.y<iterationEnd.y; ++i.y)
			{
				for (i.x=0; i.x<iterationEnd.x; ++i.x)
				{
					if (cudaConstKernel[kernelDims.linearAddressAt(kernelStart+i)]==0)
						continue;

					double temp = imageIn[imageStart+i] * cudaConstKernel[kernelDims.linearAddressAt(kernelStart+i)];
					if (temp>localMaxVal)
					{
						localMaxVal = temp;
					}
				}
			}
		}

		imageOut[coordinate] = (localMaxVal>maxVal) ? (maxVal) : ((PixelType)localMaxVal);
	}
}