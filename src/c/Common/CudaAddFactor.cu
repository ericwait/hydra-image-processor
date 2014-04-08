#include "CudaKernels.cuh"

__global__ void cudaAddFactor( CudaImageContainer<DevicePixelType> imageIn1, CudaImageContainer<DevicePixelType> imageOut, double factor, DevicePixelType minValue,
							  DevicePixelType maxValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDeviceDims())
	{
		double outValue = imageIn1[coordinate] + factor;
		imageOut[coordinate] = (outValue>maxValue) ? (maxValue) : ((outValue<minValue) ? (minValue) : (outValue));
	}
}
