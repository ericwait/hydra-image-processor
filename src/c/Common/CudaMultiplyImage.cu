#include "CudaKernels.cuh"

__global__ void cudaMultiplyImage( CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, double factor,
								  DevicePixelType minValue, DevicePixelType maxValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		double outValue = factor*imageIn[coordinate];
		imageOut[coordinate] = (outValue>maxValue) ? (maxValue) : ((outValue<minValue) ? (minValue) : (outValue));
	}
}

