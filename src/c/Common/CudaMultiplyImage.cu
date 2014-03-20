#include "CudaKernels.cuh"

__global__ void cudaMultiplyImage( CudaImageContainer imageIn, CudaImageContainer imageOut, double factor, DevicePixelType minValue,
								  DevicePixelType maxValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		imageOut[coordinate] = min((double)maxValue,max((double)minValue, factor*imageIn[coordinate]));
	}
}

