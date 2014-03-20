#include "CudaKernels.cuh"

__global__ void cudaThresholdImage( CudaImageContainer imageIn, CudaImageContainer imageOut, DevicePixelType threshold,
								   DevicePixelType minValue, DevicePixelType maxValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		if (imageIn[coordinate]>=threshold)
			imageOut[coordinate] = maxValue;
		else
			imageOut[coordinate] = minValue;
	}
}

