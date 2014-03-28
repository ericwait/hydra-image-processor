#include "CudaKernels.cuh"

__global__ void cudaPow( CudaImageContainer imageIn1, CudaImageContainer imageOut, double power, DevicePixelType minValue,
							  DevicePixelType maxValue)
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDeviceDims())
	{
		double outValue = pow((double)imageIn1[coordinate], power);
		imageOut[coordinate] = (outValue>maxValue) ? (maxValue) : ((outValue<minValue) ? (minValue) : (outValue));
	}
}
