#include "CudaKernels.cuh"

__global__ void cudaMultiplyTwoImages( CudaImageContainer imageIn1, CudaImageContainer imageIn2, CudaImageContainer imageOut )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDeviceDims())
	{
		DevicePixelType val1 = imageIn1[coordinate];
		DevicePixelType val2 = imageIn2[coordinate];
		imageOut[coordinate] = imageIn1[coordinate] * imageIn2[coordinate];
	}
}

