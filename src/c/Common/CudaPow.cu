#include "CudaKernels.cuh"

__global__ void cudaPow( CudaImageContainer imageIn, CudaImageContainer imageOut, double p )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
		imageOut[coordinate] = pow((double)imageIn[coordinate],p);
}

