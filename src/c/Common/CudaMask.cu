#include "CudaKernels.cuh"

__global__ void cudaMask( const CudaImageContainer imageIn1, const CudaImageContainer imageIn2, CudaImageContainer imageOut,
						 DevicePixelType threshold )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDeviceDims())
	{
		DevicePixelType val=0;

		if (imageIn2[coordinate] <= threshold)
			val = imageIn1[coordinate];

		imageOut[coordinate] = val;
	}
}

