#include "CudaKernels.cuh"

__global__ void cudaMaximumIntensityProjection( CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims() && coordinate.z==0)
	{
		DevicePixelType maxVal = 0;
		for (; coordinate.z<imageIn.getDepth(); ++coordinate.z)
		{
			if (maxVal<imageIn[coordinate])
			{
				maxVal = imageIn[coordinate];
			}
		}

		coordinate.z = 0;
		imageOut[coordinate] = maxVal;
	}
}

