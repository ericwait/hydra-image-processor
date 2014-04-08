#include "CudaKernels.cuh"

__global__ void cudaMedianImageReduction( CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, Vec<size_t> hostReductions)
{
	extern __shared__ DevicePixelType vals[];
	DeviceVec<size_t> reductions = hostReductions;
	DeviceVec<size_t> coordinateOut;
	coordinateOut.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinateOut.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinateOut.z = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
	offset *=  reductions.product();

	if (coordinateOut<imageOut.getDeviceDims())
	{
		int kernelVolume = 0;
		DeviceVec<size_t> mins(coordinateOut*DeviceVec<size_t>(reductions));
		DeviceVec<size_t> maxs = DeviceVec<size_t>::min(mins+reductions, imageIn.getDeviceDims());

		DeviceVec<size_t> currCorrdIn(mins);
		for (currCorrdIn.z=mins.z; currCorrdIn.z<maxs.z; ++currCorrdIn.z)
		{
			for (currCorrdIn.y=mins.y; currCorrdIn.y<maxs.y; ++currCorrdIn.y)
			{
				for (currCorrdIn.x=mins.x; currCorrdIn.x<maxs.x; ++currCorrdIn.x)
				{
					vals[offset+kernelVolume] = imageIn[currCorrdIn];
					++kernelVolume;
				}
			}
		}
		imageOut[coordinateOut] = cudaFindMedian(vals+offset,kernelVolume);
	}
	__syncthreads();
}

