#include "CudaKernels.cuh"

__global__ void cudaMeanImageReduction(CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut,
									   Vec<size_t> hostReductions)
{
	DeviceVec<size_t> reductions = hostReductions;
	DeviceVec<size_t> coordinateOut;
	coordinateOut.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinateOut.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinateOut.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinateOut<imageOut.getDeviceDims())
	{
		int kernelVolume = 0;
		double val = 0;
		DeviceVec<size_t> mins(coordinateOut*reductions);
		DeviceVec<size_t> maxs = DeviceVec<size_t>::min(mins+reductions, imageIn.getDeviceDims());

		DeviceVec<size_t> currCorrdIn(mins);
		for (currCorrdIn.z=mins.z; currCorrdIn.z<maxs.z; ++currCorrdIn.z)
		{
			for (currCorrdIn.y=mins.y; currCorrdIn.y<maxs.y; ++currCorrdIn.y)
			{
				for (currCorrdIn.x=mins.x; currCorrdIn.x<maxs.x; ++currCorrdIn.x)
				{
					val += (double)imageIn[currCorrdIn];
					++kernelVolume;
				}
			}
		}

		imageOut[coordinateOut] = val/kernelVolume;
	}
}

