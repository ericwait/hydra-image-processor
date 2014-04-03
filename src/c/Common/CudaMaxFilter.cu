#include "CudaKernels.cuh"

__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];

__global__ void cudaMaxFilter( CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostKernelDims, DevicePixelType  minVal,
							  DevicePixelType maxVal)
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		DevicePixelType localMaxVal = imageIn[coordinate];
		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> kernalOffset(0,0,0);

		DeviceVec<int> tmp = DeviceVec<int>(coordinate) - DeviceVec<int>((kernelDims+1)/2);

		if (tmp.x<0)
		{
			kernalOffset.x = -1 * tmp.x;
			tmp.x = 0;
		}

		if (tmp.y<0)
		{
			kernalOffset.y = -1 * tmp.y;
			tmp.y = 0;
		}

		if (tmp.z<0)
		{
			kernalOffset.z = -1 * tmp.z;
			tmp.z = 0;
		}

		DeviceVec<size_t> startCoord = tmp;

		//find if the kernel will go off the edge of the image
		DeviceVec<size_t> offset(0,0,0);
 		for (offset.z=0; startCoord.z+offset.z<imageOut.getDeviceDims().z && offset.z<kernelDims.z; ++offset.z)
 		{
			for (offset.y=0; startCoord.y+offset.y<imageOut.getDeviceDims().y && offset.y<kernelDims.y; ++offset.y)
			{
 				for (offset.x=0; startCoord.x+offset.x<imageOut.getDeviceDims().x && offset.x<kernelDims.x; ++offset.x)
 				{
					if (cudaConstKernel[kernelDims.linearAddressAt(offset)]==0)
						continue;

 					float temp = imageIn[startCoord+offset] * cudaConstKernel[kernelDims.linearAddressAt(offset)];
 					if (temp>localMaxVal)
					{
						localMaxVal = temp;
					}
 				}
 			}
 		}

		imageOut[coordinate] = (localMaxVal>maxVal) ? (maxVal) : (localMaxVal);
	}
}

