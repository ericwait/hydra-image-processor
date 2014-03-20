#include "CudaKernels.cuh"

__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];

__global__ void cudaMaxFilter( CudaImageContainer imageIn, CudaImageContainer imageOut, Vec<size_t> hostKernelDims )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		DevicePixelType maxVal = imageIn[coordinate];
		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = (kernelDims.x+1)/2;
		kernelMidIdx.y = (kernelDims.y+1)/2;
		kernelMidIdx.z = (kernelDims.z+1)/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageIn.getDepth() && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageIn.getHeight() && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageIn.getWidth() && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					if(cudaConstKernel[kernelDims.linearAddressAt(curCoordKrn)]>0)
					{
						maxVal = (DevicePixelType)max((float)maxVal, imageIn[curCoordIm]*
							cudaConstKernel[kernelDims.linearAddressAt(curCoordKrn)]);
					}
				}
			}
		}

		imageOut[coordinate] = maxVal;
	}
}

