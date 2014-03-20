#include "CudaKernels.cuh"

__global__ void cudaMultAddFilter( CudaImageContainer* imageIn, CudaImageContainer* imageOut, Vec<size_t> hostKernelDims, size_t kernelOffset/*=0*/ )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn->getDeviceDims())
	{
		double val = 0;
		double kernFactor = 0;

		DeviceVec<size_t> kernelDims = hostKernelDims;
		DeviceVec<size_t> kernelMidIdx;
		DeviceVec<size_t> curCoordIm; 
		DeviceVec<size_t> curCoordKrn;

		kernelMidIdx.x = kernelDims.x/2;
		kernelMidIdx.y = kernelDims.y/2;
		kernelMidIdx.z = kernelDims.z/2;

		//find if the kernel will go off the edge of the image
		curCoordIm.z = (size_t) max(0,(int)coordinate.z-(int)kernelMidIdx.z);
		curCoordKrn.z = ((int)coordinate.z-(int)kernelMidIdx.z>=0) ? (0) : (kernelMidIdx.z-coordinate.z);
		for (; curCoordIm.z<imageIn->getDepth() && curCoordKrn.z<kernelDims.z; ++curCoordIm.z, ++curCoordKrn.z)
		{
			curCoordIm.y = (size_t)max(0,(int)coordinate.y-(int)kernelMidIdx.y);
			curCoordKrn.y = ((int)coordinate.y-(int)kernelMidIdx.y>=0) ? (0) : (kernelMidIdx.y-coordinate.y);
			for (; curCoordIm.y<imageIn->getHeight() && curCoordKrn.y<kernelDims.y; ++curCoordIm.y, ++curCoordKrn.y)
			{
				curCoordIm.x = (size_t)max(0,(int)coordinate.x-(int)kernelMidIdx.x);
				curCoordKrn.x = ((int)coordinate.x-(int)kernelMidIdx.x>=0) ? (0) : (kernelMidIdx.x-coordinate.x);		
				for (; curCoordIm.x<imageIn->getWidth() && curCoordKrn.x<kernelDims.x; ++curCoordIm.x, ++curCoordKrn.x)
				{
					size_t kernIdx = kernelDims.linearAddressAt(curCoordKrn)+kernelOffset;
					kernFactor += cudaConstKernel[kernIdx];
					val += (*imageIn)[curCoordIm] * cudaConstKernel[kernIdx];
				}
			}
		}

		(*imageOut)[coordinate] = val/kernFactor;
	}
}

