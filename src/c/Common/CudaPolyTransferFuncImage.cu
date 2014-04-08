#include "CudaKernels.cuh"

__global__ void cudaPolyTransferFuncImage( CudaImageContainer<DevicePixelType> imageIn, CudaImageContainer<DevicePixelType> imageOut, double a, double b, double c,
										  DevicePixelType minPixelValue, DevicePixelType maxPixelValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn.getDeviceDims())
	{
		double pixVal = (double)imageIn[coordinate] / maxPixelValue;// place value between [0,1]
		double multiplier = a*pixVal*pixVal + b*pixVal + c;
		if (multiplier<0)
			multiplier = 0;
		if (multiplier>1)
			multiplier = 1;

		DevicePixelType newPixelVal = min((double)maxPixelValue,max((double)minPixelValue, multiplier*maxPixelValue));

		imageOut[coordinate] = newPixelVal;
	}
}

