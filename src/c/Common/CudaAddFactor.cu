#include "CudaKernels.cuh"

__global__ void cudaAddFactor( CudaImageContainer imageIn1, CudaImageContainer imageOut, double factor, DevicePixelType minValue,
							  DevicePixelType maxValue )
{
	DeviceVec<size_t> coordinate;
	coordinate.x = threadIdx.x + blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y + blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z + blockIdx.z * blockDim.z;

	if (coordinate<imageIn1.getDeviceDims())
	{
		double outValue = imageIn1[coordinate] + factor;
		//size_t idxIn1 = imageIn1.getDeviceDims().linearAddressAt(coordinate,imageIn1.isColumnMajor());
		//size_t idxOut = imageOut.getDeviceDims().linearAddressAt(coordinate,imageOut.isColumnMajor());
		imageOut[coordinate] = min((double)maxValue,max((double)minValue,outValue));
// 		DevicePixelType* im = imageOut.getDeviceImagePointer();
// 		size_t idx = coordinate.x+coordinate.y*imageOut.getWidth()+coordinate.z*imageOut.getHeight();
// 		size_t calcIdx = imageOut.getDeviceDims().linearAddressAt(coordinate);
// 		im[idx] = coordinate.x;
	}
}

