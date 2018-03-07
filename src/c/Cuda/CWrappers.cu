#include "CWrappers.h"
#include "CudaDeviceCount.cuh"
#include "CudaDeviceStats.h"
#include "CudaMemoryStats.cuh"

#include "CudaMaxFilter.cuh"


void clearDevice()
{
	cudaDeviceReset();
}

 int deviceCount()
 {
	 return cDeviceCount();
 }

 int deviceStats(DevStats** stats)
 {
	 return cDeviceStats(stats);
 }

 int memoryStats(size_t** stats)
 {
	 return cMemoryStats(stats);
 }


 void maxFilter(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, Vec<size_t> kernelDims, float* kernel /*= NULL*/, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernelDims, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<char> imageIn, ImageContainer<char>& imageOut, Vec<size_t> kernelDims, float* kernel /*= NULL*/, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernelDims, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<unsigned char> imageIn, ImageContainer<unsigned char>& imageOut, Vec<size_t> kernelDims, float* kernel /*= NULL*/, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernelDims, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<short> imageIn, ImageContainer<short>& imageOut, Vec<size_t> kernelDims, float* kernel /*= NULL*/, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernelDims, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<unsigned short> imageIn, ImageContainer<unsigned short>& imageOut, Vec<size_t> kernelDims, float* kernel /*= NULL*/, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernelDims, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<int> imageIn, ImageContainer<int>& imageOut, Vec<size_t> kernelDims, float* kernel /*= NULL*/, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernelDims, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<unsigned int> imageIn, ImageContainer<unsigned int>& imageOut, Vec<size_t> kernelDims, float* kernel /*= NULL*/, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernelDims, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<float> imageIn, ImageContainer<float>& imageOut, Vec<size_t> kernelDims, float* kernel /*= NULL*/, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernelDims, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<double> imageIn, ImageContainer<double>& imageOut, Vec<size_t> kernelDims, float* kernel /*= NULL*/, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernelDims, kernel, numIterations, device);
 }
