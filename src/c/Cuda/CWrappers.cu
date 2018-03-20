#include "CWrappers.h"
#include "CudaDeviceCount.cuh"
#include "CudaDeviceStats.h"
#include "CudaMemoryStats.cuh"

#include "CudaGaussian.cuh"
#include "CudaMaxFilter.cuh"
#include "CudaMinFilter.cuh"
#include "CudaMultiplySum.cuh"


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

 /// Example wrapper code
 //void fooFilter(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 //{
	// cFooFilter(imageIn, imageOut, kernel, numIterations, device);
 //}


 //void fooFilter(const ImageContainer<char> imageIn, ImageContainer<char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 //{
	// cFooFilter(imageIn, imageOut, kernel, numIterations, device);
 //}


 //void fooFilter(const ImageContainer<unsigned char> imageIn, ImageContainer<unsigned char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 //{
	// cFooFilter(imageIn, imageOut, kernel, numIterations, device);
 //}


 //void fooFilter(const ImageContainer<short> imageIn, ImageContainer<short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 //{
	// cFooFilter(imageIn, imageOut, kernel, numIterations, device);
 //}


 //void fooFilter(const ImageContainer<unsigned short> imageIn, ImageContainer<unsigned short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 //{
	// cFooFilter(imageIn, imageOut, kernel, numIterations, device);
 //}


 //void fooFilter(const ImageContainer<int> imageIn, ImageContainer<int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 //{
	// cFooFilter(imageIn, imageOut, kernel, numIterations, device);
 //}


 //void fooFilter(const ImageContainer<unsigned int> imageIn, ImageContainer<unsigned int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 //{
	// cFooFilter(imageIn, imageOut, kernel, numIterations, device);
 //}


 //void fooFilter(const ImageContainer<float> imageIn, ImageContainer<float>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 //{
	// cFooFilter(imageIn, imageOut, kernel, numIterations, device);
 //}


 //void fooFilter(const ImageContainer<double> imageIn, ImageContainer<double>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 //{
	// cFooFilter(imageIn, imageOut, kernel, numIterations, device);
 //}



 void maxFilter(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<char> imageIn, ImageContainer<char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<unsigned char> imageIn, ImageContainer<unsigned char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<short> imageIn, ImageContainer<short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<unsigned short> imageIn, ImageContainer<unsigned short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<int> imageIn, ImageContainer<int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<unsigned int> imageIn, ImageContainer<unsigned int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<float> imageIn, ImageContainer<float>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void maxFilter(const ImageContainer<double> imageIn, ImageContainer<double>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 void gaussian(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
	 cGaussian(imageIn, imageOut, sigmas, numIterations, device);
 }
 
 void gaussian(const ImageContainer<char> imageIn, ImageContainer<char>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cGaussian(imageIn, imageOut, sigmas, numIterations, device);
 }
 
 void gaussian(const ImageContainer<unsigned char> imageIn, ImageContainer<unsigned char>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cGaussian(imageIn, imageOut, sigmas, numIterations, device);
 }
 
 void gaussian(const ImageContainer<short> imageIn, ImageContainer<short>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cGaussian(imageIn, imageOut, sigmas, numIterations, device);
 }
 
 void gaussian(const ImageContainer<unsigned short> imageIn, ImageContainer<unsigned short>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cGaussian(imageIn, imageOut, sigmas, numIterations, device);
 }
 
 void gaussian(const ImageContainer<int> imageIn, ImageContainer<int>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cGaussian(imageIn, imageOut, sigmas, numIterations, device);
 }
 
 void gaussian(const ImageContainer<unsigned int> imageIn, ImageContainer<unsigned int>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cGaussian(imageIn, imageOut, sigmas, numIterations, device);
 }

 void gaussian(const ImageContainer<float> imageIn, ImageContainer<float>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cGaussian(imageIn, imageOut, sigmas, numIterations, device);
 }

 void gaussian(const ImageContainer<double> imageIn, ImageContainer<double>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cGaussian(imageIn, imageOut, sigmas, numIterations, device);
 }


 void maxFilter(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void maxFilter(const ImageContainer<char> imageIn, ImageContainer<char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void maxFilter(const ImageContainer<unsigned char> imageIn, ImageContainer<unsigned char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void maxFilter(const ImageContainer<short> imageIn, ImageContainer<short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void maxFilter(const ImageContainer<unsigned short> imageIn, ImageContainer<unsigned short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void maxFilter(const ImageContainer<int> imageIn, ImageContainer<int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void maxFilter(const ImageContainer<unsigned int> imageIn, ImageContainer<unsigned int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void maxFilter(const ImageContainer<float> imageIn, ImageContainer<float>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void maxFilter(const ImageContainer<double> imageIn, ImageContainer<double>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMaxFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void minFilter(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMinFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void minFilter(const ImageContainer<char> imageIn, ImageContainer<char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMinFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void minFilter(const ImageContainer<unsigned char> imageIn, ImageContainer<unsigned char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMinFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void minFilter(const ImageContainer<short> imageIn, ImageContainer<short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMinFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void minFilter(const ImageContainer<unsigned short> imageIn, ImageContainer<unsigned short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMinFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void minFilter(const ImageContainer<int> imageIn, ImageContainer<int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMinFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void minFilter(const ImageContainer<unsigned int> imageIn, ImageContainer<unsigned int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMinFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void minFilter(const ImageContainer<float> imageIn, ImageContainer<float>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMinFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void minFilter(const ImageContainer<double> imageIn, ImageContainer<double>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMinFilter(imageIn, imageOut, kernel, numIterations, device);
 }



 IMAGE_PROCESSOR_API void multiplySum(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }


 IMAGE_PROCESSOR_API void multiplySum(const ImageContainer<char> imageIn, ImageContainer<char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }


 IMAGE_PROCESSOR_API void multiplySum(const ImageContainer<unsigned char> imageIn, ImageContainer<unsigned char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }


 IMAGE_PROCESSOR_API void multiplySum(const ImageContainer<short> imageIn, ImageContainer<short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }


 IMAGE_PROCESSOR_API void multiplySum(const ImageContainer<unsigned short> imageIn, ImageContainer<unsigned short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }


 IMAGE_PROCESSOR_API void multiplySum(const ImageContainer<int> imageIn, ImageContainer<int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }


 IMAGE_PROCESSOR_API void multiplySum(const ImageContainer<unsigned int> imageIn, ImageContainer<unsigned int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }


 IMAGE_PROCESSOR_API void multiplySum(const ImageContainer<float> imageIn, ImageContainer<float>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }


 IMAGE_PROCESSOR_API void multiplySum(const ImageContainer<double> imageIn, ImageContainer<double>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }

