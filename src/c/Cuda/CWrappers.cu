#include "CWrappers.h"
#include "CudaDeviceCount.cuh"
#include "CudaDeviceStats.h"
#include "CudaMemoryStats.cuh"

#include "CudaClosure.cuh"
#include "CudaGaussian.cuh"
#include "CudaLoG.cuh"
#include "CudaMaxFilter.cuh"
#include "CudaMedianFilter.cuh"
#include "CudaMeanFilter.cuh"
#include "CudaMinFilter.cuh"
#include "CudaMultiplySum.cuh"
#include "CudaOpener.cuh"


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

 void closure(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cClosure(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void closure(const ImageContainer<char> imageIn, ImageContainer<char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cClosure(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void closure(const ImageContainer<unsigned char> imageIn, ImageContainer<unsigned char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cClosure(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void closure(const ImageContainer<short> imageIn, ImageContainer<short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cClosure(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void closure(const ImageContainer<unsigned short> imageIn, ImageContainer<unsigned short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cClosure(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void closure(const ImageContainer<int> imageIn, ImageContainer<int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cClosure(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void closure(const ImageContainer<unsigned int> imageIn, ImageContainer<unsigned int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cClosure(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void closure(const ImageContainer<float> imageIn, ImageContainer<float>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cClosure(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void closure(const ImageContainer<double> imageIn, ImageContainer<double>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cClosure(imageIn, imageOut, kernel, numIterations, device);
 }


 void gaussian(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
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


 void LoG(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cLoG(imageIn, imageOut, sigmas, numIterations, device);
 }

 void LoG(const ImageContainer<char> imageIn, ImageContainer<char>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cLoG(imageIn, imageOut, sigmas, numIterations, device);
 }

 void LoG(const ImageContainer<unsigned char> imageIn, ImageContainer<unsigned char>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cLoG(imageIn, imageOut, sigmas, numIterations, device);
 }

 void LoG(const ImageContainer<short> imageIn, ImageContainer<short>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cLoG(imageIn, imageOut, sigmas, numIterations, device);
 }

 void LoG(const ImageContainer<unsigned short> imageIn, ImageContainer<unsigned short>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cLoG(imageIn, imageOut, sigmas, numIterations, device);
 }

 void LoG(const ImageContainer<int> imageIn, ImageContainer<int>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cLoG(imageIn, imageOut, sigmas, numIterations, device);
 }

 void LoG(const ImageContainer<unsigned int> imageIn, ImageContainer<unsigned int>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cLoG(imageIn, imageOut, sigmas, numIterations, device);
 }

 void LoG(const ImageContainer<float> imageIn, ImageContainer<float>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cLoG(imageIn, imageOut, sigmas, numIterations, device);
 }

 void LoG(const ImageContainer<double> imageIn, ImageContainer<double>& imageOut, Vec<double> sigmas, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cLoG(imageIn, imageOut, sigmas, numIterations, device);
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
 

 void meanFilter(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMeanFilter(imageIn, imageOut, kernel, numIterations, device);
 }

 void meanFilter(const ImageContainer<char> imageIn, ImageContainer<char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMeanFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void meanFilter(const ImageContainer<unsigned char> imageIn, ImageContainer<unsigned char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMeanFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void meanFilter(const ImageContainer<short> imageIn, ImageContainer<short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMeanFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void meanFilter(const ImageContainer<unsigned short> imageIn, ImageContainer<unsigned short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMeanFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void meanFilter(const ImageContainer<int> imageIn, ImageContainer<int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMeanFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void meanFilter(const ImageContainer<unsigned int> imageIn, ImageContainer<unsigned int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMeanFilter(imageIn, imageOut, kernel, numIterations, device);
 }

 void meanFilter(const ImageContainer<float> imageIn, ImageContainer<float>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMeanFilter(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void meanFilter(const ImageContainer<double> imageIn, ImageContainer<double>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMeanFilter(imageIn, imageOut, kernel, numIterations, device);
 }


 void medianFilter(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMedianFilter(imageIn, imageOut, kernel, numIterations, device);
 }

 void medianFilter(const ImageContainer<char> imageIn, ImageContainer<char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMedianFilter(imageIn, imageOut, kernel, numIterations, device);
 }

 void medianFilter(const ImageContainer<unsigned char> imageIn, ImageContainer<unsigned char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMedianFilter(imageIn, imageOut, kernel, numIterations, device);
 }

 void medianFilter(const ImageContainer<short> imageIn, ImageContainer<short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMedianFilter(imageIn, imageOut, kernel, numIterations, device);
 }

 void medianFilter(const ImageContainer<unsigned short> imageIn, ImageContainer<unsigned short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMedianFilter(imageIn, imageOut, kernel, numIterations, device);
 }

 void medianFilter(const ImageContainer<int> imageIn, ImageContainer<int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMedianFilter(imageIn, imageOut, kernel, numIterations, device);
 }

 void medianFilter(const ImageContainer<unsigned int> imageIn, ImageContainer<unsigned int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMedianFilter(imageIn, imageOut, kernel, numIterations, device);
 }

 void medianFilter(const ImageContainer<float> imageIn, ImageContainer<float>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMedianFilter(imageIn, imageOut, kernel, numIterations, device);
 }

 void medianFilter(const ImageContainer<double> imageIn, ImageContainer<double>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMedianFilter(imageIn, imageOut, kernel, numIterations, device);
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
 

 void multiplySum(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void multiplySum(const ImageContainer<char> imageIn, ImageContainer<char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void multiplySum(const ImageContainer<unsigned char> imageIn, ImageContainer<unsigned char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void multiplySum(const ImageContainer<short> imageIn, ImageContainer<short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void multiplySum(const ImageContainer<unsigned short> imageIn, ImageContainer<unsigned short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void multiplySum(const ImageContainer<int> imageIn, ImageContainer<int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void multiplySum(const ImageContainer<unsigned int> imageIn, ImageContainer<unsigned int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void multiplySum(const ImageContainer<float> imageIn, ImageContainer<float>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }
 
 void multiplySum(const ImageContainer<double> imageIn, ImageContainer<double>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cMultiplySum(imageIn, imageOut, kernel, numIterations, device);
 }
 

 void opener(const ImageContainer<bool> imageIn, ImageContainer<bool>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cOpener(imageIn, imageOut, kernel, numIterations, device);
 }

 void opener(const ImageContainer<char> imageIn, ImageContainer<char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cOpener(imageIn, imageOut, kernel, numIterations, device);
 }

 void opener(const ImageContainer<unsigned char> imageIn, ImageContainer<unsigned char>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cOpener(imageIn, imageOut, kernel, numIterations, device);
 }

 void opener(const ImageContainer<short> imageIn, ImageContainer<short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cOpener(imageIn, imageOut, kernel, numIterations, device);
 }

 void opener(const ImageContainer<unsigned short> imageIn, ImageContainer<unsigned short>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cOpener(imageIn, imageOut, kernel, numIterations, device);
 }

 void opener(const ImageContainer<int> imageIn, ImageContainer<int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cOpener(imageIn, imageOut, kernel, numIterations, device);
 }

 void opener(const ImageContainer<unsigned int> imageIn, ImageContainer<unsigned int>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cOpener(imageIn, imageOut, kernel, numIterations, device);
 }

 void opener(const ImageContainer<float> imageIn, ImageContainer<float>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cOpener(imageIn, imageOut, kernel, numIterations, device);
 }

 void opener(const ImageContainer<double> imageIn, ImageContainer<double>& imageOut, ImageContainer<float> kernel, int numIterations /*= 1*/, int device /*= -1*/)
 {
	 cOpener(imageIn, imageOut, kernel, numIterations, device);
 }
