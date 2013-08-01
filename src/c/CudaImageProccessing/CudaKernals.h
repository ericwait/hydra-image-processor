#ifndef CUDA_KERNALS_H
#define CUDA_KERNALS_H

#include "Vec.h"

template<typename ImagePixelType>
__global__ void meanFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims, int kernalDiameter);

template<typename ImagePixelType1, typename ImagePixelType2, typename FactorType>
__global__ void multiplyImage(ImagePixelType1* imageIn, ImagePixelType2* imageOut, Vec<int> imageDims, FactorType factor);

template<typename ImagePixelType1, typename ImagePixelType2, typename FactorType>
__global__ void normalizeImage(ImagePixelType1* imageIn, ImagePixelType2* imageOut, Vec<int> imageDims, FactorType factor);

template<typename ImagePixelType1, typename ImagePixelType2, typename ImagePixelType3, typename FactorType>
__global__ void addTwoImagesWithFactor(ImagePixelType1* imageIn1, ImagePixelType2* imageIn2, ImagePixelType3* imageOut,
	Vec<int> imageDims, FactorType factor);

template<typename ImagePixelType>
__device__ ImagePixelType findMedian(ImagePixelType* vals, int numVals);

template<typename ImagePixelType>
__global__ void medianFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims, Vec<int> kernalDims);

template<typename ImagePixelType1, typename ImagePixelType2, typename KernalType>
__global__ void multAddFilter(ImagePixelType1* imageIn, ImagePixelType2* imageOut, Vec<int> imageDims, KernalType* kernal,
	Vec<int> kernalDims);

template<typename ImagePixelType, typename KernalType>
__global__ void minFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims, KernalType* kernal,
	Vec<int> kernalDims);

template<typename ImagePixelType, typename KernalType>
__global__ void maxFilter(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims, KernalType* kernal,
	Vec<int> kernalDims);

template<typename ImagePixelType>
__global__ void histogramCreate(ImagePixelType* imageIn, unsigned int* histogram, Vec<int> imageDims);

__global__ void normalizeHistogram(unsigned int* histogram, double* normHistogram, Vec<int> imageDims);

template<typename ImagePixelType, typename ThresholdType>
__global__ void thresholdImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims, ThresholdType threshold);

template<typename ImagePixelType>
__global__ void findMinMax(ImagePixelType* arrayIn, ImagePixelType* minArrayOut, ImagePixelType* maxArrayOut, unsigned int n);

template<typename ImagePixelType, typename ThresholdType>
__global__ void polyTransferFuncImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageDims, ThresholdType a,
	ThresholdType b, ThresholdType c, ThresholdType maxPixelValue);

template<typename ImagePixelType, typename ThresholdType>
__global__ void getCoordinates(ImagePixelType* imageIn, Vec<int>* coordinatesOut, Vec<int> imageDims, ThresholdType threshold);

__global__ void fillCoordinates(Vec<int>* coordinatesIn, Vec<int>* coordinatesOut, Vec<int> imageDims, int overDimension);

template<typename T>
__global__ void reduceArray(T* arrayIn, T* arrayOut, unsigned int n);

template<typename ImagePixelType>
__global__ void ruduceImage(ImagePixelType* imageIn, ImagePixelType* imageOut, Vec<int> imageInDims, unsigned int reductionAmount);
#endif