#include "Process.h"
#include "CudaImageBuffer.cuh"
#include "CHelpers.h"

void addConstant(const MexImagePixelType* image,  MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double additive)
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.addConstant(additive);

	cudaBuffer.retrieveImage(imageOut);
}

void addImageWith(const MexImagePixelType* image1, const MexImagePixelType* image2, MexImagePixelType* imageOut,
								Vec<unsigned int> imageDims, double factor)
{
	CudaImageBuffer<unsigned char> cudaBuffer1(imageDims,true);
	CudaImageBuffer<unsigned char> cudaBuffer2(imageDims,true);
	cudaBuffer1.loadImage(image1,imageDims);
	cudaBuffer2.loadImage(image2,imageDims);

	cudaBuffer1.addImageWith(&cudaBuffer2,factor);

	cudaBuffer1.retrieveImage(imageOut);
}

void applyPolyTransformation( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double a, double b, double c, MexImagePixelType minValue, MexImagePixelType maxValue )
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.applyPolyTransformation(a,b,c,minValue,maxValue);

	cudaBuffer.retrieveImage(imageOut);
}

void calculateMinMax(const MexImagePixelType* image, Vec<unsigned int> imageDims, double& minValue, double& maxValue)
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.calculateMinMax(minValue,maxValue);
}

void contrastEnhancement(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<float> sigmas,
									   Vec<unsigned int> medianNeighborhood)
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.contrastEnhancement(sigmas,medianNeighborhood);

	cudaBuffer.retrieveImage(imageOut);
}

void gaussianFilter( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<float> sigmas )
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.gaussianFilter(sigmas);

	cudaBuffer.retrieveImage(imageOut);
}

size_t getGlobalMemoryAvailable()
{
	CudaImageBuffer<unsigned char> cudaBuffer(1);
	return cudaBuffer.getGlobalMemoryAvailable();
}

void maxFilter( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood,
			   double* kernel/*=NULL*/)
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.maxFilter(neighborhood,kernel,true);

	cudaBuffer.retrieveImage(imageOut);
}

void maximumIntensityProjection( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims )
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.maximumIntensityProjection();

	cudaBuffer.retrieveImage(imageOut);
}

void meanFilter( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood )
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.meanFilter(neighborhood);

	cudaBuffer.retrieveImage(imageOut);
}

void medianFilter( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood )
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.medianFilter(neighborhood);

	cudaBuffer.retrieveImage(imageOut);
}

void minFilter( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood,
			   double* kernel/*=NULL*/)
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.minFilter(neighborhood,kernel,true);

	cudaBuffer.retrieveImage(imageOut);
}

void multiplyImage( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double factor )
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.multiplyImage(factor);

	cudaBuffer.retrieveImage(imageOut);
}

void multiplyImageWith( const MexImagePixelType* image1, const MexImagePixelType* image2, MexImagePixelType* imageOut, Vec<unsigned int> imageDims )
{
	CudaImageBuffer<unsigned char> cudaBuffer1(imageDims,true);
	CudaImageBuffer<unsigned char> cudaBuffer2(imageDims,true);
	cudaBuffer1.loadImage(image1,imageDims);
	cudaBuffer2.loadImage(image2,imageDims);

	cudaBuffer1.multiplyImageWith(&cudaBuffer2);

	cudaBuffer1.retrieveImage(imageOut);
}

double normalizedCovariance(const MexImagePixelType* image1, const MexImagePixelType* image2, Vec<unsigned int> imageDims)
{
	CudaImageBuffer<unsigned char> cudaBuffer1(imageDims,true);
	CudaImageBuffer<unsigned char> cudaBuffer2(imageDims,true);
	cudaBuffer1.loadImage(image1,imageDims);
	cudaBuffer2.loadImage(image2,imageDims);

	return cudaBuffer1.normalizedCovariance(&cudaBuffer2);
}

void otsuThresholdFilter(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double alpha)
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.otsuThresholdFilter((float)alpha);

	cudaBuffer.retrieveImage(imageOut);
}

MexImagePixelType otsuThesholdValue(const MexImagePixelType* image, Vec<unsigned int> imageDims)
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	return cudaBuffer.otsuThresholdValue();
}

void imagePow( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, int p )
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.imagePow(p);

	cudaBuffer.retrieveImage(imageOut);
}

double sumArray(const MexImagePixelType* image, Vec<unsigned int> imageDims)
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	double sum;
	cudaBuffer.sumArray(sum);
	return sum;
}

MexImagePixelType* reduceImage( const MexImagePixelType* image, Vec<unsigned int>& imageDims, Vec<double> reductions )
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.reduceImage(reductions);

	imageDims = cudaBuffer.getDimension();
	return cudaBuffer.retrieveImage();
}

unsigned int* retrieveHistogram(const MexImagePixelType* image, Vec<unsigned int>& imageDims, int& returnSize)
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	return cudaBuffer.retrieveHistogram(returnSize);
}

double* retrieveNormalizedHistogram(const MexImagePixelType* image, Vec<unsigned int>& imageDims, int& returnSize)
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	return cudaBuffer.retrieveNormalizedHistogram(returnSize);
}

void thresholdFilter( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double threshold )
{
	CudaImageBuffer<unsigned char> cudaBuffer(imageDims,true);
	cudaBuffer.loadImage(image,imageDims);

	cudaBuffer.thresholdFilter(threshold);

	cudaBuffer.retrieveImage(imageOut);
}

void unmix( const MexImagePixelType* image1, const MexImagePixelType* image2, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood )
{
	CudaImageBuffer<unsigned char> cudaBuffer1(imageDims,true);
	CudaImageBuffer<unsigned char> cudaBuffer2(imageDims,true);
	cudaBuffer1.loadImage(image1,imageDims);
	cudaBuffer2.loadImage(image2,imageDims);

	cudaBuffer1.unmix(&cudaBuffer2,neighborhood);

	cudaBuffer1.retrieveImage(imageOut);
}