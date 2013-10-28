#include "Process.h"
#include "CudaProcessBuffer.cuh"
#include "CHelpers.h"

CudaProcessBuffer<MexImagePixelType>* g_cudaBuffer = NULL;
CudaProcessBuffer<MexImagePixelType>* g_cudaBuffer2 = NULL;

void clear()
{
	if (g_cudaBuffer!=NULL)
		delete g_cudaBuffer;
	if (g_cudaBuffer2!=NULL)
		delete g_cudaBuffer2;
}

void set(Vec<unsigned int> imageDims)
{
	if (g_cudaBuffer==NULL)
		g_cudaBuffer = new CudaProcessBuffer<unsigned char>(imageDims,true);
}

void set2(Vec<unsigned int> imageDims)
{
	if (g_cudaBuffer2==NULL)
		g_cudaBuffer2 = new CudaProcessBuffer<unsigned char>(imageDims,true);
}

void addConstant(const MexImagePixelType* image,  MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double additive)
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->addConstant(additive);

	g_cudaBuffer->retrieveImage(imageOut);
}

void addImageWith(const MexImagePixelType* image1, const MexImagePixelType* image2, MexImagePixelType* imageOut,
								Vec<unsigned int> imageDims, double factor)
{
	set(imageDims);
	set2(imageDims);
	g_cudaBuffer->loadImage(image1,imageDims);
	g_cudaBuffer2->loadImage(image2,imageDims);

	g_cudaBuffer->addImageWith(g_cudaBuffer2,factor);

	g_cudaBuffer->retrieveImage(imageOut);
}

void applyPolyTransformation( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double a, double b, double c, MexImagePixelType minValue, MexImagePixelType maxValue )
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->applyPolyTransformation(a,b,c,minValue,maxValue);

	g_cudaBuffer->retrieveImage(imageOut);
}

void calculateMinMax(const MexImagePixelType* image, Vec<unsigned int> imageDims, double& minValue, double& maxValue)
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->calculateMinMax(minValue,maxValue);
}

void contrastEnhancement(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<float> sigmas,
									   Vec<unsigned int> medianNeighborhood)
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->contrastEnhancement(sigmas,medianNeighborhood);

	g_cudaBuffer->retrieveImage(imageOut);
}

void gaussianFilter( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<float> sigmas )
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->gaussianFilter(sigmas);

	g_cudaBuffer->retrieveImage(imageOut);
}

size_t getGlobalMemoryAvailable()
{
	CudaProcessBuffer<unsigned char> cudaBuffer(1);
	return g_cudaBuffer->getGlobalMemoryAvailable();
}

void mask(const MexImagePixelType* image1, const MexImagePixelType* image2, MexImagePixelType* imageOut, Vec<unsigned int> imageDims,
		  double threshold)
{
	set(imageDims);
	set2(imageDims);
	g_cudaBuffer->loadImage(image1,imageDims);
	g_cudaBuffer2->loadImage(image2,imageDims);

	g_cudaBuffer->mask(g_cudaBuffer2,threshold);

	g_cudaBuffer->retrieveImage(imageOut);
}

void maxFilter( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood,
			   double* kernel/*=NULL*/)
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->maxFilter(neighborhood,kernel,true);

	g_cudaBuffer->retrieveImage(imageOut);
}

void maximumIntensityProjection( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims )
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->maximumIntensityProjection();

	g_cudaBuffer->retrieveImage(imageOut);
}

void meanFilter( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood )
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->meanFilter(neighborhood);

	g_cudaBuffer->retrieveImage(imageOut);
}

void medianFilter( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood )
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->medianFilter(neighborhood);

	g_cudaBuffer->retrieveImage(imageOut);
}

void minFilter( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood,
			   double* kernel/*=NULL*/)
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->minFilter(neighborhood,kernel,true);

	g_cudaBuffer->retrieveImage(imageOut);
}

void morphClosure( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood,
			   double* kernel/*=NULL*/)
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->morphClosure(neighborhood,kernel,true);

	g_cudaBuffer->retrieveImage(imageOut);
}

void morphOpening( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood,
				  double* kernel/*=NULL*/)
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->morphOpening(neighborhood,kernel,true);

	g_cudaBuffer->retrieveImage(imageOut);
}

void multiplyImage( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double factor )
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->multiplyImage(factor);

	g_cudaBuffer->retrieveImage(imageOut);
}

void multiplyImageWith( const MexImagePixelType* image1, const MexImagePixelType* image2, MexImagePixelType* imageOut, Vec<unsigned int> imageDims )
{
	set(imageDims);
	set2(imageDims);
	g_cudaBuffer->loadImage(image1,imageDims);
	g_cudaBuffer2->loadImage(image2,imageDims);

	g_cudaBuffer->multiplyImageWith(g_cudaBuffer2);

	g_cudaBuffer->retrieveImage(imageOut);
}

double normalizedCovariance(const MexImagePixelType* image1, const MexImagePixelType* image2, Vec<unsigned int> imageDims)
{
	set(imageDims);
	set2(imageDims);
	g_cudaBuffer->loadImage(image1,imageDims);
	g_cudaBuffer2->loadImage(image2,imageDims);

	return g_cudaBuffer->normalizedCovariance(g_cudaBuffer2);
}

void otsuThresholdFilter(const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double alpha)
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->otsuThresholdFilter((float)alpha);

	g_cudaBuffer->retrieveImage(imageOut);
}

MexImagePixelType otsuThesholdValue(const MexImagePixelType* image, Vec<unsigned int> imageDims)
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	return g_cudaBuffer->otsuThresholdValue();
}

void imagePow( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, int p )
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->imagePow(p);

	g_cudaBuffer->retrieveImage(imageOut);
}

double sumArray(const MexImagePixelType* image, Vec<unsigned int> imageDims)
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	double sum;
	g_cudaBuffer->sumArray(sum);
	return sum;
}

MexImagePixelType* reduceImage( const MexImagePixelType* image, Vec<unsigned int>& imageDims, Vec<double> reductions )
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->reduceImage(reductions);

	imageDims = g_cudaBuffer->getDimension();
	return g_cudaBuffer->retrieveImage();
}

unsigned int* retrieveHistogram(const MexImagePixelType* image, Vec<unsigned int>& imageDims, int& returnSize)
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	return g_cudaBuffer->retrieveHistogram(returnSize);
}

double* retrieveNormalizedHistogram(const MexImagePixelType* image, Vec<unsigned int>& imageDims, int& returnSize)
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	return g_cudaBuffer->retrieveNormalizedHistogram(returnSize);
}

void thresholdFilter( const MexImagePixelType* image, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, double threshold )
{
	set(imageDims);
	g_cudaBuffer->loadImage(image,imageDims);

	g_cudaBuffer->thresholdFilter(threshold);

	g_cudaBuffer->retrieveImage(imageOut);
}

void unmix( const MexImagePixelType* image1, const MexImagePixelType* image2, MexImagePixelType* imageOut, Vec<unsigned int> imageDims, Vec<unsigned int> neighborhood )
{
	set(imageDims);
	set2(imageDims);
	g_cudaBuffer->loadImage(image1,imageDims);
	g_cudaBuffer2->loadImage(image2,imageDims);

	g_cudaBuffer->unmix(g_cudaBuffer2,neighborhood);

	g_cudaBuffer->retrieveImage(imageOut);
}