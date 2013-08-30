#pragma once

#include "defines.h"
#include "Vec.h"
#include "CudaKernals.h"
#include "CudaUtilities.h"

template<typename ImagePixelType>
class CudaImageBuffer
{
public:
	//////////////////////////////////////////////////////////////////////////
	// Constructor / Destructor
	//////////////////////////////////////////////////////////////////////////
	CudaImageBuffer(Vec<int> dims, int device=0);
	CudaImageBuffer(int x, int y, int z, int device=0);
	CudaImageBuffer(int n, int device=0);
	~CudaImageBuffer();
	// End Constructor / Destructor

	//////////////////////////////////////////////////////////////////////////
	// Copy Constructors
	//////////////////////////////////////////////////////////////////////////
	CudaImageBuffer (const CudaImageBuffer<ImagePixelType>& bufferIn){copy(bufferIn);}
	CudaImageBuffer& operator=(const CudaImageBuffer<ImagePixelType>& bufferIn);
	// End Copy Constructors

	//////////////////////////////////////////////////////////////////////////
	// Setters / Getters
	//////////////////////////////////////////////////////////////////////////
	void loadImage(ImagePixelType* image);
	void retrieveImage(ImagePixelType& imageOut);

	/*
	*	Returns a host pointer to the histogram data
	*	This is destroyed when this' destructor is called
	*	Will call the needed histogram creation methods if not all ready
	*/
	unsigned int* retrieveHistogram(int& returnSize);

	/*
	*	Returns a host pointer to the normalized histogram data
	*	This is destroyed when this' destructor is called
	*	Will call the needed histogram creation methods if not all ready
	*/
	double* retrieveNormalizedHistogram(int& returnSize);
	ImagePixelType* retrieveReducedImage(Vec<int>& reducedDims);
	Vec<int> getDimension() const {return imageDims;}
	int getDevice() const {return device;}
	void getROI(Vec<int> startPos, Vec<int> newSize);
	// End Setters / Getters

	//////////////////////////////////////////////////////////////////////////
	// Cuda Operators (Alphabetical order)
	//////////////////////////////////////////////////////////////////////////

	/*
	*	Adds this image to the passed in one.  You can apply a factor
	*	which is multiplied to the passed in image prior to adding
	*/
	void addImageTo(const CudaImageBuffer* image, double factor);

	/*
	*	New pixel values will be a*x^2 + b*x + c where x is the original
	*	pixel value.  This new value will be clamped between the min and
	*	max values.
	*/
	template<typename ThresholdType>
	void applyPolyTransformation(ThresholdType a, ThresholdType b, ThresholdType c, ImagePixelType minValue, ImagePixelType maxValue);

	/*
	*	This will find the min and max values of the image
	*/ 
	void calculateMinMax(ImagePixelType& minValue, ImagePixelType& maxValue);

	/*
	*	Creates Histogram on the card using the #define NUM_BINS
	*	Use retrieveHistogram to get the results off the card
	*/
	void createHistogram();

	/*
	*	Will smooth the image using the given sigmas for each dimension
	*/ 
	void gaussianFilter(Vec<double> sigmas);

	/*
	*	Sets each pixel to the max value of its neighborhood
	*	Dilates structures
	*/ 
	void maxFilter(Vec<int> neighborhood);

	/*
	*	produce an image that is the maximum value in z for each (x,y)
	*/
	void maximumIntensityProjection();

	/*
	*	Filters image where each pixel is the mean of its neighborhood 
	*/
	void meanFilter(Vec<int> neighborhood);

	/*
	*	Filters image where each pixel is the median of its neighborhood
	*/
	void medianFilter(Vec<int> neighborhood);

	/*
	*	Sets each pixel to the min value of its neighborhood
	*	Erodes structures
	*/ 
	void minFilter(Vec<int> neighborhood);

	/*
	*	Sets each pixel by multiplying by the orignal value and clamping
	*	between minValue and maxValue
	*/
	template<typename FactorType>
	void multiplyImage(FactorType factor, ImagePixelType minValue, ImagePixelType maxValue);

	/*
	*	Takes a histogram that is on the card and normalizes it
	*	Will generate the original histogram if one doesn't already exist
	*	Use retrieveNormalizedHistogram() to get a host pointer
	*/
	void normalizeHistogram();

	/*
	*	Calculates the total sum of the buffer's data
	*/
	template<typename Sumtype>
	void reduceArray(Sumtype& sum);

	/*
	*	Will reduce the size of the image by the factors passed in
	*/
	void reduceImage(Vec<double> reductions);

	/*
	*	This creates a image with values of 0 where the pixels fall below
	*	the threshold and 1 where equal or greater than the threshold
	*	
	*	If you want a viewable image after this, you may want to use the
	*	multiplyImage routine to turn the 1 values to the max values of
	*	the type
	*/
	template<typename ThresholdType>
	void thresholdFilter(ThresholdType threshold);
	// End Cuda Operators

private:
	CudaImageBuffer();

	void deviceSetup();
	void MemoryAllocation();
	void copy(const CudaImageBuffer<ImagePixelType>& bufferIn);
	void defaults();
	void clean();
	ImagePixelType* getCurrentBuffer() const;
	ImagePixelType* getNextBuffer();
	void incrementBufferNumber();

	Vec<int> imageDims;
	Vec<int> reducedDims;
	int device;
	cudaDeviceProp deviceProp;
	dim3 blocks, threads;
	char currentBuffer;
	ImagePixelType* imageBuffers[NUM_BUFFERS];
	ImagePixelType* reducedImageHost;
	ImagePixelType* reducedImageDevice;
	unsigned int* histogramHost;
	unsigned int* histogramDevice;
	double* normalizedHistogramHost;
	double* normalizedHistogramDevice;
};
