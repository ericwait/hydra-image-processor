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

	CudaImageBuffer(Vec<int> dims, int device=0)
	{
		defaults();

		this->imageDims = dims;
		this->device = device;

		deviceSetup();
		MemoryAllocation();
	}

	CudaImageBuffer(int x, int y, int z, int device=0)
	{
		defaults();

		imageDims = Vec<int>(x,y,z);
		this->device = device;

		deviceSetup();
		MemoryAllocation();
	}

	CudaImageBuffer(int n, int device=0)
	{
		defaults();

		imageDims = Vec<int>(n,1,1);
		this->device = device;

		deviceSetup();
		MemoryAllocation();
	}

	~CudaImageBuffer()
	{
		clean();
	}

// End Constructor / Destructor

//////////////////////////////////////////////////////////////////////////
// Copy Constructors
//////////////////////////////////////////////////////////////////////////

	CudaImageBuffer (const CudaImageBuffer<ImagePixelType>& bufferIn){copy(bufferIn);}
	CudaImageBuffer& operator=(const CudaImageBuffer<ImagePixelType>& bufferIn)
	{
		if (this == &bufferIn)
			return *this;

		clean();
		copy(bufferIn);
	}

// End Copy Constructors

//////////////////////////////////////////////////////////////////////////
// Setters / Getters
//////////////////////////////////////////////////////////////////////////

	void loadImage(ImagePixelType* image)
	{
		incrementBufferNumber();
		HANDLE_ERROR(cudaMemcpy(imageBuffers[currentBuffer],image,sizeof(ImagePixelType)*imageDims.product(),cudaMemcpyHostToDevice));
	}

	void retrieveImage(ImagePixelType& imageOut)
	{
		if (currentBuffer<0 || currentBuffer>NUM_BUFFERS)
		{
			imageOut = NULL;
			return;
		}

		HANDLE_ERROR(cudaMemcpy(image,imageBuffers[currentBuffer],sizeof(ImagePixelType)*imageDims.product(),cudaMemcpyDeviceToHost));
	}

	Vec<int> getDimension() const {return imageDims;}
	int getDevice() const {return device;}

// End Setters / Getters

//////////////////////////////////////////////////////////////////////////
// Cuda Operators (Alphabetical order)
//////////////////////////////////////////////////////////////////////////

	/*
	 *	Adds this image to the passed in one.  You can apply a factor
	 *	which is multiplied to the passed in image prior to adding
	 */
	void addImageTo(const CudaImageBuffer* image, double factor)
	{
		addTwoImagesWithFactor<<<blocks,threads>>>(getCurrentBuffer(),image->getCurrentBuffer(),getNextBuffer(),imageDims,factor);
	}

	/*
	 *	New pixel values will be a*x^2 + b*x + c where x is the original
	 *	pixel value.  This new value will be clamped between the min and
	 *	max values.
	 */
	template<typename ThresholdType>
	void applyPolyTransformation(ThresholdType a, ThresholdType b, ThresholdType c, ImagePixelType minValue, ImagePixelType maxValue)
	{
		polyTransferFuncImage<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,a,b,c,maxValue,minValue);
	}

	/*
	 *	This will find the min and max values of the image
	 */ 
	void calculateMinMax(ImagePixelType& minValue, ImagePixelType& maxValue)
	{
		double* maxValuesHost = new double[(blocks.x+1)/2];
		double* minValuesHost = new double[(blocks.x+1)/2];
		double* maxValuesDevice;
		double* minValuesDevice;
		
		HANDLE_ERROR(cudaMalloc((void**)&maxValuesDevice,sizeof(double)*(blocks.x+1)/2));
		HANDLE_ERROR(cudaMalloc((void**)&minValuesDevice,sizeof(double)*(blocks.x+1)/2));

		findMinMax<<<blocks.x,threads.x,2*sizeof(double)*threads.x>>>(getNextBuffer(),minValuesDevice,maxValuesDevice,
			imageDims.product());

		HANDLE_ERROR(cudaMemcpy(maxValuesHost,maxValuesDevice,sizeof(double)*(blocks.x)/2,cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(minValuesHost,minValuesDevice,sizeof(double)*(blocks.x)/2,cudaMemcpyDeviceToHost));

		maxValue = maxValuesHost[0];
		minValue = minValuesHost[0];

		for (int i=0; i<blocks.x/2; ++i)
		{
			if (maxValue < maxValuesHost[i])
				maxValue = maxValuesHost[i];

			if (minValue > minValuesHost[i])
				minValue = minValuesHost[i];
		}
	}

	/*
	*	Creates Histogram on the card using the #define NUM_BINS
	*	Use retrieveHistogram to get the results off the card
	*/
	void createHistogram()
	{
	}

	/*
	 *	Will smooth the image using the given sigmas for each dimension
	 */ 
	void gaussianFilter(Vec<double> sigmas)
	{
	}

	/*
	 *	Sets each pixel to the max value of its neighborhood
	 */ 
	void maxFilter(int neighborhood)
	{
	}

	/*
	 *	produce an image that is the maximum value in z for each (x,y)
	 */
	void maximumIntensityProjection()
	{
	}

	/*
	 *	Filters image where each pixel is the mean of its neighborhood 
	 */
	void meanFilter(int neighborhood)
	{
	}

	/*
	 *	Filters image where each pixel is the median of its neighborhood
	 */
	void medianFilter();

	/*
	 *	Sets each pixel to the min value of its neighborhood
	 */ 
	void minFilter();

	/*
	 *	Sets each pixel by multiplying by the orignal value and clamping
	 *	between minValue and maxValue
	 */
	template<typename FactorType>
	void multiplyImage(FactorType factor, ImagePixelType minValue, ImagePixelType maxValue)
	{
	}

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
	void reduceArray(Sumtype& sum)
	{
	}

	/*
	 *	Will reduce the size of the image by the factors passed in
	 */
	void reduceImage(Vec<int> reductions)
	{

	}
	
	/*
	 *	Returns a host pointer to the histogram data
	 *	This is destroyed when this' destructor is called
	 */
	void retrieveHistogram()
	{

	}
	
	/*
	 *	Returns a host pointer to the normalized histogram data
	 *	This is destroyed when this' destructor is called
	 */
	void retrieveNormalizedHistogram();

	/*
	 *	This creates a image with values of 0 where the pixels fall below
	 *	the threshold and 1 where equal or greater than the threshold
	 *	
	 *	If you want a viewable image after this, you may want to use the
	 *	multiplyImage routine to turn the 1 values to the max values of
	 *	the type
	 */
	void thresholdFilter()
	{

	}

// End Cuda Operators


private:
	CudaImageBuffer();

	void deviceSetup() 
	{
		HANDLE_ERROR(cudaSetDevice(device));
		HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp,device));
		calcBlockThread(imageDims,deviceProp,blocks,threads);
	}

	void MemoryAllocation()
	{
		for (int i=0; i<NUM_BUFFERS; ++i)
		{
			HANDLE_ERROR(cudaMalloc((void**)&imageBuffers[i],sizeof(ImagePixelType)*imageDims.product()));
		}

		currentBuffer = -1;
	}

	void copy(const CudaImageBuffer<ImagePixelType>& bufferIn)
	{
		imageDims = bufferIn.getDimension();
		device = bufferIn.getDevice();

		defaults();
		MemoryAllocation();

		currentBuffer = 0;
		ImagePixelType* inImage = bufferIn.getCurrentBuffer();

		if (inImage!=NULL)
			HANDLE_ERROR(cudaMemcpy(imageBuffers[currentBuffer],inImage,sizeof(ImagePixelType)*imageDims.product(),cudaMemcpyHostToHost));

		if (bufferIn.histogramHost!=NULL)
			memcpy(histogramHost,bufferIn.histogramHost,sizeof(unsigned int)*imageDims.product());

		if (bufferIn.histogramDevice!=NULL)
			HANDLE_ERROR(cudaMemcpy(histogramDevice,bufferIn.histogramDevice,sizeof(unsigned int)*NUM_BINS,cudaMemcpyHostToHost));

		if (bufferIn.normalizedHistogramHost!=NULL)
			memcpy(normalizedHistogramHost,bufferIn.normalizedHistogramHost,sizeof(double)*imageDims.product());

		if (bufferIn.normalizedHistogramDevice!=NULL)
			HANDLE_ERROR(cudaMemcpy(normalizedHistogramDevice,bufferIn.normalizedHistogramDevice,sizeof(double)*NUM_BINS,cudaMemcpyHostToHost));
	}

	void defaults()
	{
		imageDims = Vec<int>(-1,-1,-1);
		device = -1;
		currentBuffer = -1;
		for (int i=0; i<NUM_BUFFERS; ++i)
		{
			imageBuffers[i] = NULL;
		}
		histogramHost = NULL;
		histogramDevice = NULL;
		normalizedHistogramHost = NULL;
		normalizedHistogramDevice = NULL;
	}

	void clean() 
	{
		for (int i=0; i<NUM_BUFFERS; ++i)
		{
			if (imageBuffers[i]!=NULL)
			{
				HANDLE_ERROR(cudaFree(imageBuffers[0]));
				imageBuffers[i] = NULL;
			}
		}

		if (histogramHost!=NULL)
		{
			delete[] histogramHost;
		}

		if (histogramDevice!=NULL)
		{
			HANDLE_ERROR(cudaFree(histogramDevice));
			histogramDevice = NULL;
		}

		if (normalizedHistogramHost!=NULL)
		{
			delete[] normalizedHistogramHost;
		}

		if (normalizedHistogramDevice!=NULL)
		{
			HANDLE_ERROR(cudaFree(normalizedHistogramDevice));
			normalizedHistogramDevice = NULL;
		}
	}

	ImagePixelType* getCurrentBuffer() const 
	{
		if (currentBuffer<0 || currentBuffer>NUM_BUFFERS)
			return NULL;

		return imageBuffers[currentBuffer];
	}

	ImagePixelType* getNextBuffer()
	{
		int nextIndex = currentBuffer +1;
		if (nextIndex>=NUM_BUFFERS)
			nextIndex = 0;

		return imageBuffers[nextIndex];
	}

	void incrementBufferNumber()
	{
		++currentBuffer;

		if (currentBuffer>=NUM_BUFFERS)
			currentBuffer = 0;
	}

	Vec<int> imageDims;
	int device;
	cudaDeviceProp deviceProp;
	dim3 blocks, threads;
	char currentBuffer;
	ImagePixelType* imageBuffers[NUM_BUFFERS];
	unsigned int* histogramHost;
	unsigned int* histogramDevice;
	double* normalizedHistogramHost;
	double* normalizedHistogramDevice;
};