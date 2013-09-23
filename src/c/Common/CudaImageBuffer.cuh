#pragma once

#include "defines.h"
#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC
#include "CudaKernals.cuh"
#include "CudaUtilities.cuh"
#include "assert.h"

template<typename ImagePixelType>
class CudaImageBuffer
{
public:
	//////////////////////////////////////////////////////////////////////////
	// Constructor / Destructor
	//////////////////////////////////////////////////////////////////////////

	CudaImageBuffer(Vec<unsigned int> dims, int device=0)
	{
		defaults();

		this->imageDims = dims;
		this->device = device;

		deviceSetup();
		memoryAllocation();
	}

	CudaImageBuffer(unsigned int x, unsigned int y, unsigned int z, int device=0)
	{
		defaults();

		imageDims = Vec<unsigned int>(x,y,z);
		this->device = device;

		deviceSetup();
		memoryAllocation();
	}

	CudaImageBuffer(int n, int device=0)
	{
		defaults();

		imageDims = Vec<unsigned int>(n,1,1);
		this->device = device;

		deviceSetup();
		memoryAllocation();
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
		return *this;
	}

	// End Copy Constructors

	//////////////////////////////////////////////////////////////////////////
	// Setters / Getters
	//////////////////////////////////////////////////////////////////////////

	void loadImage(const ImagePixelType* image)
	{
		calcBlockThread(imageDims,deviceProp,blocks,threads);
		incrementBufferNumber();
		HANDLE_ERROR(cudaMemcpy(imageBuffers[currentBuffer],image,sizeof(ImagePixelType)*imageDims.product(),cudaMemcpyHostToDevice));
	}

	/*
	 *	This will use the pointer that has been passed in to fill.  
	 *	Ensure that it is big enough to hold the current buffer.
	 *
	 *  If no pointer is provided, then new heap memory will be allocated
	 *  and a pointer to this memory returned.
	 *  
	 *  Clean-up of each of these are your responsibility
	 */
	ImagePixelType* retrieveImage(ImagePixelType* imageOut=NULL)
	{
		if (currentBuffer<0 || currentBuffer>NUM_BUFFERS)
		{
			return NULL;
		}
		if (imageOut==NULL)
			imageOut = new ImagePixelType[imageDims.product()];

		HANDLE_ERROR(cudaMemcpy((void*)imageOut,imageBuffers[currentBuffer],sizeof(ImagePixelType)*imageDims.product(),cudaMemcpyDeviceToHost));
		return imageOut;
	}

	/*
	*	Returns a host pointer to the histogram data
	*	This is destroyed when this' destructor is called
	*	Will call the needed histogram creation methods if not all ready
	*/
	unsigned int* retrieveHistogram(int& returnSize)
	{
		if (histogramDevice==NULL)
			createHistogram();

		HANDLE_ERROR(cudaMemcpy(histogramHost,histogramDevice,sizeof(unsigned int)*NUM_BINS,cudaMemcpyDeviceToHost));
		returnSize = NUM_BINS;

		return histogramHost;
	}

	/*
	*	Returns a host pointer to the normalized histogram data
	*	This is destroyed when this' destructor is called
	*	Will call the needed histogram creation methods if not all ready
	*/
	double* retrieveNormalizedHistogram(int& returnSize)
	{
		if (normalizedHistogramDevice==NULL)
			normalizeHistogram();

		HANDLE_ERROR(cudaMemcpy(normalizedHistogramHost,normalizedHistogramDevice,sizeof(double)*NUM_BINS,cudaMemcpyDeviceToHost));
		returnSize = NUM_BINS;

		return normalizedHistogramHost;
	}

	ImagePixelType* retrieveReducedImage(Vec<unsigned int>& reducedDims)
	{
		reducedDims = this->reducedDims;

		if (reducedImageDevice!=NULL)
		{
			HANDLE_ERROR(cudaMemcpy(reducedImageHost,reducedImageDevice,sizeof(ImagePixelType)*reducedDims.product(),cudaMemcpyDeviceToHost));
		}

		return reducedImageHost;
	}

	Vec<unsigned int> getDimension() const {return imageDims;}
	int getDevice() const {return device;}
	size_t getBufferSize() {return bufferSize;}

	/*
	*	This will replace this' cuda image buffer with the region of interest
	*	from the passed in buffer.
	*	****ENSURE that this' original size is big enough to accommodates the
	*	the new buffer size.  Does not do error checking thus far.
	*/
	void copyROI(const CudaImageBuffer<ImagePixelType>& bufferIn, Vec<unsigned int> starts, Vec<unsigned int> sizes)
	{
		assert(sizes.product()<=bufferSize);

		imageDims = sizes;
		device = bufferIn.getDevice();
		currentBuffer = 0;
		bufferIn.getRoi(getCurrentBuffer(),starts,sizes);
		calcBlockThread(imageDims,deviceProp,blocks,threads);
	}

	void copyImage(const CudaImageBuffer<ImagePixelType>& bufferIn)
	{
		assert(bufferIn.getDimension().product()<=bufferSize);

		imageDims = bufferIn.getDimension();
		device = bufferIn.getDevice();
		calcBlockThread(imageDims,deviceProp,blocks,threads);

		currentBuffer = 0;
		HANDLE_ERROR(cudaMemcpy(getCurrentBuffer(),bufferIn.getCudaBuffer(),sizeof(ImagePixelType)*imageDims.product(),
			cudaMemcpyDeviceToDevice));
	}

	const ImagePixelType* getCudaBuffer() const
	{
		return getCurrentBuffer();
	}

	// End Setters / Getters

	//////////////////////////////////////////////////////////////////////////
	// Cuda Operators (Alphabetical order)
	//////////////////////////////////////////////////////////////////////////

	/*
	*	Add a constant to all pixel values
	*/
	template<typename T>
	void addConstant(T additive)
	{
		cudaAddFactor<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,additive);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
		incrementBufferNumber();
	}

	/*
	*	Adds this image to the passed in one.  You can apply a factor
	*	which is multiplied to the passed in image prior to adding
	*/
	void addImageWith(const CudaImageBuffer* image, double factor)
	{
		calcBlockThread(imageDims,deviceProp,blocks,threads);
		ImagePixelType mn = std::numeric_limits<ImagePixelType>::min();
		ImagePixelType mx = std::numeric_limits<ImagePixelType>::max();
		cudaAddTwoImagesWithFactor<<<blocks,threads>>>(getCurrentBuffer(),image->getCurrentBuffer(),getNextBuffer(),
			imageDims,factor,mn,mx);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
		incrementBufferNumber();
	}

	/*
	*	New pixel values will be a*x^2 + b*x + c where x is the original
	*	pixel value.  This new value will be clamped between the min and
	*	max values.
	*/
	template<typename ThresholdType>
	void applyPolyTransformation(ThresholdType a, ThresholdType b, ThresholdType c, ImagePixelType minValue, ImagePixelType maxValue)
	{
		cudaPolyTransferFuncImage<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,a,b,c,maxValue,minValue);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
		incrementBufferNumber();
	}

	/*
	*	This will find the min and max values of the image
	*/ 
	void calculateMinMax(ImagePixelType& minValue, ImagePixelType& maxValue)
	{
		ImagePixelType* maxValuesHost = new ImagePixelType[(blocks.x+1)/2];
		ImagePixelType* minValuesHost = new ImagePixelType[(blocks.x+1)/2];
		ImagePixelType* maxValuesDevice;
		ImagePixelType* minValuesDevice;

		HANDLE_ERROR(cudaMalloc((void**)&maxValuesDevice,sizeof(double)*(blocks.x+1)/2));
		HANDLE_ERROR(cudaMalloc((void**)&minValuesDevice,sizeof(double)*(blocks.x+1)/2));

		cudaFindMinMax<<<blocks.x,threads.x,2*sizeof(double)*threads.x>>>(getCurrentBuffer(),minValuesDevice,maxValuesDevice,
			imageDims.product());
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG

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

		delete[] maxValuesHost;
		delete[] minValuesHost;
	}

	/*
	*	Creates Histogram on the card using the #define NUM_BINS
	*	Use retrieveHistogram to get the results off the card
	*/
	void createHistogram()
	{
		histogramHost = new unsigned int[NUM_BINS];
		HANDLE_ERROR(cudaMalloc((void**)&histogramDevice,NUM_BINS*sizeof(unsigned int)));

		memset(histogramHost,0,NUM_BINS*sizeof(unsigned int));
		HANDLE_ERROR(cudaMemset(histogramDevice,0,NUM_BINS*sizeof(unsigned int)));

		cudaHistogramCreate<<<deviceProp.multiProcessorCount*2,NUM_BINS,sizeof(unsigned int)*NUM_BINS>>>
			(getCurrentBuffer(),histogramDevice,imageDims);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
	}

	/*
	*	Will smooth the image using the given sigmas for each dimension
	*/ 
	void gaussianFilter(Vec<double> sigmas)
	{
	}

	/*
	*	Sets each pixel to the max value of its neighborhood
	*	Dilates structures
	*/ 
	void maxFilter(Vec<int> neighborhood)
	{
		cudaMaxFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,neighborhood);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
		incrementBufferNumber();
	}

	/*
	*	produce an image that is the maximum value in z for each (x,y)
	*	Images that are copied out of the buffer will have a z size of 1
	*/
	void maximumIntensityProjection()
	{
		cudaMaximumIntensityProjection<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
		imageDims.z = 1;
		calcBlockThread(imageDims,deviceProp,blocks,threads);
		incrementBufferNumber();
	}

	/*
	*	Filters image where each pixel is the mean of its neighborhood 
	*/
	void meanFilter(Vec<int> neighborhood)
	{
		cudaMeanFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,neighborhood);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
		incrementBufferNumber();
	}

	/*
	*	Filters image where each pixel is the median of its neighborhood
	*/
	void medianFilter(Vec<int> neighborhood)
	{
		cudaMedianFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,neighborhood);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
		incrementBufferNumber();
	}

	/*
	*	Sets each pixel to the min value of its neighborhood
	*	Erodes structures
	*/ 
	void minFilter(Vec<int> neighborhood)
	{
		cudaMinFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,neighborhood);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
		incrementBufferNumber();
	}

	/*
	*	Sets each pixel by multiplying by the original value and clamping
	*	between minValue and maxValue
	*/
	template<typename FactorType>
	void multiplyImage(FactorType factor, ImagePixelType minValue, ImagePixelType maxValue)
	{
		cudaMultiplyImage<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,factor,minValue,maxValue);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
		incrementBufferNumber();
	}

	/*
	*	Multiplies this image to the passed in one.
	*/
	void multiplyImageWith(const CudaImageBuffer* image)
	{
		cudaMultiplyTwoImages<<<blocks,threads>>>(getCurrentBuffer(),image->getCurrentBuffer(),getNextBuffer(),imageDims);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
		incrementBufferNumber();
	}

	/*
	*	Takes a histogram that is on the card and normalizes it
	*	Will generate the original histogram if one doesn't already exist
	*	Use retrieveNormalizedHistogram() to get a host pointer
	*/
	void normalizeHistogram()
	{
		if(histogramDevice==NULL)
			createHistogram();

		normalizedHistogramHost = new double[NUM_BINS];
		HANDLE_ERROR(cudaMalloc((void**)&normalizedHistogramDevice,NUM_BINS*sizeof(double)));
		cudaNormalizeHistogram<<<NUM_BINS,1>>>(histogramDevice,normalizedHistogramDevice,imageDims);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
	}

	/*
	*	Raise each pixel to a power
	*/
	template<typename PowerType>
	void pow(PowerType p)
	{
		cudaPow<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,p);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
		incrementBufferNumber();
	}

	/*
	*	Calculates the total sum of the buffer's data
	*/
	template<typename Sumtype>
	void sumArray(Sumtype& sum)
	{
		calcBlockThread(Vec<unsigned int>((unsigned int)imageDims.product(),1,1),deviceProp,sumBlocks,sumThreads);
		sumBlocks.x = (sumBlocks.x+1) / 2;

		cudaSumArray<<<sumBlocks,sumThreads,sizeof(double)*sumThreads.x>>>(getCurrentBuffer(),deviceSum,(unsigned int)imageDims.product());
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG

		hostSum = new double[sumBlocks.x];
		HANDLE_ERROR(cudaMemcpy(hostSum,deviceSum,sizeof(double)*sumBlocks.x,cudaMemcpyDeviceToHost));

		sum = 0;
		for (unsigned int i=0; i<sumBlocks.x; ++i)
		{
			sum += hostSum[i];
		}

		delete[] hostSum;
	}

	/*
	*	Will reduce the size of the image by the factors passed in
	*/
	void reduceImage(Vec<double> reductions)
	{
		reducedDims = Vec<unsigned int>(
			(unsigned int)(imageDims.x/reductions.x),
			(unsigned int)(imageDims.y/reductions.y),
			(unsigned int)(imageDims.z/reductions.z));

		cudaRuduceImage<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,reducedDims,reductions);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
		incrementBufferNumber();
	}

	/*
	*	This creates a image with values of 0 where the pixels fall below
	*	the threshold and 1 where equal or greater than the threshold
	*	
	*	If you want a viewable image after this, you may want to use the
	*	multiplyImage routine to turn the 1 values to the max values of
	*	the type
	*/
	template<typename ThresholdType>
	void thresholdFilter(ThresholdType threshold)
	{
		cudaThresholdImage<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,threshold);
#ifdef _DEBUG
		gpuErrchk( cudaPeekAtLastError() );
#endif // _DEBUG
		incrementBufferNumber();
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

	void memoryAllocation()
	{
		assert(sizeof(ImagePixelType)*imageDims.product()*NUM_BUFFERS < deviceProp.totalGlobalMem*.6);

		for (int i=0; i<NUM_BUFFERS; ++i)
		{
			HANDLE_ERROR(cudaMalloc((void**)&imageBuffers[i],sizeof(ImagePixelType)*imageDims.product()));
		}

		currentBuffer = -1;
		bufferSize = imageDims.product();

		calcBlockThread(Vec<unsigned int>((unsigned int)imageDims.product(),1,1),deviceProp,sumBlocks,sumThreads);

		sumBlocks.x = (sumBlocks.x+1) / 2;

		HANDLE_ERROR(cudaMalloc((void**)&deviceSum,sizeof(double)*sumBlocks.x));
	}

	void reduceMemory()
	{
		for (int i=0; i<NUM_BUFFERS; ++i)
		{
			if (imageBuffers[i]!=NULL)
				HANDLE_ERROR(cudaFree(imageBuffers[i]));
		}

		for (int i=0; i<NUM_BUFFERS; ++i)
		{
			HANDLE_ERROR(cudaMalloc((void**)&imageBuffers[i],sizeof(ImagePixelType)*imageDims.product()));
		}

		currentBuffer = -1;
	}

	void getRoi(ImagePixelType* roi, Vec<unsigned int> starts, Vec<unsigned int> sizes) const
	{
		cudaGetROI<<<blocks,threads>>>(getCurrentBuffer(),roi,imageDims,starts,sizes);
	}

	void copy(const CudaImageBuffer<ImagePixelType>& bufferIn)
	{
		defaults();

		imageDims = bufferIn.getDimension();
		device = bufferIn.getDevice();

		memoryAllocation();

		calcBlockThread(imageDims,deviceProp,blocks,threads);

		currentBuffer = 0;
		ImagePixelType* inImage = bufferIn.getCurrentBuffer();

		if (inImage!=NULL)
			HANDLE_ERROR(cudaMemcpy(imageBuffers[currentBuffer],inImage,sizeof(ImagePixelType)*imageDims.product(),cudaMemcpyDeviceToDevice));

		if (bufferIn.reducedImageHost!=NULL)
			memcpy(reducedImageHost,bufferIn.reducedImageHost,sizeof(ImagePixelType)*reducedDims.product());

		if (bufferIn.reducedImageDevice!=NULL)
			HANDLE_ERROR(cudaMemcpy(reducedImageDevice,bufferIn.reducedImageDevice,sizeof(ImagePixelType)*reducedDims.product(),cudaMemcpyDeviceToDevice));

		if (bufferIn.histogramHost!=NULL)
			memcpy(histogramHost,bufferIn.histogramHost,sizeof(unsigned int)*imageDims.product());

		if (bufferIn.histogramDevice!=NULL)
			HANDLE_ERROR(cudaMemcpy(histogramDevice,bufferIn.histogramDevice,sizeof(unsigned int)*NUM_BINS,cudaMemcpyDeviceToDevice));

		if (bufferIn.normalizedHistogramHost!=NULL)
			memcpy(normalizedHistogramHost,bufferIn.normalizedHistogramHost,sizeof(double)*imageDims.product());

		if (bufferIn.normalizedHistogramDevice!=NULL)
			HANDLE_ERROR(cudaMemcpy(normalizedHistogramDevice,bufferIn.normalizedHistogramDevice,sizeof(double)*NUM_BINS,cudaMemcpyDeviceToDevice));
	}

	void defaults()
	{
		imageDims = Vec<unsigned int>((unsigned int)-1,(unsigned int)-1,(unsigned int)-1);
		reducedDims = Vec<unsigned int>((unsigned int)-1,(unsigned int)-1,(unsigned int)-1);
		device = -1;
		currentBuffer = -1;
		for (int i=0; i<NUM_BUFFERS; ++i)
		{
			imageBuffers[i] = NULL;
		}

		reducedImageHost = NULL;
		reducedImageDevice = NULL;
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
				HANDLE_ERROR(cudaFree(imageBuffers[i]));
		}

		if (reducedImageHost!=NULL)
			delete[] reducedImageHost;

		if (reducedImageDevice!=NULL)
			HANDLE_ERROR(cudaFree(reducedImageDevice));

		if (histogramHost!=NULL)
			delete[] histogramHost;

		if (histogramDevice!=NULL)
			HANDLE_ERROR(cudaFree(histogramDevice));

		if (normalizedHistogramHost!=NULL)
			delete[] normalizedHistogramHost;

		if (normalizedHistogramDevice!=NULL)
			HANDLE_ERROR(cudaFree(normalizedHistogramDevice));

		defaults();
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

	//This is the maximum size that the current buffer can handle 
	size_t bufferSize;

	//This is the original size of the loaded images and the size of the buffers
	Vec<unsigned int> imageDims;

	//This is the dimensions of the reduced image buffer
	Vec<unsigned int> reducedDims;

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
	dim3 sumBlocks, sumThreads;
	double* deviceSum;
	double* hostSum;
};
