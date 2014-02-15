#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC
#include "Defines.h"
#include "ImageContainer.h"
#include "CudaImageContainerClean.cuh"
#include <vector>

struct ImageChunk
{
	Vec<size_t> startImageIdx;
	Vec<size_t> startBuffIdx;
	Vec<size_t> endImageIdx;
	Vec<size_t> endBuffIdx;

	ImageContainer* image;
};

class CudaProcessBuffer
{
public:
	//////////////////////////////////////////////////////////////////////////
	// Constructor / Destructor
	//////////////////////////////////////////////////////////////////////////
	
	// Creates an empty object on the desired device
	CudaProcessBuffer(int device=0);
	
	// Creates an object with all of the buffers set up to accommodate an image
	// of size dims and loads the passed int image into them
	// If the image will fit directly into the device buffers, the image memory
	// is used directly.  In other words, imageIn is not copied but used directly
	//CudaProcessBuffer(DevicePixelType* imageIn, Vec<size_t> dims, int device=0);

	// Destroys all pointers to memory on the device and the host
	// ENSURE that you have retrieved all data from this object prior to destruction
	~CudaProcessBuffer();

	//////////////////////////////////////////////////////////////////////////
	// Copy Constructors
	//////////////////////////////////////////////////////////////////////////
	CudaProcessBuffer (const CudaProcessBuffer* bufferIn);
	CudaProcessBuffer& operator=(const CudaProcessBuffer* bufferIn);

	//////////////////////////////////////////////////////////////////////////
	// Setters / Getters
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	// Cuda Operators (Alphabetical order)
	//////////////////////////////////////////////////////////////////////////
	// 	/*
	// 	*	Add a constant to all pixel values
	// 	*/
	 	void addConstant(double additive);
	// 	{
	// #if (CUDA_CALLS_ON)
	// 		cudaAddFactor<<<blocks,threads>>>(*getCurrentBuffer(),*getNextBuffer(),additive,minPixel,maxPixel);
	// #endif
	// 		incrementBufferNumber();
	// 	}
	// 
	// 	/*
	// 	*	Adds this image to the passed in one.  You can apply a factor
	// 	*	which is multiplied to the passed in image prior to adding
	// 	*/
	 	void addImageWith(const DevicePixelType* image, double factor);
	// 	{
	// #if (CUDA_CALLS_ON)
	// 		cudaAddTwoImagesWithFactor<<<blocks,threads>>>(*getCurrentBuffer(),*(image->getCurrentBuffer()),*getNextBuffer(),factor,
	// 			minPixel,maxPixel);
	// #endif
	// 		incrementBufferNumber();
	// 	}
	// 
	// 	/*
	// 	*	New pixel values will be a*x^2 + b*x + c where x is the original
	// 	*	pixel value.  This new value will be clamped between the min and
	// 	*	max values.
	// 	*/
	 	void applyPolyTransformation(double a, double b, double c, DevicePixelType minValue, DevicePixelType maxValue);
	// 	{
	// #if CUDA_CALLS_ON
	// 		cudaPolyTransferFuncImage<<<blocks,threads>>>(*getCurrentBuffer(),*getNextBuffer(),a,b,c,minValue,maxValue);
	// #endif
	// 		incrementBufferNumber();
	// 	}
	// 
	// 	/*
	// 	*	This will find the min and max values of the image
	// 	*/ 
	// 	template<typename rtnValueType>
	 	void calculateMinMax(double& minValue, double& maxValue);
	// 	{
	// 		double* maxValuesHost = new double[(blocks.x+1)/2];
	// 		double* minValuesHost = new double[(blocks.x+1)/2];
	// 
	// #if CUDA_CALLS_ON
	// 		cudaFindMinMax<<<sumBlocks,sumThreads,2*sizeof(double)*sumThreads.x>>>(*getCurrentBuffer(),minValuesDevice,deviceSum,
	// 			imageDims.product());
	// #endif
	// 
	// 		HANDLE_ERROR(cudaMemcpy(maxValuesHost,deviceSum,sizeof(double)*sumBlocks.x,cudaMemcpyDeviceToHost));
	// 		HANDLE_ERROR(cudaMemcpy(minValuesHost,minValuesDevice,sizeof(double)*sumBlocks.x,cudaMemcpyDeviceToHost));
	// 
	// 		maxValue = maxValuesHost[0];
	// 		minValue = minValuesHost[0];
	// 
	// 		for (size_t i=1; i<sumBlocks.x; ++i)
	// 		{
	// 			if (maxValue < maxValuesHost[i])
	// 				maxValue = maxValuesHost[i];
	// 
	// 			if (minValue > minValuesHost[i])
	// 				minValue = minValuesHost[i];
	// 		}
	// 
	// 		delete[] maxValuesHost;
	// 		delete[] minValuesHost;
	// 	}
	// 
	// 	/*
	// 	*	Contrast Enhancement will run the Michel High Pass Filter and then a mean filter
	// 	*	Pass in the sigmas that will be used for the Gaussian filter to subtract off and the mean neighborhood dimensions
	// 	*/
	 	void contrastEnhancement(Vec<float> sigmas, Vec<size_t> medianNeighborhood);
	// 	{
	// 		reserveCurrentBuffer();
	// 
	// 		gaussianFilter(sigmas);
	// #if CUDA_CALLS_ON
	// 		cudaAddTwoImagesWithFactor<<<blocks,threads>>>(*getReservedBuffer(),*getCurrentBuffer(),*getNextBuffer(),-1.0,minPixel,maxPixel);
	// #endif
	// 
	// 
	// 		incrementBufferNumber();
	// 		releaseReservedBuffer();
	// 
	// 		medianFilter(medianNeighborhood);
	// 	}
	// 
	// 	/*
	// 	*	Creates Histogram on the card using the #define NUM_BINS
	// 	*	Use retrieveHistogram to get the results off the card
	// 	*/
	 	void createHistogram();
	// 	{
	// 		if (isCurrentHistogramDevice)
	// 			return;
	// 
	// 		memset(histogramHost,0,NUM_BINS*sizeof(size_t));
	// 		HANDLE_ERROR(cudaMemset(histogramDevice,0,NUM_BINS*sizeof(size_t)));
	// 
	// #if CUDA_CALLS_ON
	// 		cudaHistogramCreate<<<deviceProp.multiProcessorCount*2,NUM_BINS,sizeof(size_t)*NUM_BINS>>>
	// 			(*getCurrentBuffer(),histogramDevice);
	// #endif
	// 
	// 		isCurrentHistogramDevice = true;
	// 	}
	// 
	// 	/*
	// 	*	Will smooth the image using the given sigmas for each dimension
	// 	*/ 
	 	DevicePixelType* gaussianFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<float> sigmas, DevicePixelType** imageOut=NULL);
	// 
	// 	/*
	// 	*	Mask will mask out the pixels of this buffer given an image and a threshold.
	// 	*	The threshold it to allow the input image to be gray scale instead of logical.
	// 	*	This buffer will get zeroed out where the imageMask is less than or equal to the threshold.
	// 	*/
	 	void mask(const DevicePixelType* imageMask, DevicePixelType threshold=1);
	// 	{
	// #if CUDA_CALLS_ON
	// 		cudaMask<<<blocks,threads>>>(*getCurrentBuffer(),*(imageMask->getCudaBuffer()),*getNextBuffer(),threshold);
	// #endif
	// 		incrementBufferNumber();
	// 	}
	// 
	// 	/*
	// 	*	Sets each pixel to the max value of its neighborhood
	// 	*	Dilates structures
	// 	*/ 
	 	void maxFilter(Vec<size_t> neighborhood, double* kernel=NULL);
	// 	{
	// 		if (kernel==NULL)
	// 			constKernelOnes();
	// 		else
	// 			setConstKernel(kernel,neighborhood);
	// 
	// #if CUDA_CALLS_ON
	// 		cudaMaxFilter<<<blocks,threads>>>(*getCurrentBuffer(),*getNextBuffer(),neighborhood);
	// #endif
	// 		incrementBufferNumber();
	// 	}
	// 
	// 	/*
	// 	*	produce an image that is the maximum value in z for each (x,y)
	// 	*	Images that are copied out of the buffer will have a z size of 1
	// 	*/
	 	void maximumIntensityProjection();
	// 	{
	// #if CUDA_CALLS_ON
	// 		cudaMaximumIntensityProjection<<<blocks,threads>>>(*getCurrentBuffer(),*getNextBuffer());
	// #endif
	// 		imageDims.z = 1;
	// 		updateBlockThread();
	// 		incrementBufferNumber();
	// 	}

		

		// Filters image where each pixel is the mean of its neighborhood 
		// If imageOut is null, then a new image pointer will be created and returned.
		// In either case the caller must clean up the the return image correctly
	 	DevicePixelType* meanFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, DevicePixelType** imageOut=NULL);

	// 	/*
	// 	*	Filters image where each pixel is the median of its neighborhood
	// 	*/
	 	void medianFilter(Vec<size_t> neighborhood);
	// 	{
	// 		static dim3 localBlocks = blocks;
	// 		static dim3 localThreads = threads;
	// 		size_t sharedMemorySize = neighborhood.product()*localThreads.x*localThreads.y*localThreads.z;
	// 		if (sizeof(ImagePixelType)*sharedMemorySize>deviceProp.sharedMemPerBlock)
	// 		{
	// 			float maxThreads = (float)deviceProp.sharedMemPerBlock/(sizeof(ImagePixelType)*neighborhood.product());
	// 			size_t threadDim = (size_t)pow(maxThreads,1/3.0f);
	// 			localThreads.x = (unsigned int)threadDim;
	// 			localThreads.y = (unsigned int)threadDim;
	// 			localThreads.z = (unsigned int)threadDim;
	// 
	// 			localBlocks.x = (size_t)ceil((float)imageDims.x/localThreads.x);
	// 			localBlocks.y = (size_t)ceil((float)imageDims.y/localThreads.y);
	// 			localBlocks.z = (size_t)ceil((float)imageDims.z/localThreads.z);
	// 
	// 			sharedMemorySize = neighborhood.product()*localThreads.x*localThreads.y*localThreads.z;
	// 		}
	// 
	// #if CUDA_CALLS_ON
	// 		cudaMedianFilter<<<localBlocks,localThreads,sizeof(ImagePixelType)*sharedMemorySize>>>(*getCurrentBuffer(),*getNextBuffer(),
	// 			neighborhood);
	// #endif
	// 
	// 		incrementBufferNumber();
	// 	}
	// 
	// 	/*
	// 	*	Sets each pixel to the min value of its neighborhood
	// 	*	Erodes structures
	// 	*/ 
	 	void minFilter(Vec<size_t> neighborhood, double* kernel=NULL);
	// 	{
	// 		if (kernel==NULL)
	// 			constKernelOnes();
	// 		else
	// 			setConstKernel(kernel,neighborhood);
	// 
	// #if CUDA_CALLS_ON
	// 		cudaMinFilter<<<blocks,threads>>>(*getCurrentBuffer(),*getNextBuffer(),neighborhood);
	// #endif
	// 		incrementBufferNumber();
	// 	}
	// 
	 	void morphClosure(Vec<size_t> neighborhood, double* kernel=NULL);
	// 	{
	// 		maxFilter(neighborhood,kernel);
	// 		minFilter(neighborhood,kernel);
	// 	}
	// 
	 	void morphOpening(Vec<size_t> neighborhood, double* kernel=NULL);
	// 	{
	// 		minFilter(neighborhood,kernel);
	// 		maxFilter(neighborhood,kernel);
	// 	}
	// 
	// 	/*
	// 	*	Sets each pixel by multiplying by the original value and clamping
	// 	*	between minValue and maxValue
	// 	*/
	// 	template<typename FactorType>
	 	void multiplyImage(double factor);
	// 	{
	// #if CUDA_CALLS_ON
	// 		cudaMultiplyImage<<<blocks,threads>>>(*getCurrentBuffer(),*getNextBuffer(),factor,minPixel,maxPixel);
	// #endif
	// 		incrementBufferNumber();
	// 	}
	// 
	// 	/*
	// 	*	Multiplies this image to the passed in one.
	// 	*/
	 	void multiplyImageWith(const DevicePixelType* image);
	// 	{
	// #if CUDA_CALLS_ON
	// 		cudaMultiplyTwoImages<<<blocks,threads>>>(*getCurrentBuffer(),*(image->getCurrentBuffer()),*getNextBuffer());
	// #endif
	// 		incrementBufferNumber();
	// 	}
	// 
	// 	/*
	// 	*	This will calculate the normalized covariance between the two images A and B
	// 	*	returns (sum over all{(A-mu(A)) X (B-mu(B))}) / (sigma(A)Xsigma(B)
	// 	*	The images buffers will not change the original data 
	// 	*/
	 	double normalizedCovariance(DevicePixelType* otherImage);
	// 	{
	// 		ImagePixelType* aOrg = this->retrieveImage();
	// 		ImagePixelType* bOrg = otherImage->retrieveImage();
	// 
	// 		double aSum1 = 0.0;
	// 		double bSum1 = 0.0;
	// 
	// 		this->sumArray(aSum1);
	// 		otherImage->sumArray(bSum1);
	// 
	// 		double aMean = aSum1/this->imageDims.product();
	// 		double bMean = bSum1/otherImage->getDimension().product();
	// 
	// 		this->addConstant(-aMean);
	// 		otherImage->addConstant(-bMean);
	// 
	// 		double aMidSum;
	// 		double bMidSum;
	// 
	// 		if (imageDims.z>1)
	// 		{
	// 			this->sumArray(aMidSum);
	// 			otherImage->sumArray(bMidSum);
	// 		}
	// 
	// 		this->reserveCurrentBuffer();
	// 		otherImage->reserveCurrentBuffer();
	// 
	// 		this->imagePow(2);
	// 		otherImage->imagePow(2);
	// 
	// 		double aSum2 = 0.0;
	// 		double bSum2 = 0.0;
	// 		this->sumArray(aSum2);
	// 		otherImage->sumArray(bSum2);
	// 
	// 		double aSigma = sqrt(aSum2/this->getDimension().product());
	// 		double bSigma = sqrt(bSum2/otherImage->getDimension().product());
	// 
	// 		this->currentBuffer = this->reservedBuffer;
	// 		otherImage->currentBuffer = otherImage->reservedBuffer;
	// 
	// 		this->releaseReservedBuffer();
	// 		otherImage->releaseReservedBuffer();
	// 
	// #if CUDA_CALLS_ON
	// 		cudaMultiplyTwoImages<<<blocks,threads>>>(*(this->getCurrentBuffer()),*(otherImage->getCurrentBuffer()),*(this->getNextBuffer()));
	// #endif
	// 		this->incrementBufferNumber();
	// 
	// 		double multSum = 0.0;
	// 		this->sumArray(multSum);
	// 
	// 		this->loadImage(aOrg,this->getDimension());
	// 		otherImage->loadImage(bOrg,otherImage->getDimension());
	// 
	// 		delete[] aOrg;
	// 		delete[] bOrg;
	// 
	// 		double rtn = multSum/(aSigma*bSigma) / this->getDimension().product();
	// 
	// 		return rtn;
	// 	}
	// 
	// 	/*
	// 	*	Takes a histogram that is on the card and normalizes it
	// 	*	Will generate the original histogram if one doesn't already exist
	// 	*	Use retrieveNormalizedHistogram() to get a host pointer
	// 	*/
	 	void normalizeHistogram();
	// 	{
	// 		if (isCurrentNormHistogramDevice)
	// 			return;
	// 
	// 		if(!isCurrentHistogramDevice)
	// 			createHistogram();
	// 
	// #if CUDA_CALLS_ON
	// 		cudaNormalizeHistogram<<<NUM_BINS,1>>>(histogramDevice,normalizedHistogramDevice,imageDims);
	// #endif
	// 		isCurrentNormHistogramDevice = true;
	// 	}
	// 
	 	void otsuThresholdFilter(float alpha=1.0f);
	// 	{
	// 		ImagePixelType thresh = otsuThresholdValue();
	// 		thresholdFilter(thresh*alpha);
	// 	}
	// 
	// 	/*
	// 	*	Raise each pixel to a power
	// 	*/
	 	void imagePow(int p);
	// 	{
	// #if CUDA_CALLS_ON
	// 		cudaPow<<<blocks,threads>>>(*getCurrentBuffer(),*getNextBuffer(),p);
	// #endif
	// 		incrementBufferNumber();
	// 	}
	// 
	// 	/*
	// 	*	Calculates the total sum of the buffer's data
	// 	*/
	 	void sumArray(double& sum);
	// 	{
	// #if CUDA_CALLS_ON
	// 		cudaSumArray<<<sumBlocks,sumThreads,sizeof(double)*sumThreads.x>>>(*getCurrentBuffer(),deviceSum,imageDims.product());		
	// #endif
	// 		HANDLE_ERROR(cudaMemcpy(hostSum,deviceSum,sizeof(double)*sumBlocks.x,cudaMemcpyDeviceToHost));
	// 
	// 		sum = 0;
	// 		for (size_t i=0; i<sumBlocks.x; ++i)
	// 		{
	// 			sum += hostSum[i];
	// 		}
	// 	}
	// 
	// 	/*
	// 	*	Will reduce the size of the image by the factors passed in
	// 	*/
	 	HostPixelType* CudaProcessBuffer::reduceImage(const DevicePixelType* image, Vec<size_t> dims, Vec<double> reductions, Vec<size_t>& reducedDims);

		/*
		*	This creates a image with values of 0 where the pixels fall below
		*	the threshold and 1 where equal or greater than the threshold
		*	
		*	If you want a viewable image after this, you may want to use the
		*	multiplyImage routine to turn the 1 values to the max values of
		*	the type
		*/
	// 	template<typename ThresholdType>
	 	void thresholdFilter(double threshold);
	// 	{
	// #if CUDA_CALLS_ON
	// 		cudaThresholdImage<<<blocks,threads>>>(*getCurrentBuffer(),*getNextBuffer(),(DevicePixelType)threshold,
	// 			(DevicePixelType)minPixel,(DevicePixelType)maxPixel);
	// #endif
	// 		incrementBufferNumber();
	// 	}
	// 
	 	void unmix(const DevicePixelType* image, Vec<size_t> neighborhood);
	// 	{
	// #if CUDA_CALLS_ON
	// 		cudaUnmixing<<<blocks,threads>>>(*getCurrentBuffer(),*(image->getCudaBuffer()),*getNextBuffer(), neighborhood,minPixel,maxPixel);
	// #endif
	// 		incrementBufferNumber();
	// 	}

private:

	//////////////////////////////////////////////////////////////////////////
	// Private Helper Methods
	//////////////////////////////////////////////////////////////////////////
	void updateBlockThread();
	void deviceSetup();
	void calculateChunking(Vec<size_t> kernalDims);
	void defaults();
	void createBuffers();
	void clearBuffers();

	void clearDeviceBuffers()
	{
		for (int i=0; i<deviceImageBuffers.size(); ++i)
		{
			if (deviceImageBuffers[i]!=NULL)
			{
				delete deviceImageBuffers[i];
				deviceImageBuffers[i] = NULL;
			}
		}

		deviceImageBuffers.clear();
	}

	void loadImage(HostPixelType* imageIn);
	void createDeviceBuffers(int numBuffersNeeded, Vec<size_t> kernalDims);
	void CudaProcessBuffer::incrementBufferNumber();
	CudaImageContainer* CudaProcessBuffer::getCurrentBuffer();
	CudaImageContainer* CudaProcessBuffer::getNextBuffer();
	bool loadNextChunk(const DevicePixelType* imageIn);
	void saveCurChunk(DevicePixelType* imageOut);

	//////////////////////////////////////////////////////////////////////////
	// Private Member Variables
	//////////////////////////////////////////////////////////////////////////

	// Device that this buffer will utilize
	int device;
	cudaDeviceProp deviceProp;

	dim3 blocks, threads;

	// This is the size that the input and output images will be
	Vec<size_t> orgImageDims;

	// This is how many chunks are being used to cover the whole original image
	Vec<size_t> numChunks;
	Vec<size_t> curChunkIdx;

	ImageChunk* hostImageBuffers;

	int currentBufferIdx;
	Vec<size_t> deviceDims;
	std::vector<CudaImageContainerClean*> deviceImageBuffers;

	// This is the maximum size that we are allowing a constant kernel to exit
	// on the device
	float hostKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];

};
