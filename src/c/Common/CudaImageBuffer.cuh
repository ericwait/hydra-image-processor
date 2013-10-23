#pragma once

#include "defines.h"
#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC
#include "CudaKernels.cuh"
#include "CudaUtilities.cuh"
#include "assert.h"
#include <limits>

template<typename ImagePixelType>
class CudaImageBuffer
{
public:
	//////////////////////////////////////////////////////////////////////////
	// Constructor / Destructor
	//////////////////////////////////////////////////////////////////////////

	CudaImageBuffer(Vec<unsigned int> dims, bool columnMajor=false, int device=0)
	{
		defaults();
		isColumnMajor = columnMajor;
		this->imageDims = dims;
		this->device = device;
		UNSET = Vec<unsigned int>((unsigned int)-1,(unsigned int)-1,(unsigned int)-1);
		deviceSetup();
		memoryAllocation();
	}

	CudaImageBuffer(unsigned int x, unsigned int y, unsigned int z, bool columnMajor=false, int device=0)
	{
		defaults();
		isColumnMajor = columnMajor;
		imageDims = Vec<unsigned int>(x,y,z);
		this->device = device;
		UNSET = Vec<unsigned int>((unsigned int)-1,(unsigned int)-1,(unsigned int)-1);
		deviceSetup();
		memoryAllocation();
	}

	CudaImageBuffer(int n, bool columnMajor=false, int device=0)
	{
		defaults();
		isColumnMajor = columnMajor;
		imageDims = Vec<unsigned int>(n,1,1);
		this->device = device;

		UNSET = Vec<unsigned int>((unsigned int)-1,(unsigned int)-1,(unsigned int)-1);
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

	void loadImage(const ImagePixelType* image, Vec<unsigned int> imageInDims)
	{
		if (imageInDims.product()>bufferSize)
		{
			bool wasColumnMajor = isColumnMajor;
			int device = this->device;
			clean();
			isColumnMajor = wasColumnMajor;
			this->device = device;
			imageDims = imageInDims;
			deviceSetup();
			memoryAllocation();
		}
		else
		{
			isCurrentHistogramHost = false;
			isCurrentHistogramDevice = false;
			isCurrentNormHistogramHost = false;
			isCurrentNormHistogramDevice = false;
		}

		imageDims = imageInDims;
		currentBuffer = 0;
		reservedBuffer = -1;
		HANDLE_ERROR(cudaMemcpy((void*)getCurrentBuffer(),image,sizeof(ImagePixelType)*imageDims.product(),cudaMemcpyHostToDevice));
	}
	
	ImagePixelType otsuThresholdValue()
	{
		int temp;//TODO
		return calcOtsuThreshold(retrieveNormalizedHistogram(temp),NUM_BINS);
	}

	ImagePixelType* retrieveImage(ImagePixelType* imageOut=NULL)
	{
		if (currentBuffer<0 || currentBuffer>NUM_BUFFERS)
		{
			return NULL;
		}
		if (imageOut==NULL)
			imageOut = new ImagePixelType[imageDims.product()];

		HANDLE_ERROR(cudaMemcpy(imageOut,getCurrentBuffer(),sizeof(ImagePixelType)*imageDims.product(),cudaMemcpyDeviceToHost));
		return imageOut;
	}

	/*
	*	Returns a host pointer to the histogram data
	*	This is destroyed when this' destructor is called
	*	Will call the needed histogram creation methods if not all ready
	*/
	unsigned int* retrieveHistogram(int& returnSize)
	{
		if (!isCurrentNormHistogramHost)
		{
			createHistogram();

			HANDLE_ERROR(cudaMemcpy(histogramHost,histogramDevice,sizeof(unsigned int)*NUM_BINS,cudaMemcpyDeviceToHost));
			isCurrentHistogramHost = true;
		}

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
		if (!isCurrentNormHistogramHost)
		{
			normalizeHistogram();

			HANDLE_ERROR(cudaMemcpy(normalizedHistogramHost,normalizedHistogramDevice,sizeof(double)*NUM_BINS,cudaMemcpyDeviceToHost));
			isCurrentNormHistogramHost = true;
		}

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
		if (sizes.product()>bufferSize)
		{
			bool wasColumnMajor = isColumnMajor;
			int device = this->device;
			clean();
			isColumnMajor = wasColumnMajor;
			this->device = device;
			imageDims = bufferIn.getDimension();
			deviceSetup();
			memoryAllocation();
		}

		imageDims = sizes;
		device = bufferIn.getDevice();
		currentBuffer = 0;
		bufferIn.getRoi(getCurrentBuffer(),starts,sizes);
		updateBlockThread();
	}

	void copyImage(const CudaImageBuffer<ImagePixelType>& bufferIn)
	{
		if (bufferIn.getDimension().product()>bufferSize)
		{
			bool wasColumnMajor = isColumnMajor;
			int device = this->device;
			clean();
			isColumnMajor = wasColumnMajor;
			this->device = device;
			imageDims = bufferIn.getDimension();
			deviceSetup();
			memoryAllocation();
		}

		imageDims = bufferIn.getDimension();
		device = bufferIn.getDevice();
		updateBlockThread();

		currentBuffer = 0;
		HANDLE_ERROR(cudaMemcpy(getCurrentBuffer(),bufferIn.getCudaBuffer(),sizeof(ImagePixelType)*imageDims.product(),
			cudaMemcpyDeviceToDevice));
	}

	const ImagePixelType* getCudaBuffer() const
	{
		return getCurrentBuffer();
	}

	size_t getMemoryUsed() {return memoryUsage;}
	size_t getGlobalMemoryAvailable() {return deviceProp.totalGlobalMem;}

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
		cudaAddFactor<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,additive,minPixel,maxPixel,isColumnMajor);
		incrementBufferNumber();
	}

	/*
	*	Adds this image to the passed in one.  You can apply a factor
	*	which is multiplied to the passed in image prior to adding
	*/
	void addImageWith(const CudaImageBuffer* image, double factor)
	{
		cudaAddTwoImagesWithFactor<<<blocks,threads>>>(getCurrentBuffer(),image->getCurrentBuffer(),getNextBuffer(),
			imageDims,factor,minPixel,maxPixel,isColumnMajor);
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
		cudaPolyTransferFuncImage<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,a,b,c,minValue,maxValue,isColumnMajor);
		incrementBufferNumber();
	}

	/*
	*	This will find the min and max values of the image
	*/ 
	template<typename rtnValueType>
	void calculateMinMax(rtnValueType& minValue, rtnValueType& maxValue)
	{
		double* maxValuesHost = new double[(blocks.x+1)/2];
		double* minValuesHost = new double[(blocks.x+1)/2];

		cudaFindMinMax<<<sumBlocks,sumThreads,2*sizeof(double)*sumThreads.x>>>(getCurrentBuffer(),minValuesDevice,deviceSum,
			imageDims.product());

		HANDLE_ERROR(cudaMemcpy(maxValuesHost,deviceSum,sizeof(double)*sumBlocks.x,cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(minValuesHost,minValuesDevice,sizeof(double)*sumBlocks.x,cudaMemcpyDeviceToHost));

		maxValue = maxValuesHost[0];
		minValue = minValuesHost[0];

		for (unsigned int i=1; i<sumBlocks.x; ++i)
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
	 *	Contrast Enhancement will run the Michel High Pass Filter and then a mean filter
	 *	Pass in the sigmas that will be used for the Gaussian filter to subtract off and the mean neighborhood dimensions
	 */
	void contrastEnhancement(Vec<float> sigmas, Vec<unsigned int> medianNeighborhood)
	{
		reserveCurrentBuffer();

 		gaussianFilter(sigmas);
		cudaAddTwoImagesWithFactor<<<blocks,threads>>>(getReservedBuffer(),getCurrentBuffer(),getNextBuffer(),imageDims,
			-1.0,minPixel,maxPixel,isColumnMajor);

		incrementBufferNumber();
		releaseReservedBuffer();

		medianFilter(medianNeighborhood);
	}

	/*
	*	Creates Histogram on the card using the #define NUM_BINS
	*	Use retrieveHistogram to get the results off the card
	*/
	void createHistogram()
	{
		if (isCurrentHistogramDevice)
			return;

		memset(histogramHost,0,NUM_BINS*sizeof(unsigned int));
		HANDLE_ERROR(cudaMemset(histogramDevice,0,NUM_BINS*sizeof(unsigned int)));

		cudaHistogramCreate<<<deviceProp.multiProcessorCount*2,NUM_BINS,sizeof(unsigned int)*NUM_BINS>>>
			(getCurrentBuffer(),histogramDevice,imageDims);

		isCurrentHistogramDevice = true;
	}

	/*
	*	Will smooth the image using the given sigmas for each dimension
	*/ 
	void gaussianFilter(Vec<float> sigmas)
	{
		if (constKernelDims==UNSET || sigmas!=gausKernelSigmas)
		{
			constKernelZeros();
			gausKernelSigmas = sigmas;
			constKernelDims = createGaussianKernel(gausKernelSigmas,hostKernel,gaussIterations);
			HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel,hostKernel,sizeof(float)*(constKernelDims.x+constKernelDims.y+constKernelDims.z)));
		}

		for (int x=0; x<gaussIterations.x; ++x)
		{
			cudaMultAddFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,Vec<unsigned int>(constKernelDims.x,1,1),
				isColumnMajor);
			incrementBufferNumber();
		}

		for (int y=0; y<gaussIterations.y; ++y)
		{
			cudaMultAddFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,Vec<unsigned int>(1,constKernelDims.y,1),
				constKernelDims.x,isColumnMajor);
			incrementBufferNumber();
		}

		for (int z=0; z<gaussIterations.z; ++z)
		{
			cudaMultAddFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,Vec<unsigned int>(1,1,constKernelDims.z),
				constKernelDims.x+constKernelDims.y,isColumnMajor);
			incrementBufferNumber();
		}
	}

	/*
	*	Sets each pixel to the max value of its neighborhood
	*	Dilates structures
	*/ 
	void maxFilter(Vec<unsigned int> neighborhood, double* kernel=NULL, bool columnMajor=false)
	{
		if (kernel==NULL)
			constKernelOnes();
		else
			setConstKernel(kernel,neighborhood,columnMajor);

		cudaMaxFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,neighborhood,isColumnMajor);
		incrementBufferNumber();
	}

	/*
	*	produce an image that is the maximum value in z for each (x,y)
	*	Images that are copied out of the buffer will have a z size of 1
	*/
	void maximumIntensityProjection()
	{
		cudaMaximumIntensityProjection<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,isColumnMajor);
		imageDims.z = 1;
		updateBlockThread();
		incrementBufferNumber();
	}

	/*
	*	Filters image where each pixel is the mean of its neighborhood 
	*/
	void meanFilter(Vec<unsigned int> neighborhood)
	{
		cudaMeanFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,neighborhood,isColumnMajor);
		incrementBufferNumber();
	}

	/*
	*	Filters image where each pixel is the median of its neighborhood
	*/
	void medianFilter(Vec<unsigned int> neighborhood)
	{
		static dim3 localBlocks = blocks;
		static dim3 localThreads = threads;
		int sharedMemorySize = neighborhood.product()*localThreads.x*localThreads.y*localThreads.z;
		if (sizeof(ImagePixelType)*sharedMemorySize>deviceProp.sharedMemPerBlock)
		{
			float maxThreads = (float)deviceProp.sharedMemPerBlock/(sizeof(ImagePixelType)*neighborhood.product());
			unsigned int threadDim = (unsigned int)pow(maxThreads,1/3.0f);
			localThreads.x = threadDim;
			localThreads.y = threadDim;
			localThreads.z = threadDim;

			localBlocks.x = (unsigned int)ceil((float)imageDims.x/localThreads.x);
			localBlocks.y = (unsigned int)ceil((float)imageDims.y/localThreads.y);
			localBlocks.z = (unsigned int)ceil((float)imageDims.z/localThreads.z);

			sharedMemorySize = neighborhood.product()*localThreads.x*localThreads.y*localThreads.z;
		}

		cudaMedianFilter<<<localBlocks,localThreads,sizeof(ImagePixelType)*sharedMemorySize>>>(getCurrentBuffer(),getNextBuffer(),
			imageDims,neighborhood,isColumnMajor);

		incrementBufferNumber();
	}

	/*
	*	Sets each pixel to the min value of its neighborhood
	*	Erodes structures
	*/ 
	void minFilter(Vec<unsigned int> neighborhood, double* kernel=NULL, bool columnMajor=false)
	{
		if (kernel==NULL)
			constKernelOnes();
		else
			setConstKernel(kernel,neighborhood,columnMajor);

		cudaMinFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,neighborhood,isColumnMajor);
		incrementBufferNumber();
	}

	/*
	*	Sets each pixel by multiplying by the original value and clamping
	*	between minValue and maxValue
	*/
	template<typename FactorType>
	void multiplyImage(FactorType factor)
	{
		cudaMultiplyImage<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,factor,minPixel,maxPixel,isColumnMajor);
		incrementBufferNumber();
	}

	/*
	*	Multiplies this image to the passed in one.
	*/
	void multiplyImageWith(const CudaImageBuffer* image)
	{
		cudaMultiplyTwoImages<<<blocks,threads>>>(getCurrentBuffer(),image->getCurrentBuffer(),getNextBuffer(),imageDims,isColumnMajor);
		incrementBufferNumber();
	}

	/*
	 *	This will calculate the normalized covariance between the two images A and B
	 *	returns (sum over all{(A-mu(A)) X (B-mu(B))}) / (sigma(A)Xsigma(B)
	 *	The images buffers will not change the original data 
	 */
	double normalizedCovariance(CudaImageBuffer* otherImage)
	{
		ImagePixelType* aOrg = this->retrieveImage();
		ImagePixelType* bOrg = otherImage->retrieveImage();

		double aSum1 = 0.0;
		double bSum1 = 0.0;

		this->sumArray(aSum1);
		otherImage->sumArray(bSum1);

		double aMean = aSum1/this->imageDims.product();
		double bMean = bSum1/otherImage->getDimension().product();

		this->addConstant(-aMean);
		otherImage->addConstant(-bMean);

		double aMidSum;
		double bMidSum;

		if (imageDims.z>1)
		{
			this->sumArray(aMidSum);
			otherImage->sumArray(bMidSum);
		}

		this->reserveCurrentBuffer();
		otherImage->reserveCurrentBuffer();

		this->imagePow(2);
		otherImage->imagePow(2);

		double aSum2 = 0.0;
		double bSum2 = 0.0;
		this->sumArray(aSum2);
		otherImage->sumArray(bSum2);

		double aSigma = sqrt(aSum2/this->getDimension().product());
		double bSigma = sqrt(bSum2/otherImage->getDimension().product());

		this->currentBuffer = this->reservedBuffer;
		otherImage->currentBuffer = otherImage->reservedBuffer;

		this->releaseReservedBuffer();
		otherImage->releaseReservedBuffer();

		cudaMultiplyTwoImages<<<blocks,threads>>>(this->getCurrentBuffer(),otherImage->getCurrentBuffer(),this->getNextBuffer(),
			this->imageDims,this->isColumnMajor);
		this->incrementBufferNumber();

		double multSum = 0.0;
		this->sumArray(multSum);

		this->loadImage(aOrg,this->getDimension());
		otherImage->loadImage(bOrg,otherImage->getDimension());

		delete[] aOrg;
		delete[] bOrg;

		double rtn = multSum/(aSigma*bSigma) / this->getDimension().product();

		return rtn;
	}

	/*
	*	Takes a histogram that is on the card and normalizes it
	*	Will generate the original histogram if one doesn't already exist
	*	Use retrieveNormalizedHistogram() to get a host pointer
	*/
	void normalizeHistogram()
	{
		if (isCurrentNormHistogramDevice)
			return;

		if(!isCurrentHistogramDevice)
			createHistogram();

		cudaNormalizeHistogram<<<NUM_BINS,1>>>(histogramDevice,normalizedHistogramDevice,imageDims);
		isCurrentNormHistogramDevice = true;
	}

	void otsuThresholdFilter(float alpha=1.0f)
	{
		ImagePixelType thresh = otsuThresholdValue();
		thresholdFilter(thresh*alpha);
	}

	/*
	*	Raise each pixel to a power
	*/
	void imagePow(int p)
	{
		cudaPow<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,p,isColumnMajor);
		incrementBufferNumber();
	}

	/*
	*	Calculates the total sum of the buffer's data
	*/
	void sumArray(double& sum)
	{
		cudaSumArray<<<sumBlocks,sumThreads,sizeof(double)*sumThreads.x>>>(getCurrentBuffer(),deviceSum,imageDims.product());		
		HANDLE_ERROR(cudaMemcpy(hostSum,deviceSum,sizeof(double)*sumBlocks.x,cudaMemcpyDeviceToHost));

		sum = 0;
		for (unsigned int i=0; i<sumBlocks.x; ++i)
		{
			sum += hostSum[i];
		}
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

		cudaRuduceImage<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,reducedDims,reductions,isColumnMajor);
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
		cudaThresholdImage<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,threshold,minPixel,maxPixel,isColumnMajor);
		incrementBufferNumber();
	}

	void unmix(const CudaImageBuffer* image, Vec<unsigned int> neighborhood)
	{
		cudaUnmixing<<<blocks,threads>>>(getCurrentBuffer(),image->getCudaBuffer(),getNextBuffer(),
			imageDims,neighborhood,minPixel,maxPixel);
		incrementBufferNumber();
	}

	// End Cuda Operators

private:
	CudaImageBuffer();

	void updateBlockThread()
	{
		calcBlockThread(imageDims,deviceProp,blocks,threads);
		calcBlockThread(Vec<unsigned int>((unsigned int)imageDims.product(),1,1),deviceProp,sumBlocks,sumThreads);
		sumBlocks.x = (sumBlocks.x+1) / 2;
	}

	void deviceSetup() 
	{
		HANDLE_ERROR(cudaSetDevice(device));
		HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp,device));
		updateBlockThread();
	}

	void memoryAllocation()
	{
		assert(sizeof(ImagePixelType)*imageDims.product()*NUM_BUFFERS < deviceProp.totalGlobalMem*.7);

		for (int i=0; i<NUM_BUFFERS; ++i)
		{
			HANDLE_ERROR(cudaMalloc((void**)&imageBuffers[i],sizeof(ImagePixelType)*imageDims.product()));
			memoryUsage += sizeof(ImagePixelType)*imageDims.product();
		}

		currentBuffer = -1;
		bufferSize = imageDims.product();

		updateBlockThread();

		sizeSum = sumBlocks.x;
		HANDLE_ERROR(cudaMalloc((void**)&deviceSum,sizeof(double)*sumBlocks.x));
		memoryUsage += sizeof(double)*sumBlocks.x;
		hostSum = new double[sumBlocks.x];

		HANDLE_ERROR(cudaMalloc((void**)&minValuesDevice,sizeof(double)*sumBlocks.x));
		memoryUsage += sizeof(double)*sumBlocks.x;

		histogramHost = new unsigned int[NUM_BINS];
		HANDLE_ERROR(cudaMalloc((void**)&histogramDevice,NUM_BINS*sizeof(unsigned int)));
		memoryUsage += NUM_BINS*sizeof(unsigned int);

		normalizedHistogramHost = new double[NUM_BINS];
		HANDLE_ERROR(cudaMalloc((void**)&normalizedHistogramDevice,NUM_BINS*sizeof(double)));
		memoryUsage += NUM_BINS*sizeof(double);

		minPixel = std::numeric_limits<ImagePixelType>::min();
		maxPixel = std::numeric_limits<ImagePixelType>::max();
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

		deviceSetup();
		memoryAllocation();

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

	void constKernelOnes()
	{
		memset(hostKernel,1,sizeof(float)*MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM);
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel,hostKernel,sizeof(float)*MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM));
	}

	void constKernelZeros()
	{
		memset(hostKernel,1,sizeof(float)*MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM);
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel,hostKernel,sizeof(float)*MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM));
	}

	void setConstKernel(double* kernel, Vec<unsigned int> kernelDims, bool columnMajor)
	{
		memset(hostKernel,0,sizeof(float)*MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM);
		
		Vec<unsigned int> coordinate(0,0,0);
		for (; coordinate.x<kernelDims.x; ++coordinate.x)
		{
			coordinate.y = 0;
			for (; coordinate.y<kernelDims.y; ++coordinate.y)
			{
				coordinate.z = 0;
				for (; coordinate.z<kernelDims.z; ++coordinate.z)
				{
					hostKernel[kernelDims.linearAddressAt(coordinate,false)] = (float)kernel[kernelDims.linearAddressAt(coordinate,columnMajor)];
				}
			}
		}
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel,hostKernel,sizeof(float)*kernelDims.product()));
	}

	void defaults()
	{
		imageDims = UNSET;
		reducedDims = UNSET;
		constKernelDims = UNSET;
		gausKernelSigmas  = Vec<float>(0.0f,0.0f,0.0f);
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
		isCurrentHistogramHost = false;
		isCurrentHistogramDevice = false;
		isCurrentNormHistogramHost = false;
		isCurrentNormHistogramDevice = false;
		deviceSum = NULL;
		minValuesDevice = NULL;
		hostSum = NULL;
		gaussIterations = Vec<int>(0,0,0);
		reservedBuffer = -1;
		memoryUsage = 0;
		isColumnMajor = false;
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

		if (deviceSum!=NULL)
			HANDLE_ERROR(cudaFree(deviceSum));

		if (hostSum!=NULL)
			delete[] hostSum;

		if (minValuesDevice!=NULL)
			HANDLE_ERROR(cudaFree(minValuesDevice));

		memset(hostKernel,0,sizeof(float)*MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM);

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
		return imageBuffers[getNextBufferNum()];
	}

	int getNextBufferNum()
	{
		int nextIndex = currentBuffer;
		do 
		{
			++nextIndex;
			if (nextIndex>=NUM_BUFFERS)
				nextIndex = 0;
		} while (nextIndex==reservedBuffer);
		return nextIndex;
	}

	ImagePixelType* getReservedBuffer()
	{
		if (reservedBuffer<0)
			return NULL;

		return imageBuffers[reservedBuffer];
	}

	void reserveCurrentBuffer()
	{
		reservedBuffer = currentBuffer;
	}

	void releaseReservedBuffer()
	{
		reservedBuffer = -1;
	}

	void incrementBufferNumber()
	{
		cudaThreadSynchronize();
		#ifdef _DEBUG
				gpuErrchk( cudaPeekAtLastError() );
		#endif // _DEBUG

		currentBuffer = getNextBufferNum();
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
	int currentBuffer;
	size_t memoryUsage;
	ImagePixelType* imageBuffers[NUM_BUFFERS];
	ImagePixelType* reducedImageHost;
	ImagePixelType* reducedImageDevice;
	double* minValuesDevice;
	ImagePixelType minPixel;
	ImagePixelType maxPixel;
	unsigned int* histogramHost;
	unsigned int* histogramDevice;
	double* normalizedHistogramHost;
	double* normalizedHistogramDevice;
	bool isCurrentHistogramHost, isCurrentHistogramDevice, isCurrentNormHistogramHost, isCurrentNormHistogramDevice;
	dim3 sumBlocks, sumThreads;
	double* deviceSum;
	double* hostSum;
	int sizeSum;
	Vec<unsigned int> constKernelDims;
	Vec<int> gaussIterations;
	Vec<float> gausKernelSigmas;
	float hostKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
	int reservedBuffer;
	Vec<unsigned int> UNSET;
	bool isColumnMajor;
};
