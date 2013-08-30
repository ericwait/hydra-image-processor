#include "CudaImageBuffer.h"

template<typename ImagePixelType>
CudaImageBuffer::CudaImageBuffer(Vec<int> dims, int device=0)
{
	defaults();

	this->imageDims = dims;
	this->device = device;

	deviceSetup();
	MemoryAllocation();
}

template<typename ImagePixelType>
CudaImageBuffer::CudaImageBuffer(int x, int y, int z, int device=0)
{
	defaults();

	imageDims = Vec<int>(x,y,z);
	this->device = device;

	deviceSetup();
	MemoryAllocation();
}

template<typename ImagePixelType>
CudaImageBuffer::CudaImageBuffer(int n, int device=0)
{
	defaults();

	imageDims = Vec<int>(n,1,1);
	this->device = device;

	deviceSetup();
	MemoryAllocation();
}

template<typename ImagePixelType>
CudaImageBuffer::~CudaImageBuffer()
{
	clean();
}
// End Constructor / Destructor

//////////////////////////////////////////////////////////////////////////
// Copy Constructors
//////////////////////////////////////////////////////////////////////////

template<typename ImagePixelType>
CudaImageBuffer& CudaImageBuffer::operator=(const CudaImageBuffer<ImagePixelType>& bufferIn)
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

template<typename ImagePixelType>
void CudaImageBuffer::loadImage(ImagePixelType* image)
{
	incrementBufferNumber();
	HANDLE_ERROR(cudaMemcpy(imageBuffers[currentBuffer],image,sizeof(ImagePixelType)*imageDims.product(),cudaMemcpyHostToDevice));
}

template<typename ImagePixelType>
void CudaImageBuffer::retrieveImage(ImagePixelType& imageOut)
{
	if (currentBuffer<0 || currentBuffer>NUM_BUFFERS)
	{
		imageOut = NULL;
		return;
	}

	HANDLE_ERROR(cudaMemcpy(image,imageBuffers[currentBuffer],sizeof(ImagePixelType)*imageDims.product(),cudaMemcpyDeviceToHost));
}

/*
*	Returns a host pointer to the histogram data
*	This is destroyed when this' destructor is called
*	Will call the needed histogram creation methods if not all ready
*/
template<typename ImagePixelType>
unsigned int* CudaImageBuffer::retrieveHistogram(int& returnSize)
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
template<typename ImagePixelType>
double* CudaImageBuffer::retrieveNormalizedHistogram(int& returnSize)
{
	if (normalizedHistogramDevice==NULL)
		normalizeHistogram();

	HANDLE_ERROR(cudaMemcpy(normalizedHistogramHost,normalizedHistogramDevice,sizeof(double)*NUM_BINS,cudaMemcpyDeviceToHost));
	returnSize = NUM_BINS;

	return normalizedHistogramHost;
}

template<typename ImagePixelType>
ImagePixelType* CudaImageBuffer::retrieveReducedImage(Vec<int>& reducedDims)
{
	reducedDims = this->reducedDims;

	if (reducedImageDevice!=NULL)
	{
		HANDLE_ERROR(cudaMemcpy(reducedImageHost,reducedImageDevice,sizeof(ImagePixelType)*reducedDims.product(),cudaMemcpyDeviceToHost));
	}

	return reducedImageHost;
}

template<typename ImagePixelType>
void CudaImageBuffer::getROI(Vec<int> startPos, Vec<int> newSize)
{
	// TODO: stub
	// 		cudaGetROI<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,startPos,newSize);
	// 		incrementBufferNumber();
}
// End Setters / Getters

//////////////////////////////////////////////////////////////////////////
// Cuda Operators (Alphabetical order)
//////////////////////////////////////////////////////////////////////////

/*
*	Adds this image to the passed in one.  You can apply a factor
*	which is multiplied to the passed in image prior to adding
*/
template<typename ImagePixelType>
void CudaImageBuffer::addImageTo(const CudaImageBuffer* image, double factor)
{
	cudaAddTwoImagesWithFactor<<<blocks,threads>>>(getCurrentBuffer(),image->getCurrentBuffer(),getNextBuffer(),imageDims,factor);
	incrementBufferNumber();
}

/*
*	New pixel values will be a*x^2 + b*x + c where x is the original
*	pixel value.  This new value will be clamped between the min and
*	max values.
*/
template<typename ThresholdType>
void CudaImageBuffer::applyPolyTransformation(ThresholdType a, ThresholdType b, ThresholdType c, ImagePixelType minValue, ImagePixelType maxValue)
{
	cudaPolyTransferFuncImage<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,a,b,c,maxValue,minValue);
	incrementBufferNumber();
}

/*
*	This will find the min and max values of the image
*/ 
template<typename ImagePixelType>
void CudaImageBuffer::calculateMinMax(ImagePixelType& minValue, ImagePixelType& maxValue)
{
	double* maxValuesHost = new double[(blocks.x+1)/2];
	double* minValuesHost = new double[(blocks.x+1)/2];
	double* maxValuesDevice;
	double* minValuesDevice;

	HANDLE_ERROR(cudaMalloc((void**)&maxValuesDevice,sizeof(double)*(blocks.x+1)/2));
	HANDLE_ERROR(cudaMalloc((void**)&minValuesDevice,sizeof(double)*(blocks.x+1)/2));

	cudaFindMinMax<<<blocks.x,threads.x,2*sizeof(double)*threads.x>>>(getCurrentBuffer(),minValuesDevice,maxValuesDevice,
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
template<typename ImagePixelType>
void CudaImageBuffer::createHistogram()
{
	histogramHost = new unsigned int[NUM_BINS];
	HANDLE_ERROR(cudaMalloc((void**)&histogramDevice,NUM_BINS*sizeof(unsigned int)));

	memset(histogramHost,0,NUM_BINS*sizeof(unsigned int));
	HANDLE_ERROR(cudaMemset(histogramDevice,0,NUM_BINS*sizeof(unsigned int)));

	cudaHistogramCreate<<<deviceProp.multiProcessorCount*2,NUM_BINS,sizeof(unsigned int)*NUM_BINS>>>
		(getCurrentBuffer(),histogramDevice,imageDims);
}

/*
*	Will smooth the image using the given sigmas for each dimension
*/ 
template<typename ImagePixelType>
void CudaImageBuffer::gaussianFilter(Vec<double> sigmas)
{
}

/*
*	Sets each pixel to the max value of its neighborhood
*	Dilates structures
*/ 
template<typename ImagePixelType>
void CudaImageBuffer::maxFilter(Vec<int> neighborhood)
{
	cudaMaxFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,neighborhood);
	incrementBufferNumber();
}

/*
*	produce an image that is the maximum value in z for each (x,y)
*/
template<typename ImagePixelType>
void CudaImageBuffer::maximumIntensityProjection()
{
	cudaMaximumIntensityProjection<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims);
	incrementBufferNumber();
}

/*
*	Filters image where each pixel is the mean of its neighborhood 
*/
template<typename ImagePixelType>
void CudaImageBuffer::meanFilter(Vec<int> neighborhood)
{
	cudaMeanFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,neighborhood);
	incrementBufferNumber();
}

/*
*	Filters image where each pixel is the median of its neighborhood
*/
template<typename ImagePixelType>
void CudaImageBuffer::medianFilter(Vec<int> neighborhood)
{
	cudaMedianFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,neighborhood);
	incrementBufferNumber();
}

/*
*	Sets each pixel to the min value of its neighborhood
*	Erodes structures
*/ 
template<typename ImagePixelType>
void CudaImageBuffer::minFilter(Vec<int> neighborhood)
{
	cudaMinFilter<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,neighborhood);
	incrementBufferNumber();
}

/*
*	Sets each pixel by multiplying by the orignal value and clamping
*	between minValue and maxValue
*/
template<typename FactorType>
void CudaImageBuffer::multiplyImage(FactorType factor, ImagePixelType minValue, ImagePixelType maxValue)
{
	cudaMultiplyImage<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,factor,minValue,maxValue);
	incrementBufferNumber();
}

/*
*	Takes a histogram that is on the card and normalizes it
*	Will generate the original histogram if one doesn't already exist
*	Use retrieveNormalizedHistogram() to get a host pointer
*/
template<typename ImagePixelType>
void CudaImageBuffer::normalizeHistogram()
{
	if(histogramDevice==NULL)
		createHistogram();

	normalizedHistogramHost = new double[NUM_BINS];
	HANDLE_ERROR(cudaMalloc((void**)&normalizedHistogramDevice,NUM_BINS*sizeof(double)));
	cudaNormalizeHistogram<<<NUM_BINS,1>>>(histogramDevice,normalizedHistogramDevice,imageDims);
}

/*
*	Calculates the total sum of the buffer's data
*/
template<typename Sumtype>
void CudaImageBuffer::reduceArray(Sumtype& sum)
{
}

/*
*	Will reduce the size of the image by the factors passed in
*/
template<typename ImagePixelType>
void CudaImageBuffer::reduceImage(Vec<double> reductions)
{
	reducedDims = Vec<int>(
		imageDims.x/reductions.x,
		imageDims.y/reductions.y,
		imageDims.z/reductions.z);

	cudaRuduceImage<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,reducedDims,reductions);
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
void CudaImageBuffer::thresholdFilter(ThresholdType threshold)
{
	cudaThresholdImage<<<blocks,threads>>>(getCurrentBuffer(),getNextBuffer(),imageDims,threshold);
	incrementBufferNumber();
}

// End Cuda Operators

template<typename ImagePixelType>
void CudaImageBuffer::deviceSetup() 
{
	HANDLE_ERROR(cudaSetDevice(device));
	HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp,device));
	calcBlockThread(imageDims,deviceProp,blocks,threads);
}

template<typename ImagePixelType>
void CudaImageBuffer::MemoryAllocation()
{
	assert(sizeof(ImagePixelType)*imageDims.product()*NUM_BUFFERS < deviceProp.totalGlobalMem*.6);

	for (int i=0; i<NUM_BUFFERS; ++i)
	{
		HANDLE_ERROR(cudaMalloc((void**)&imageBuffers[i],sizeof(ImagePixelType)*imageDims.product()));
	}

	currentBuffer = -1;
}

template<typename ImagePixelType>
void CudaImageBuffer::copy(const CudaImageBuffer<ImagePixelType>& bufferIn)
{
	imageDims = bufferIn.getDimension();
	device = bufferIn.getDevice();

	defaults();
	MemoryAllocation();

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

template<typename ImagePixelType>
void CudaImageBuffer::defaults()
{
	imageDims = Vec<int>(-1,-1,-1);
	reducedDims = Vec<int>(-1,-1,-1);
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

template<typename ImagePixelType>
void CudaImageBuffer::clean() 
{
	for (int i=0; i<NUM_BUFFERS; ++i)
	{
		if (imageBuffers[i]!=NULL)
			HANDLE_ERROR(cudaFree(imageBuffers[0]));
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

template<typename ImagePixelType>
ImagePixelType* CudaImageBuffer::getCurrentBuffer() const 
{
	if (currentBuffer<0 || currentBuffer>NUM_BUFFERS)
		return NULL;

	return imageBuffers[currentBuffer];
}

template<typename ImagePixelType>
ImagePixelType* CudaImageBuffer::getNextBuffer()
{
	int nextIndex = currentBuffer +1;
	if (nextIndex>=NUM_BUFFERS)
		nextIndex = 0;

	return imageBuffers[nextIndex];
}

template<typename ImagePixelType>
void CudaImageBuffer::incrementBufferNumber()
{
	++currentBuffer;

	if (currentBuffer>=NUM_BUFFERS)
		currentBuffer = 0;
}
