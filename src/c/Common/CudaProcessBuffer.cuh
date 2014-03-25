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
#include "ImageChunk.cuh"

std::vector<ImageChunk> calculateBuffers(Vec<size_t> imageDims, int numBuffersNeeded, size_t memAvailable, const cudaDeviceProp& prop,
										 Vec<size_t> kernalDims=Vec<size_t>(0,0,0));

std::vector<ImageChunk> calculateChunking(Vec<size_t> orgImageDims, Vec<size_t> deviceDims, const cudaDeviceProp& prop, Vec<size_t> kernalDims=Vec<size_t>(0,0,0));

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

	/*
	*	Add a constant to all pixel values
	*/
	DevicePixelType* addConstant(const DevicePixelType* imageIn, Vec<size_t> dims, double additive, DevicePixelType** imageOut=NULL);

	/*
	*	Adds this image to the passed in one.  You can apply a factor
	*	which is multiplied to the passed in image prior to adding
	*/
	DevicePixelType* addImageWith(const DevicePixelType* imageIn1, const DevicePixelType* imageIn2, Vec<size_t> dims, double additive,
		DevicePixelType** imageOut=NULL);

	/*
	*	New pixel values will be a*x^2 + b*x + c where x is the original
	*	pixel value.  This new value will be clamped between the min and
	*	max values.
	*/
	DevicePixelType* applyPolyTransformation(const DevicePixelType* imageIn, Vec<size_t> dims, double a, double b, double c,
		DevicePixelType minValue, DevicePixelType maxValue, DevicePixelType** imageOut=NULL);

	/*
	*	This will find the min and max values of the image
	*/ 
	void calculateMinMax(double& minValue, double& maxValue);

	/*
	*	Contrast Enhancement will run the Michel High Pass Filter and then a mean filter
	*	Pass in the sigmas that will be used for the Gaussian filter to subtract off and the mean neighborhood dimensions
	*/
	DevicePixelType* contrastEnhancement(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<float> sigmas,
		Vec<size_t> medianNeighborhood, DevicePixelType** imageOut=NULL);

	/*
	*	Creates Histogram on the card using the #define NUM_BINS
	*/
	size_t* createHistogram(const DevicePixelType* imageIn, Vec<size_t> dims, int& arraySize);

	/*
	*	Will smooth the image using the given sigmas for each dimension
	*/ 
	DevicePixelType* gaussianFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<float> sigmas, DevicePixelType** imageOut=NULL);

	/*
	*	Mask will mask out the pixels of this buffer given an image and a threshold.
	*	The threshold it to allow the input image to be gray scale instead of logical.
	*	This buffer will get zeroed out where the imageMask is less than or equal to the threshold.
	*/
	void mask(const DevicePixelType* imageMask, DevicePixelType threshold=1);

	/*
	*	Sets each pixel to the max value of its neighborhood
	*	Dilates structures
	*/ 
	void maxFilter(Vec<size_t> neighborhood, double* kernel=NULL);

	/*
	*	produce an image that is the maximum value in z for each (x,y)
	*	Images that are copied out of the buffer will have a z size of 1
	*/
	void maximumIntensityProjection();

	/*
	*	Filters image where each pixel is the mean of its neighborhood
	*  If imageOut is null, then a new image pointer will be created and returned.
	*  In either case the caller must clean up the the return image correctly
	*/
	DevicePixelType* meanFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, DevicePixelType** imageOut=NULL);

	/*
	*	Filters image where each pixel is the median of its neighborhood
	*/
	DevicePixelType* medianFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, DevicePixelType** imageOut=NULL);

	/*
	*	Sets each pixel to the min value of its neighborhood
	*	Erodes structures
	*/ 
	void minFilter(Vec<size_t> neighborhood, double* kernel=NULL);

	void morphClosure(Vec<size_t> neighborhood, double* kernel=NULL);

	void morphOpening(Vec<size_t> neighborhood, double* kernel=NULL);

	void multiplyImage(double factor);

	/*
	*	Multiplies this image to the passed in one.
	*/
	void multiplyImageWith(const DevicePixelType* image);

	/*
	*	This will calculate the normalized covariance between the two images A and B
	*	returns (sum over all{(A-mu(A)) X (B-mu(B))}) / (sigma(A)Xsigma(B)
	*	The images buffers will not change the original data 
	*/
	double normalizedCovariance(DevicePixelType* otherImage);

	/*
	*	Takes a histogram that is on the card and normalizes it
	*/
	double* normalizeHistogram(const DevicePixelType* imageIn, Vec<size_t> dims, int& arraySize);

	DevicePixelType* otsuThresholdFilter(const DevicePixelType* imageIn, Vec<size_t> dims, double alpha=1.0, DevicePixelType** imageOut=NULL);

	double otsuThresholdValue(const DevicePixelType* imageIn, Vec<size_t> dims);

	/*
	*	Raise each pixel to a power
	*/
	void imagePow(int p);

	/*
	*	Calculates the total sum of the buffer's data
	*/
	void sumArray(double& sum);

	/*
	*	Will reduce the size of the image by the factors passed in
	*/
	DevicePixelType* CudaProcessBuffer::reduceImage(const DevicePixelType* image, Vec<size_t> dims, Vec<size_t> reductions,
		Vec<size_t>& reducedDims, DevicePixelType** imageOut=NULL);

	/*
	*	This creates a image with values of 0 where the pixels fall below
	*	the threshold and 1 where equal or greater than the threshold
	*	
	*	If you want a viewable image after this, you may want to use the
	*	multiplyImage routine to turn the 1 values to the max values of
	*	the type
	*/
	DevicePixelType* thresholdFilter(const DevicePixelType* image, Vec<size_t> dims, DevicePixelType threshold, DevicePixelType** imageOut=NULL);

	void unmix(const DevicePixelType* image, Vec<size_t> neighborhood);

private:

	//////////////////////////////////////////////////////////////////////////
	// Private Helper Methods
	//////////////////////////////////////////////////////////////////////////
	void deviceSetup();

	void defaults();

	void setMaxDeviceDims(std::vector<ImageChunk> &chunks, Vec<size_t> &maxDeviceDims);

	DevicePixelType* setUpOutIm(Vec<size_t> dims, DevicePixelType** imageOut);

	//////////////////////////////////////////////////////////////////////////
	// Private Member Variables
	//////////////////////////////////////////////////////////////////////////

	// Device that this buffer will utilize
	int device;
	cudaDeviceProp deviceProp;

	// This is the size that the input and output images will be
	Vec<size_t> orgImageDims;

	// This is the maximum size that we are allowing a constant kernel to exit
	// on the device
	float hostKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];

	Vec<size_t> maxDeviceDims;
};
