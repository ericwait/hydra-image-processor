#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#define DEVICE_VEC
#include "Vec.h"
#undef DEVICE_VEC
#include "Defines.h"
#include "CudaImageContainerClean.cuh"
#include <vector>
#include "ImageChunk.cuh"

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
	*	New pixel values will be a*x^2 + b*x + c where x is the original
	*	pixel value.  This new value will be clamped between the min and
	*	max values.
	*/
	DevicePixelType* applyPolyTransformation(const DevicePixelType* imageIn, Vec<size_t> dims, double a, double b, double c,
		DevicePixelType minValue, DevicePixelType maxValue, DevicePixelType** imageOut=NULL);

	/*
	*	Contrast Enhancement will run the Michel High Pass Filter and then a mean filter
	*	Pass in the sigmas that will be used for the Gaussian filter to subtract off and the mean neighborhood dimensions
	*/
	DevicePixelType* contrastEnhancement(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<float> sigmas,
		Vec<size_t> medianNeighborhood, DevicePixelType** imageOut=NULL);

	/*
	*	Filters image where each pixel is the mean of its neighborhood
	*  If imageOut is null, then a new image pointer will be created and returned.
	*  In either case the caller must clean up the the return image correctly
	*/
	DevicePixelType* meanFilter(const DevicePixelType* imageIn, Vec<size_t> dims, Vec<size_t> neighborhood, DevicePixelType** imageOut=NULL);

	/*
	*	This will calculate the normalized covariance between the two images A and B
	*	returns (sum over all{(A-mu(A)) X (B-mu(B))}) / (sigma(A)Xsigma(B)
	*	The images buffers will not change the original data 
	*/
	double normalizedCovariance(const DevicePixelType* imageIn1, const DevicePixelType* imageIn2, Vec<size_t> dims);

	DevicePixelType* otsuThresholdFilter(const DevicePixelType* imageIn, Vec<size_t> dims, double alpha=1.0, DevicePixelType** imageOut=NULL);

	double otsuThresholdValue(const DevicePixelType* imageIn, Vec<size_t> dims);

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

	//////////////////////////////////////////////////////////////////////////
	// Private Member Variables
	//////////////////////////////////////////////////////////////////////////

	// Device that this buffer will utilize
	int device;
	cudaDeviceProp deviceProp;

	// This is the size that the input and output images will be
	Vec<size_t> orgImageDims;

	Vec<size_t> maxDeviceDims;
};
