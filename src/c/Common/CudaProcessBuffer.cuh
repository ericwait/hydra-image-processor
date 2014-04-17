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
	*	This will calculate the normalized covariance between the two images A and B
	*	returns (sum over all{(A-mu(A)) X (B-mu(B))}) / (sigma(A)Xsigma(B)
	*	The images buffers will not change the original data 
	*/
	double normalizedCovariance(const DevicePixelType* imageIn1, const DevicePixelType* imageIn2, Vec<size_t> dims);


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
