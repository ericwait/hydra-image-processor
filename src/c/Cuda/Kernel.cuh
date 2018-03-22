#pragma once
#include "Vec.h"
#include "ImageContainer.h"

#include <cuda_runtime.h>
#include <vector>

class Kernel
{
public:
	__host__ Kernel(Vec<size_t> dimensions, float* values, int curDevice, size_t startOffset = 0);
	__host__ Kernel(size_t dimensions, float* values, int curDevice, size_t startOffset = 0);
	__host__ Kernel(ImageContainer<float> kernelIn, int curDevice);
	__host__ __device__ Kernel(Kernel& other);
	__host__ __device__ ~Kernel() { init(); }

	__host__ Kernel& operator=(const Kernel& other);
	__host__ void clean();

	__host__ __device__ float* getDevicePtr() { return cudaKernel; }
	__host__ Kernel& getOffsetCopy(Vec<size_t> dimensions, size_t startOffset = 0);
	__host__ __device__ Vec<size_t> getDims() const { return dims; }
	__device__ float operator[](size_t idx);
	__device__ float operator()(Vec<size_t> coordinate);
	
//private:
	__host__ __device__ Kernel() { init(); }

	__host__ void load(Vec<size_t> dimensions, float* values, size_t startOffset=0);

	__host__ __device__ void init();
	__host__ void setOnes(float** kernel);

	Vec<size_t> dims;

	float* cudaKernel;
	bool cleanUpDevice;

	int device;
};
