#pragma once
#include "Vec.h"
#include "ImageView.h"

#include <cuda_runtime.h>
#include <vector>

class Kernel
{
public:
	__host__ Kernel(Vec<std::size_t> dimensions, float* values, int curDevice, std::size_t startOffset = 0);
	__host__ Kernel(std::size_t dimensions, float* values, int curDevice, std::size_t startOffset = 0);
	__host__ Kernel(ImageView<float> kernelIn, int curDevice);
	__host__ __device__ Kernel(Kernel& other);
	__host__ __device__ ~Kernel() { init(); }

	__host__ Kernel& operator=(const Kernel& other);
	__host__ void clean();

	__host__ __device__ float* getDevicePtr() { return cudaKernel; }
	__host__ Kernel& getOffsetCopy(Vec<std::size_t> dimensions, std::size_t startOffset = 0);
	__host__ __device__ Vec<std::size_t> getDims() const { return dims; }
	__device__ float operator[](std::size_t idx);
	__device__ float operator()(Vec<std::size_t> coordinate);
	
//private:
	__host__ __device__ Kernel() { init(); }

	__host__ void load(Vec<std::size_t> dimensions, float* values, std::size_t startOffset=0);

	__host__ __device__ void init();
	__host__ void setOnes(float** kernel);

	Vec<std::size_t> dims;

	float* cudaKernel;
	bool cleanUpDevice;

	int device;
};
