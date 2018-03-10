#pragma once
#include "Vec.h"
#include "ImageContainer.h"

#include <cuda_runtime.h>
#include <vector>

class Kernel
{
public:
	__host__ Kernel(Vec<size_t> dimensions, float* values, int deviceIn);
	__host__ Kernel(ImageContainer<float> kernelIn, int deviceIn);
	__host__ __device__ Kernel(const Kernel& other);

	__host__ __device__ ~Kernel() {}
	__host__ void clean();

	__host__ __device__ Vec<size_t> getDimensions() const { return dims; }
	__device__ float operator[](size_t idx);
	__device__ float operator()(Vec<size_t> coordinate);
	
private:
	__host__ __device__ Kernel();

	__host__ void load(Vec<size_t> dimensions, float* values, int deviceIn);

	__host__ void init();
	__host__ void setOnes();

	Vec<size_t> dims;

	float* kernel;
	bool cleanUpHost;

	float* cudaKernel;
	bool cleanUpDevice;

	int device;
};
