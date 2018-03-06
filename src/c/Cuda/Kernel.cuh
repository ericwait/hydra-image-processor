#pragma once
#include "Vec.h"

#include <cuda_runtime.h>
#include <vector>

#ifndef CUDA_CONST_KERNEL
#define CUDA_CONST_KERNEL
__constant__ float cudaConstKernel[MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM];
#endif

class Kernel
{
public:
	Kernel(Vec<size_t> dimensions);
	Kernel(Vec<size_t> dimensions, float* values);

	~Kernel() { kernel.clear(); }

	Vec<size_t> getDimensions() const { return dims; }
	const float* getConstPtr() const { return kernel.data(); }
	
private:
	Kernel();
	void setOnes();
	std::vector<float> kernel;
	Vec<size_t> dims;
};
