#include "Kernel.cuh"
#include "CudaUtilities.h"

#ifndef CUDA_CONST_KERNEL
#define CUDA_CONST_KERNEL
__constant__ float cudaConstKernel[CONST_KERNEL_NUM_EL];
#endif

__host__ Kernel::Kernel(Vec<size_t> dimensions, float* values)
{
	load(dimensions, values);
}

__device__ Kernel::Kernel(const Kernel& other)
{
	dims = other.dims;
	kernel = other.kernel;
	cleanUpHost = other.cleanUpHost;
	cudaKernel = other.cudaKernel;
	cleanUpDevice = other.cleanUpDevice;
}


__host__ Kernel::Kernel(ImageContainer<float> kernelIn)
{
	load(kernelIn.getSpatialDims(), kernelIn.getPtr());
}


__host__ void Kernel::load(Vec<size_t> dimensions, float* values)
{
	init();

	dims = dimensions;

	if (values == NULL)
	{
		setOnes();
	}
	else
	{
		kernel = values;
	}

	if (dimensions.product() < CONST_KERNEL_NUM_EL)
	{
		HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, kernel, sizeof(float)*dims.product()));
		cudaKernel = cudaConstKernel;
	}
	else
	{
		HANDLE_ERROR(cudaMalloc(&cudaKernel, sizeof(float)*dims.product()));
		cleanUpDevice = true;
	}
}

__host__ void Kernel::clean()
{
	if (cleanUpHost)
	{
		delete[] kernel;
		cleanUpHost = false;
	}

	if (cleanUpDevice)
	{
		cudaFree(cudaKernel);
		cleanUpDevice = false;
	}

	init();
}


__device__ float Kernel::operator[](size_t idx)
{
	return cudaKernel[idx];
}


__device__ float Kernel::operator()(Vec<size_t> coordinate)
{
	return cudaKernel[dims.linearAddressAt(coordinate)];
}


__host__ void Kernel::init()
{
	dims = Vec<size_t>(0);
	kernel = NULL;
	cleanUpHost = false;
	cudaKernel = NULL;
	cleanUpDevice = false;
}


__host__ void Kernel::setOnes()
{
	kernel = new float[dims.product()];
	for (int i = 0; i < dims.product(); ++i)
		kernel[i] = 1.0f;

	cleanUpHost = true;
}


