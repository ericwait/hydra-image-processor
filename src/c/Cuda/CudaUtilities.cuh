#pragma once

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdexcept>
#include <vector>
#include "Vec.h"
#include "Defines.h"

#include <cuda_occupancy.h>

template <typename T>
int getKernelMaxThreads(T func, int threadLimit=0)
{
    int blockSizeMax = 0;
    int minGridSize = 0; 

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeMax, func, threadLimit, 0);

    return blockSizeMax;
}


#ifdef _DEBUG
#define DEBUG_KERNEL_CHECK() { cudaThreadSynchronize(); gpuErrchk( cudaPeekAtLastError() ); }
#else
#define DEBUG_KERNEL_CHECK() {}
#endif // _DEBUG

static void HandleError( cudaError_t err, const char *file, int line ) 
{
	if (err != cudaSuccess) 
	{
		char* errorMessage = new char[255];
		sprintf_s(errorMessage, 255, "%s in %s at line %d\n", cudaGetErrorString( err ),	file, line );
		throw std::runtime_error(errorMessage);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		char buff[255];
		sprintf_s(buff, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		throw std::runtime_error(buff);
	}
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

struct Lock 
{
	int* mutex;

	Lock()
	{
#if __CUDA_ARCH__ >= 200
		int state = 0;
		HANDLE_ERROR(cudaMalloc((void**)&mutex,sizeof(int)));
		HANDLE_ERROR(cudaMemcpy(mutex,&state,sizeof(int),cudaMemcpyHostToDevice));
#endif
	}

	~Lock()
	{
#if __CUDA_ARCH__ >= 200
		cudaFree(mutex);
#endif
	}

	__device__ void lock()
	{
#if __CUDA_ARCH__ >= 200
		while(atomicCAS(mutex,0,1) != 0);
#endif
	}

	__device__ void unlock()
	{
#if __CUDA_ARCH__ >= 200
		atomicExch(mutex,0);
#endif
	}
};

template <class PixelType>
PixelType* setUpOutIm(Vec<size_t> dims, PixelType** imageOut)
{
	PixelType* imOut;
	if (imageOut==NULL)
		imOut = new PixelType[dims.product()];
	else
		imOut = *imageOut;

	return imOut;
}

size_t memoryAvailable(int device, size_t* totalOut=NULL);
bool checkFreeMemory(size_t needed, int device, bool throws=false);
void calcBlockThread(const Vec<size_t>& dims, const cudaDeviceProp &prop, dim3 &blocks, dim3 &threads,
					 size_t maxThreads=std::numeric_limits<size_t>::max());
Vec<size_t> createGaussianKernel(Vec<float> sigma, float** kernel, Vec<int>& iterations);
Vec<size_t> createGaussianKernelFull(Vec<float> sigma, float** kernelOut, Vec<size_t> maxKernelSize = Vec<size_t>(std::numeric_limits<size_t>::max()));
