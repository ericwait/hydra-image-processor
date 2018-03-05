#pragma once

#include "Vec.h"
#include "Defines.h"
#include "ImageDimensions.cuh"
#include "ImageContainer.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>
#include <vector>
#include <cuda_occupancy.h>
#include <functional>

template <typename T, typename U>
int getKernelMaxThreadsSharedMem(T func, U f, int threadLimit = 0)
{
	int blockSizeMax = 0;
	int minGridSize = 0;

	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSizeMax, func, f, threadLimit);

	return blockSizeMax;
}

template <typename T>
int getKernelMaxThreads(T func, int threadLimit=0)
{
    int blockSizeMax = 0;
    int minGridSize = 0; 

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeMax, func, 0, threadLimit);

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

template <class PixelType>
void setUpOutIm(ImageDimensions dims, ImageContainer<PixelType>& imageOut)
{
	if(imageOut.getPtr()==NULL)
	{
		imageOut.getPtr() = new PixelType[dims.getNumElements()];
		imageOut.dimensions = dims;
	}
	else if(imageOut.dimensions!=dims)
	{
		throw std::runtime_error("Image out buffer malformed!");
	}
}

Vec<size_t> GetThreadBlockCoordinate()
{
	Vec<size_t> coordinate;
	coordinate.x = threadIdx.x+blockIdx.x * blockDim.x;
	coordinate.y = threadIdx.y+blockIdx.y * blockDim.y;
	coordinate.z = threadIdx.z+blockIdx.z * blockDim.z;

	return coordinate;
}

size_t memoryAvailable(int device, size_t* totalOut=NULL);
bool checkFreeMemory(size_t needed, int device, bool throws=false);
void calcBlockThread(const Vec<size_t>& dims, const cudaDeviceProp &prop, dim3 &blocks, dim3 &threads, size_t maxThreads=std::numeric_limits<size_t>::max());
Vec<size_t> createGaussianKernel(Vec<float> sigma, float** kernel, Vec<int>& iterations);
Vec<size_t> createLoGKernel(Vec<float> sigma, float** kernelOut, size_t& kernSize);
Vec<size_t> createGaussianKernelFull(Vec<float> sigma, float** kernelOut, Vec<size_t> maxKernelSize = Vec<size_t>(std::numeric_limits<size_t>::max()));
