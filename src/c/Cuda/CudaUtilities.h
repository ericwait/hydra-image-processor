#pragma once

#include "Vec.h"
#include "Defines.h"
#include "ImageView.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>
#include <vector>
#include <cuda_occupancy.h>

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

static void HandleError( cudaError_t err, const char *file, int line ) 
{
	if (err != cudaSuccess) 
	{
		char errorMessage[255];
		sprintf(errorMessage, "%s in %s at line %d\n", cudaGetErrorString( err ),	file, line );
		throw std::runtime_error(errorMessage);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#ifdef _DEBUG
#define DEBUG_KERNEL_CHECK() { cudaThreadSynchronize(); HandleError( cudaPeekAtLastError(), __FILE__, __LINE__ ); }
#else
#define DEBUG_KERNEL_CHECK() {}
#endif // _DEBUG

void calcBlockThread(const Vec<std::size_t>& dims, std::size_t maxThreads, dim3 &blocks, dim3 &threads);
