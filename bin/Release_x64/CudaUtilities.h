#ifndef CUDA_UTILITIES_H
#define CUDA_UTILITIES_H

#include "Vec.h"
#include "cuda_runtime.h"
#include <stdio.h>

#define NUM_BINS 255

static void HandleError( cudaError_t err, const char *file, int line ) 
{
	if (err != cudaSuccess) 
	{
		char* errorMessage = new char[255];
		sprintf(errorMessage, "%s in %s at line %d\n", cudaGetErrorString( err ),	file, line );
		throw(errorMessage);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
		{ 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{   -1, -1 }
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
	return nGpuArchCoresPerSM[7].Cores;
}
// end of GPU Architecture definitions

void calcBlockThread(const Vec<int>& dims, const cudaDeviceProp &prop, dim3 &blocks, dim3 &threads);

struct Lock 
{
	int* mutex;

	Lock()
	{
		int state = 0;
		HANDLE_ERROR(cudaMalloc((void**)&mutex,sizeof(int)));
		HANDLE_ERROR(cudaMemcpy(mutex,&state,sizeof(int),cudaMemcpyHostToDevice));
	}

	~Lock()
	{
		cudaFree(mutex);
	}

	__device__ void lock()
	{
		while(atomicCAS(mutex,0,1) != 0);
	}

	__device__ void unlock()
	{
		atomicExch(mutex,0);
	}
};
#endif