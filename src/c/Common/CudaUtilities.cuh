#pragma once

#define DEVICE_VEC
#include "Vec.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdexcept>
#include <vector>


#ifdef _DEBUG
#define DEBUG_KERNEL_CHECK() { cudaThreadSynchronize(); gpuErrchk( cudaPeekAtLastError() ); }
#else
#define DEBUG_KERNEL_CHECK() {}
#endif // _DEBUG


static void HandleError( cudaError_t err, const char *file, int line ) 
{
	char* errorMessage = new char[255];
	if (err != cudaSuccess) 
	{
		sprintf_s(errorMessage, 255, "%s in %s at line %d\n", cudaGetErrorString( err ),	file, line );
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		char buff[255];
		sprintf_s(buff, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		throw std::runtime_error(buff);
	}
}

void calcBlockThread(const Vec<size_t>& dims, const cudaDeviceProp &prop, dim3 &blocks, dim3 &threads);

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

Vec<size_t> createGaussianKernel(Vec<float> sigma, float* kernel, int& iterations);

Vec<size_t> createGaussianKernel(Vec<float> sigma, float* kernel, Vec<int>& iterations);
