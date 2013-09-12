#pragma once

#include "Vec.h"
#include "cuda_runtime.h"
#include <stdio.h>

static void HandleError( cudaError_t err, const char *file, int line ) 
{
	char* errorMessage = new char[255];
	if (err != cudaSuccess) 
	{
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	}
}

void calcBlockThread(const Vec<unsigned int>& dims, const cudaDeviceProp &prop, dim3 &blocks, dim3 &threads)
{
	if (dims.z==1)
	{
		if (dims.y==1)
		{
			if (dims.x<(unsigned int)prop.maxThreadsPerBlock)
			{
				threads.x = dims.x;
				threads.y = 1;
				threads.z = 1;
				blocks.x = 1;
				blocks.y = 1;
				blocks.z = 1;
			} 
			else
			{
				threads.x = prop.maxThreadsPerBlock;
				threads.y = 1;
				threads.z = 1;
				blocks.x = (int)ceil((float)dims.x/prop.maxThreadsPerBlock);
				blocks.y = 1;
				blocks.z = 1;
			}
		}
		else
		{
			if (dims.x*dims.y<(unsigned int)prop.maxThreadsPerBlock)
			{
				threads.x = dims.x;
				threads.y = dims.y;
				threads.z = 1;
				blocks.x = 1;
				blocks.y = 1;
				blocks.z = 1;
			} 
			else
			{
				int dim = (int)sqrt((double)prop.maxThreadsPerBlock*prop.maxThreadsPerBlock);
				threads.x = dim;
				threads.y = dim;
				threads.z = 1;
				blocks.x = (int)ceil((float)dims.x/dim);
				blocks.y = (int)ceil((float)dims.y/dim);
				blocks.z = 1;
			}
		}
	}
	else
	{
		if(dims.x*dims.y*dims.z < (unsigned int)prop.maxThreadsPerBlock)
		{
			blocks.x = 1;
			blocks.y = 1;
			blocks.z = 1;
			threads.x = dims.x;
			threads.y = dims.y;
			threads.z = dims.z;
		}
		else
		{
			int dim = (int)pow((float)prop.maxThreadsPerBlock,1/3.0f);
			int extra = (prop.maxThreadsPerBlock-dim*dim*dim)/(dim*dim);
			threads.x = dim + extra;
			threads.y = dim;
			threads.z = dim;

			blocks.x = (unsigned int)ceil((float)dims.x/threads.x);
			blocks.y = (unsigned int)ceil((float)dims.y/threads.y);
			blocks.z = (unsigned int)ceil((float)dims.z/threads.z);
		}
	}
}

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
