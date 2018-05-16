#pragma once
#include "ImageChunk.h"
#include "CudaDeviceImages.cuh"
#include "Kernel.cuh"
#include "CudaMultiplySum.cuh"

#include <cuda_runtime.h>


template <class PixelType>
void SeparableMultiplySum(ImageChunk chunk, CudaDeviceImages<PixelType> &deviceImages, Kernel constKernelMem_x, Kernel constKernelMem_y, Kernel constKernelMem_z, const PixelType MIN_VAL, const PixelType MAX_VAL)
{
	cudaMultiplySum<<<chunk.blocks, chunk.threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constKernelMem_x, MIN_VAL, MAX_VAL);
	DEBUG_KERNEL_CHECK();
	deviceImages.incrementBuffer();

	cudaMultiplySum<<<chunk.blocks, chunk.threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constKernelMem_y, MIN_VAL, MAX_VAL);
	DEBUG_KERNEL_CHECK();
	deviceImages.incrementBuffer();

	cudaMultiplySum <<<chunk.blocks, chunk.threads>>>(*(deviceImages.getCurBuffer()), *(deviceImages.getNextBuffer()), constKernelMem_z, MIN_VAL, MAX_VAL);
	DEBUG_KERNEL_CHECK();
	deviceImages.incrementBuffer();
}
