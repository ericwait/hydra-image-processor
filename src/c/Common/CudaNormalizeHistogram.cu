#include "CudaKernels.cuh"

__global__ void cudaNormalizeHistogram(size_t* histogram, double* normHistogram, Vec<size_t> imageDims)
{
	int x = blockIdx.x;
	normHistogram[x] = (double)(histogram[x]) / (imageDims.x*imageDims.y*imageDims.z);
}

