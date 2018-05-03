#include "Kernel.cuh"
#include "CudaUtilities.h"

__constant__ float CUDA_KERNEL[CONST_KERNEL_NUM_EL];

__host__ Kernel::Kernel(Vec<size_t> dimensions, float* values, int curDevice, size_t startOffset/* = 0*/)
{
	device = curDevice;
	HANDLE_ERROR(cudaSetDevice(curDevice));
	load(dimensions, values, startOffset);
}

__host__ __device__ Kernel::Kernel(Kernel& other)
{
	init();

	device = other.device;
	dims = other.getDims();
	cudaKernel = other.getDevicePtr();
	cleanUpDevice = false;
}

__host__ Kernel::Kernel(ImageContainer<float> kernelIn, int curDevice)
{
	HANDLE_ERROR(cudaSetDevice(curDevice));
	load(kernelIn.getSpatialDims(), kernelIn.getPtr());
}

__host__ Kernel::Kernel(size_t dimensions, float* values, int curDevice, size_t startOffset /*= 0*/)
{
	HANDLE_ERROR(cudaSetDevice(curDevice));
	load(Vec<size_t>(dimensions, 1, 1), values, startOffset);
}

__device__ float Kernel::operator[](size_t idx)
{
	float val = cudaKernel[idx];
	return val;
}

__device__ float Kernel::operator()(Vec<size_t> coordinate)
{
	size_t idx = dims.linearAddressAt(coordinate);
	float val = cudaKernel[idx];
	return val;
}

__host__ Kernel& Kernel::operator=(const Kernel& other)
{
	dims = other.dims;
	cudaKernel = other.cudaKernel;
	cleanUpDevice = other.cleanUpDevice;

	return *this;
}

__host__ void Kernel::load(Vec<size_t> dimensions, float* values, size_t startOffset/* = 0*/)
{
	init();

	dims = dimensions;
	float* kernel = NULL;
	bool cleanKernel = false;
	if (values == NULL)
	{
		setOnes(&kernel);
		cleanKernel = true;
	}
	else
	{
		kernel = values;
	}

	if (dimensions.product() + startOffset < CONST_KERNEL_NUM_EL)
	{
		HANDLE_ERROR(cudaGetSymbolAddress((void**)&cudaKernel, CUDA_KERNEL));
		cudaKernel += startOffset;
		HANDLE_ERROR(cudaMemcpyToSymbol(CUDA_KERNEL, kernel, sizeof(float)*dims.product()));
	}
	else
	{
		HANDLE_ERROR(cudaMalloc(&cudaKernel, sizeof(float)*dims.product()));
		HANDLE_ERROR(cudaMemcpy(cudaKernel, values, sizeof(float)*dims.product(),cudaMemcpyHostToDevice));
		cleanUpDevice = true;
	}

	if (cleanKernel)
		delete[] kernel;
}

__host__ void Kernel::clean()
{
	if (cleanUpDevice)
	{
		cudaFree(cudaKernel);
		cleanUpDevice = false;
	}

	init();
}

__host__ __device__ void Kernel::init()
{
	dims = Vec<size_t>(0);
	cudaKernel = NULL;
	cleanUpDevice = false;
}

__host__ void Kernel::setOnes(float** kernel)
{
	*kernel = new float[dims.product()];
	for (int i = 0; i < dims.product(); ++i)
		(*kernel)[i] = 1.0f;
}

__host__ Kernel& Kernel::getOffsetCopy(Vec<size_t> dimensions, size_t startOffset /*= 0*/)
{
	Kernel* kernOut = new Kernel();
	kernOut->init();

	if (dims.product() < startOffset + dimensions.product())
		std::runtime_error("Trying to make a Kernel that access outside of the original memory space!");

	kernOut->dims = dimensions;
	kernOut->cudaKernel = cudaKernel + startOffset;

	return *kernOut;
}
