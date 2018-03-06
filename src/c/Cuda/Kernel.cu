#include "Kernel.cuh"
#include "CudaUtilities.cuh"

Kernel::Kernel(Vec<size_t> dimensions)
{
	dims = dimensions;
	kernel.resize(dims.product());


	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, kernel.data(), sizeof(float)*dims.product()));
}


Kernel::Kernel(Vec<size_t> dimensions, float* values)
{
	dims = dimensions;

	if (values == NULL)
	{
		setOnes();
	}
	else
	{
		kernel.resize(dims.product());
		memcpy(kernel.data(), values, sizeof(float)*dims.product());
	}

	HANDLE_ERROR(cudaMemcpyToSymbol(cudaConstKernel, kernel.data(), sizeof(float)*dims.product()));
}


void Kernel::setOnes()
{
	for (int i = 0; i < dims.product(); ++i)
		kernel.push_back(1);
}


