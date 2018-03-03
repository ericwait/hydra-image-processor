#include "KernelGenerators.h"

#include <float.h>
#include <functional>

Vec<size_t> createGaussianKernel(Vec<float> sigma, float** kernelOut, Vec<int>& iterations)
{
	Vec<size_t> kernelDims(1, 1, 1);
	iterations = Vec<int>(1, 1, 1);

	if ((sigma.x + sigma.y + sigma.z) * 3 > MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM)
	{
		iterations.x = (int)MAX(1.0f, ceil(9.0f*SQR(sigma.x) / SQR(MAX_KERNEL_DIM)));
		iterations.y = (int)MAX(1.0f, ceil(9.0f*SQR(sigma.y) / SQR(MAX_KERNEL_DIM)));
		iterations.z = (int)MAX(1.0f, ceil(9.0f*SQR(sigma.z) / SQR(MAX_KERNEL_DIM)));

		//TODO: Optimize iterations per dim
		sigma.x = sigma.x / sqrt((float)iterations.x);
		sigma.y = sigma.y / sqrt((float)iterations.y);
		sigma.z = sigma.z / sqrt((float)iterations.z);
	}

	kernelDims.x = (size_t)MAX(1.0f, (3 * sigma.x));
	kernelDims.y = (size_t)MAX(1.0f, (3 * sigma.y));
	kernelDims.z = (size_t)MAX(1.0f, (3 * sigma.z));

	kernelDims.x = (kernelDims.x % 2 == 0) ? (kernelDims.x + 1) : (kernelDims.x);
	kernelDims.y = (kernelDims.y % 2 == 0) ? (kernelDims.y + 1) : (kernelDims.y);
	kernelDims.z = (kernelDims.z % 2 == 0) ? (kernelDims.z + 1) : (kernelDims.z);

	Vec<size_t> mid;
	mid.x = kernelDims.x / 2;
	mid.y = kernelDims.y / 2;
	mid.z = kernelDims.z / 2;

	*kernelOut = new float[kernelDims.sum()];
	float* kernel = *kernelOut;

	float total = 0.0;
	if (sigma.x == 0)
	{
		kernel[0] = 1.0f;
	}
	else
	{
		for (size_t x = 0; x < kernelDims.x; ++x)
			total += kernel[x] = exp(-(int)(SQR(mid.x - x)) / (2 * SQR(sigma.x)));
		for (size_t x = 0; x < kernelDims.x; ++x)
			kernel[x] /= total;
	}

	total = 0.0;
	if (sigma.y == 0)
	{
		kernel[kernelDims.x] = 1;
	}
	else
	{
		for (size_t y = 0; y < kernelDims.y; ++y)
			total += kernel[y + kernelDims.x] = exp(-(int)(SQR(mid.y - y)) / (2 * SQR(sigma.y)));
		for (size_t y = 0; y < kernelDims.y; ++y)
			kernel[y + kernelDims.x] /= total;
	}

	total = 0.0;
	if (sigma.z == 0)
	{
		kernel[kernelDims.x + kernelDims.y] = 1;
	}
	else
	{
		for (size_t z = 0; z < kernelDims.z; ++z)
			total += kernel[z + kernelDims.x + kernelDims.y] = exp(-(int)(SQR(mid.z - z)) / (2 * SQR(sigma.z)));
		for (size_t z = 0; z < kernelDims.z; ++z)
			kernel[z + kernelDims.x + kernelDims.y] /= total;
	}

	return kernelDims;
}

Vec<size_t> createGaussianKernelFull(Vec<float> sigma, float** kernelOut, Vec<size_t> maxKernelSize)
{
	Vec<size_t> kernelDims = Vec<size_t>(sigma.clamp(Vec<float>(1.0f), FLT_MAX));

	for (float numStd = 3.0f; numStd > 1.0f; numStd -= 0.2f)
	{
		if (sigma.product()*numStd < maxKernelSize.product())
		{
			kernelDims = sigma*numStd + 0.9999f;
			break;
		}
	}

	kernelDims = kernelDims.clamp(Vec<size_t>(1), Vec<size_t>(ULLONG_MAX));
	sigma = sigma.clamp(Vec<float>(0.1f), Vec<float>(FLT_MAX));

	if (kernelDims.product() > MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM)
	{
		kernelDims = Vec<size_t>(MAX_KERNEL_DIM, MAX_KERNEL_DIM, MAX_KERNEL_DIM);
	}

	Vec<float> center = (kernelDims - 1.0f) / 2.0f;

	*kernelOut = new float[kernelDims.product()];
	float* kernel = *kernelOut;

	float total = 0.0f;
	Vec<size_t> pos(0, 0, 0);

	Vec<float> denominator = SQR(sigma) * 2;
	for (pos.z = 0; pos.z < kernelDims.z; ++pos.z)
	{
		for (pos.y = 0; pos.y < kernelDims.y; ++pos.y)
		{
			for (pos.x = 0; pos.x < kernelDims.x; ++pos.x)
			{
				Vec<float> mahal = SQR(center - pos) / denominator;
				kernel[kernelDims.linearAddressAt(pos)] = exp(-(mahal.sum()));
				total += kernel[kernelDims.linearAddressAt(pos)];
			}
		}
	}

	for (int i = 0; i < kernelDims.product(); ++i)
		kernel[i] /= total;

	return kernelDims;
}
