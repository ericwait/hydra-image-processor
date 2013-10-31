#include "CudaUtilities.cuh"

void calcBlockThread(const Vec<size_t>& dims, const cudaDeviceProp &prop, dim3 &blocks, dim3 &threads)
{
	if (dims.z==1)
	{
		if (dims.y==1)
		{
			if ((int)dims.x<prop.maxThreadsPerBlock)
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
			if ((int)(dims.x*dims.y)<prop.maxThreadsPerBlock)
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
				int dim = (int)sqrt((double)prop.maxThreadsPerBlock);

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
		if((int)(dims.x*dims.y*dims.z)<prop.maxThreadsPerBlock)
		{
			threads.x = dims.x;
			threads.y = dims.y;
			threads.z = dims.z;

			blocks.x = 1;
			blocks.y = 1;
			blocks.z = 1;
		}
		else
		{
			int dim = (int)pow((float)prop.maxThreadsPerBlock,1/3.0f);
			float extra = (float)(prop.maxThreadsPerBlock-dim*dim*dim)/(dim*dim);

			threads.x = dim + (int)extra;
			threads.y = dim;
			threads.z = dim;

			blocks.x = (size_t)ceil((float)dims.x/threads.x);
			blocks.y = (size_t)ceil((float)dims.y/threads.y);
			blocks.z = (size_t)ceil((float)dims.z/threads.z);
		}
	}
}

Vec<size_t> createGaussianKernel(Vec<float> sigma, float* kernel, int& iterations)
{
	Vec<size_t> kernelDims(1,1,1);
	iterations = 1;

	if (sigma.product()*27>MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM)
	{
		Vec<int> minIterations;
		minIterations.x = (int)ceil(9.0f*SQR(sigma.x)/SQR(MAX_KERNEL_DIM));
		minIterations.y = (int)ceil(9.0f*SQR(sigma.y)/SQR(MAX_KERNEL_DIM));
		minIterations.z = (int)ceil(9.0f*SQR(sigma.z)/SQR(MAX_KERNEL_DIM));

		iterations = minIterations.maxValue();
		sigma = sigma/sqrt((float)iterations);
	}

	kernelDims.x = (size_t)(3*sigma.x);
	kernelDims.y = (size_t)(3*sigma.y);
	kernelDims.z = (size_t)(3*sigma.z);

	kernelDims.x = kernelDims.x%2==0 ? kernelDims.x+1 : kernelDims.x;
	kernelDims.y = kernelDims.y%2==0 ? kernelDims.y+1 : kernelDims.y;
	kernelDims.z = kernelDims.z%2==0 ? kernelDims.z+1 : kernelDims.z;

	Vec<size_t> mid;
	mid.x = kernelDims.x/2;
	mid.y = kernelDims.y/2;
	mid.z = kernelDims.z/2;

	Vec<size_t> cur(0,0,0);
	Vec<float> gaus;
	float total = 0.0;
	for (cur.x=0; cur.x<kernelDims.x ; ++cur.x)
	{
		for (cur.y=0; cur.y<kernelDims.y ; ++cur.y)
		{
			for (cur.z=0; cur.z<kernelDims.z ; ++cur.z)
			{
				gaus.x = exp(-(int)(SQR(mid.x-cur.x)) / (2*SQR(sigma.x)));
				gaus.y = exp(-(int)(SQR(mid.y-cur.y)) / (2*SQR(sigma.y)));
				gaus.z = exp(-(int)(SQR(mid.z-cur.z)) / (2*SQR(sigma.z)));

				total += kernel[kernelDims.linearAddressAt(cur)] = (float)gaus.product();
			}
		}
	}

	for (cur.x=0; cur.x < kernelDims.x ; ++cur.x)
		for (cur.y=0; cur.y < kernelDims.y ; ++cur.y)
			for (cur.z=0; cur.z < kernelDims.z ; ++cur.z)
				kernel[kernelDims.linearAddressAt(cur)] /= total;

	return kernelDims;
}

Vec<size_t> createGaussianKernel(Vec<float> sigma, float* kernel, Vec<int>& iterations)
{
	Vec<size_t> kernelDims(1,1,1);
	iterations = Vec<int>(1,1,1);

	if ((sigma.x+sigma.y+sigma.z)*3>MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM)
	{
		iterations.x = (int)ceil(9.0f*SQR(sigma.x)/SQR(MAX_KERNEL_DIM));
		iterations.y = (int)ceil(9.0f*SQR(sigma.y)/SQR(MAX_KERNEL_DIM));
		iterations.z = (int)ceil(9.0f*SQR(sigma.z)/SQR(MAX_KERNEL_DIM));

		//TODO: Optimize iterations per dim
		sigma.x = sigma.x/sqrt((float)iterations.x);
		sigma.y = sigma.y/sqrt((float)iterations.y);
		sigma.z = sigma.z/sqrt((float)iterations.z);
	}

	kernelDims.x = (size_t)(3*sigma.x);
	kernelDims.y = (size_t)(3*sigma.y);
	kernelDims.z = (size_t)(3*sigma.z);

	kernelDims.x = kernelDims.x%2==0 ? kernelDims.x+1 : kernelDims.x;
	kernelDims.y = kernelDims.y%2==0 ? kernelDims.y+1 : kernelDims.y;
	kernelDims.z = kernelDims.z%2==0 ? kernelDims.z+1 : kernelDims.z;

	Vec<size_t> mid;
	mid.x = kernelDims.x/2;
	mid.y = kernelDims.y/2;
	mid.z = kernelDims.z/2;

	float total = 0.0;
	for (size_t x=0; x<kernelDims.x ; ++x)
		total += kernel[x] =  exp(-(int)(SQR(mid.x-x)) / (2*SQR(sigma.x)));
	for (size_t x=0; x<kernelDims.x ; ++x)
		kernel[x] /= total;

	total = 0.0;
	for (size_t y=0; y<kernelDims.y ; ++y)
		total += kernel[y+kernelDims.x] = exp(-(int)(SQR(mid.y-y)) / (2*SQR(sigma.y)));
	for (size_t y=0; y < kernelDims.y ; ++y)
		kernel[y+kernelDims.x] /= total;

	total = 0.0;
	for (size_t z=0; z<kernelDims.z ; ++z)
		total += kernel[z+kernelDims.x+kernelDims.y] = exp(-(int)(SQR(mid.z-z)) / (2*SQR(sigma.z)));
	for (size_t z=0; z < kernelDims.z ; ++z)
		kernel[z+kernelDims.x+kernelDims.y] /= total;


	return kernelDims;
}

int calcOtsuThreshold(const double* normHistogram, int numBins)
{
	//code modified from http://www.dandiggins.co.uk/arlib-9.html
	double totalMean = 0.0f;
	for (int i=0; i<numBins; ++i)
		totalMean += i*normHistogram[i];

	double class1Prob=0, class1Mean=0, temp1, curThresh;
	double bestThresh = 0.0;
	int bestIndex = 0;
	for (int i=0; i<numBins; ++i)
	{
		class1Prob += normHistogram[i];
		class1Mean += i * normHistogram[i];

		temp1 = totalMean * class1Prob - class1Mean;
		curThresh = (temp1*temp1) / (class1Prob*(1.0f-class1Prob));

		if(curThresh>bestThresh)
		{
			bestThresh = curThresh;
			bestIndex = i;
		}
	}

	return bestIndex;
}