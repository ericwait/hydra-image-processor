#include "CudaUtilities.cuh"

void calcBlockThread(const Vec<size_t>& dims, size_t maxThreads, dim3 &blocks, dim3 &threads)
{
	if (dims.z <= 1)
	{
		if (dims.y <= 1)
		{
			if (dims.x < maxThreads)
			{
				threads.x = (unsigned int)dims.x;
				threads.y = 1;
				threads.z = 1;
			} 
			else
			{
				threads.x = (unsigned int)maxThreads;
				threads.y = 1;
				threads.z = 1;
			}
		}
		else
		{
			if (dims.x*dims.y < maxThreads)
			{
				threads.x = (unsigned int)dims.x;
				threads.y = (unsigned int)dims.y;
				threads.z = 1;
			} 
			else
			{
				int dim = (unsigned int)sqrt((double)maxThreads);

				threads.x = dim;
				threads.y = dim;
				threads.z = 1;
			}
		}
	}
	else
	{
		if(dims.x*dims.y*dims.z < maxThreads)
		{
			threads.x = (unsigned int)dims.x;
			threads.y = (unsigned int)dims.y;
			threads.z = (unsigned int)dims.z;
		}
		else
		{
			unsigned long index;
			_BitScanReverse(&index,unsigned long(maxThreads));

			int dim = index/3;
			threads.x = 1 << MAX(dim,(int)index - 2*dim);
			threads.y = 1 << dim;
			threads.z = 1 << MIN(dim,(int)index - 2*dim);
		}
	}

	blocks.x = (unsigned int)ceil((float)dims.x/threads.x);
	blocks.y = (unsigned int)ceil((float)dims.y/threads.y);
	blocks.z = (unsigned int)ceil((float)dims.z/threads.z);
}

Vec<size_t> createGaussianKernel(Vec<float> sigma, float** kernelOut, Vec<int>& iterations)
{
	Vec<size_t> kernelDims(1,1,1);
	iterations = Vec<int>(1,1,1);

	if ((sigma.x+sigma.y+sigma.z)*3>MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM)
	{
		iterations.x = (int)MAX(1.0f,ceil(9.0f*SQR(sigma.x)/SQR(MAX_KERNEL_DIM)));
		iterations.y = (int)MAX(1.0f,ceil(9.0f*SQR(sigma.y)/SQR(MAX_KERNEL_DIM)));
		iterations.z = (int)MAX(1.0f,ceil(9.0f*SQR(sigma.z)/SQR(MAX_KERNEL_DIM)));

		//TODO: Optimize iterations per dim
		sigma.x = sigma.x/sqrt((float)iterations.x);
		sigma.y = sigma.y/sqrt((float)iterations.y);
		sigma.z = sigma.z/sqrt((float)iterations.z);
	}

	kernelDims.x = (size_t)MAX(1.0f,(3*sigma.x));
	kernelDims.y = (size_t)MAX(1.0f,(3*sigma.y));
	kernelDims.z = (size_t)MAX(1.0f,(3*sigma.z));

	kernelDims.x = (kernelDims.x%2==0) ? (kernelDims.x+1) : (kernelDims.x);
	kernelDims.y = (kernelDims.y%2==0) ? (kernelDims.y+1) : (kernelDims.y);
	kernelDims.z = (kernelDims.z%2==0) ? (kernelDims.z+1) : (kernelDims.z);

	Vec<size_t> mid;
	mid.x = kernelDims.x/2;
	mid.y = kernelDims.y/2;
	mid.z = kernelDims.z/2;

	*kernelOut = new float[kernelDims.sum()];
	float* kernel = *kernelOut;

	float total = 0.0;
	if (sigma.x==0)
	{
		kernel[0] = 1.0f;
	}
	else
	{
		for (size_t x=0; x<kernelDims.x ; ++x)
			total += kernel[x] =  exp(-(int)(SQR(mid.x-x)) / (2*SQR(sigma.x)));
		for (size_t x=0; x<kernelDims.x ; ++x)
			kernel[x] /= total;
	}

	total = 0.0;
	if (sigma.y==0)
	{
		kernel[kernelDims.x] = 1;
	}
	else
	{
		for (size_t y=0; y<kernelDims.y ; ++y)
			total += kernel[y+kernelDims.x] = exp(-(int)(SQR(mid.y-y)) / (2*SQR(sigma.y)));
		for (size_t y=0; y < kernelDims.y ; ++y)
			kernel[y+kernelDims.x] /= total;
	}

	total = 0.0;
	if (sigma.z==0)
	{
		kernel[kernelDims.x+kernelDims.y] = 1;
	}
	else
	{
		for (size_t z=0; z<kernelDims.z ; ++z)
			total += kernel[z+kernelDims.x+kernelDims.y] = exp(-(int)(SQR(mid.z-z)) / (2*SQR(sigma.z)));
		for (size_t z=0; z < kernelDims.z ; ++z)
			kernel[z+kernelDims.x+kernelDims.y] /= total;
	}

	return kernelDims;
}

Vec<size_t> createLoGKernel(Vec<float> sigma, float** kernelOut, size_t& kernSize)
{
	const double PI = std::atan(1.0)*4;

	Vec<size_t> kernelDims = Vec<size_t>(sigma*10.0f);

	// make odd
	kernelDims.x = (kernelDims.x!=0 && kernelDims.x%2==0) ? (kernelDims.x+1) : (kernelDims.x);
	kernelDims.y = (kernelDims.y!=0 && kernelDims.y%2==0) ? (kernelDims.y+1) : (kernelDims.y);
	kernelDims.z = (kernelDims.z!=0 && kernelDims.z%2==0) ? (kernelDims.z+1) : (kernelDims.z);

	Vec<int> center(Vec<float>(kernelDims-1)/2.0f);
	kernSize = 0;
	if (kernelDims.x>0)
		kernSize = kernelDims.x;

	if(kernelDims.y>0)
	{
		if(kernSize>0)
			kernSize *= kernelDims.y;
		else
			kernSize = kernelDims.y;
	}

	if(kernelDims.z>0)
	{
		if(kernSize>0)
			kernSize *= kernelDims.z;
		else
			kernSize = kernelDims.z;
	}

	*kernelOut = new float[kernSize];
	float* kernel = *kernelOut;

	for(int i = 0; i<kernSize; ++i)
		kernel[i] = 0.12345f;

	bool is3d = sigma!=Vec<float>(0.0f, 0.0f, 0.0f);

	Vec<double> sigmaSqr = sigma.pwr(2);
	Vec<double> twoSigmaSqr = sigmaSqr*2;

	if(is3d)
	{
		// LaTeX form of LoG
		// $\frac{\Big(\frac{(x-\mu_x)^2}{\sigma_x^4}-\frac{1}{\sigma_x^2}+\frac{(y-\mu_y)^2}{\sigma_y^4}-\frac{1}{\sigma_y^2}+\frac{(z-\mu_z)^2}{\sigma_z^4}-\frac{1}{\sigma_z^2}\Big)\exp\Big(-\frac{(x-\mu_x)^2}{2\sigma_x^2}-\frac{(y-\mu_y)^2}{2\sigma_y^2}-\frac{(z-\mu)^2}{2\sigma_z^2}\Big)}{(2\pi)^{\frac{3}{2}}\sigma_x\sigma_y\sigma_z}$
		Vec<double> sigma4th = sigma.pwr(4);
		double subtractor = (Vec<double>(1.0f, 1.0f, 1.0f)/sigmaSqr).sum();
		double denominator = pow(2.0*PI, 3.0/2.0)*sigma.product();

		Vec<int> curPos(0, 0, 0);
		for(curPos.z = 0; curPos.z<kernelDims.z||curPos.z==0; ++curPos.z)
		{
			double zMuSub = SQR(curPos.z-center.z);
			double zMuSubSigSqr = zMuSub/(2*sigmaSqr.z);
			double zAdditive = zMuSub/sigma4th.z-1.0/sigmaSqr.z;
			for(curPos.y = 0; curPos.y<kernelDims.y||curPos.y==0; ++curPos.y)
			{
				double yMuSub = SQR(curPos.y-center.y);
				double yMuSubSigSqr = yMuSub/(2*sigmaSqr.y);
				double yAdditive = yMuSub/sigma4th.y-1.0/sigmaSqr.y;
				for(curPos.x = 0; curPos.x<kernelDims.x||curPos.x==0; ++curPos.x)
				{
					double xMuSub = SQR(curPos.x-center.x);
					double xMuSubSigSqr = xMuSub/(2*sigmaSqr.x);
					double xAdditive = xMuSub/sigma4th.x-1.0/sigmaSqr.x;
					
					kernel[kernelDims.linearAddressAt(curPos)] = ((xAdditive+yAdditive+zAdditive)*exp(-xMuSubSigSqr-yMuSubSigSqr-zMuSubSigSqr))/denominator;
				}
			}
		}
	}
	else
	{
		// LaTeX form of LoG
		// $\frac{-1}{\pi\sigma_x^2\sigma_y^2}\Bigg(1-\frac{(x-\mu_x)^2}{2\sigma_x^2}-\frac{(y-\mu_y)^2}{2\sigma_y^2}\Bigg)\exp\Bigg(-\frac{(x-\mu_x)^2}{2\sigma_x^2}-\frac{(y-\mu_y)^2}{2\sigma_y^2}\Bigg)$
		// figure out which dim is zero
		double sigProd = 1.0;
		if (sigma.x!=0)
			sigProd *= sigma.x;
		if(sigma.y!=0)
			sigProd *= sigma.y;
		if(sigma.z!=0)
			sigProd *= sigma.z;
		
		double denominator = -PI*sigProd;

		Vec<double> gaussComp(0);
		Vec<int> curPos(0, 0, 0);
		for(curPos.z = 0; curPos.z<kernelDims.z || curPos.z==0; ++curPos.z)
		{
			if(sigma.z!=0)
			{
				gaussComp.z = SQR(curPos.z-center.z)/twoSigmaSqr.z;
			}
			for(curPos.y = 0; curPos.y<kernelDims.y || curPos.y==0; ++curPos.y)
			{
				if(sigma.y!=0)
				{
					gaussComp.y = SQR(curPos.y-center.y)/twoSigmaSqr.y;
				}
				for(curPos.x =0; curPos.x<kernelDims.x || curPos.x==0; ++curPos.x)
				{
					if(sigma.x!=0)
					{
						gaussComp.x = SQR(curPos.x-center.x)/twoSigmaSqr.x;
					}
					double gauss = gaussComp.sum();
					kernel[kernelDims.linearAddressAt(curPos)] = ((1.0-gauss)*exp(-gauss))/denominator;
				}
			}
		}
	}

	kernelDims.x = (0==kernelDims.x) ? (1) : (kernelDims.x);
	kernelDims.y = (0==kernelDims.y) ? (1) : (kernelDims.y);
	kernelDims.z = (0==kernelDims.z) ? (1) : (kernelDims.z);

	return kernelDims;
}

Vec<size_t> createGaussianKernelFull(Vec<float> sigma, float** kernelOut, Vec<size_t> maxKernelSize)
{
    Vec<size_t> kernelDims = Vec<size_t>(sigma.clamp(Vec<float>(1.0f), std::numeric_limits<float>::max()));

    for(float numStd = 3.0f; numStd>1.0f; numStd -= 0.2f)
    {
        if(sigma.product()*numStd<maxKernelSize.product())
        {
            kernelDims = sigma*numStd+0.9999f;
            break;
        }
    }

    kernelDims = kernelDims.clamp(Vec<size_t>(1), Vec<size_t>(std::numeric_limits<size_t>::max()));
    sigma = sigma.clamp(Vec<float>(0.1f), Vec<float>(std::numeric_limits<float>::max()));

    if(kernelDims.product()>MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM)
    {
        kernelDims = Vec<size_t>(MAX_KERNEL_DIM, MAX_KERNEL_DIM, MAX_KERNEL_DIM);
    }

    Vec<float> center = (kernelDims-1.0f)/2.0f;

    *kernelOut = new float[kernelDims.product()];
    float* kernel = *kernelOut;

    float total = 0.0f;
    Vec<size_t> pos(0, 0, 0);
    
    Vec<float> denominator = SQR(sigma)*2;
    for(pos.z = 0; pos.z<kernelDims.z; ++pos.z)
    {
        for(pos.y = 0; pos.y<kernelDims.y; ++pos.y)
        {
            for(pos.x = 0; pos.x<kernelDims.x; ++pos.x)
            {
                Vec<float> mahal = SQR(center-pos)/denominator;
                kernel[kernelDims.linearAddressAt(pos)] = exp(-(mahal.sum()));
                total += kernel[kernelDims.linearAddressAt(pos)];
            }
        }
    }

    for(int i = 0; i<kernelDims.product(); ++i)
        kernel[i] /= total;

    return kernelDims;
}

size_t memoryAvailable(int device, size_t* totalOut/*=NULL*/)
{
	HANDLE_ERROR(cudaSetDevice(device));
	size_t free, total;
	HANDLE_ERROR(cudaMemGetInfo(&free,&total));

	if (totalOut!=NULL)
		*totalOut = total;

	return free;
}

bool checkFreeMemory(size_t needed, int device, bool throws/*=false*/)
{
	size_t free = memoryAvailable(device);
	if (needed>free)
	{
		if (throws)
		{
			char buff[255];
			sprintf_s(buff,"Out of CUDA Memory!\nNeed: %zu\nHave: %zu\n",needed,free);
			throw std::runtime_error(buff);
		}
		return false;
	}
	return true;
}
