#include "CHelpers.h"

double* createCircleKernel(int rad, Vec<unsigned int>& kernelDims)
{
	kernelDims.x = rad*2+1;
	kernelDims.y = rad*2+1;
	kernelDims.z = rad*2+1;

	double* kernel = new double[kernelDims.product()];
	memset(kernel,0,sizeof(double)*kernelDims.product());

	Vec<unsigned int> mid;
	mid.x = kernelDims.x/2+1;
	mid.y = kernelDims.y/2+1;
	mid.z = kernelDims.z/2+1;

	Vec<unsigned int> cur(0,0,0);
	for (cur.z=0; cur.z<kernelDims.z ; ++cur.z)
	{
		for (cur.y=0; cur.y<kernelDims.y ; ++cur.y)
		{
			for (cur.x=0; cur.x<kernelDims.x ; ++cur.x)
			{
				double dist = cur.EuclideanDistanceTo(mid);
				double dist2 = sqrt((double)((cur.x-mid.x)*(cur.x-mid.x)+(cur.y-mid.y)*(cur.y-mid.y)+(cur.z-mid.z)*(cur.z-mid.z)));
				if (dist!=dist2)
					kernel[0]=1e6;
				if (dist<=rad)
					kernel[kernelDims.linearAddressAt(cur)] = 1.0f;
			}
		}
	}

	return kernel;
}
