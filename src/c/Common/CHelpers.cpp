#include "CHelpers.h"

double* createEllipsoidKernel(Vec<unsigned int> radii, Vec<unsigned int>& kernelDims)
{
	kernelDims.x = radii.x*2+1;
	kernelDims.y = radii.y*2+1;
	kernelDims.z = radii.z*2+1;

	double* kernel = new double[kernelDims.product()];
	memset(kernel,0,sizeof(double)*kernelDims.product());

	Vec<unsigned int> mid;
	mid.x = (kernelDims.x+1)/2;
	mid.y = (kernelDims.y+1)/2;
	mid.z = (kernelDims.z+1)/2;
	Vec<float> dimScale(1.0f/((float)SQR(radii.x)),1.0f/((float)SQR(radii.y)),1.0f/((float)SQR(radii.z)));

	Vec<unsigned int> cur(0,0,0);
	for (cur.z=0; cur.z<kernelDims.z ; ++cur.z)
	{
		for (cur.y=0; cur.y<kernelDims.y ; ++cur.y)
		{
			for (cur.x=0; cur.x<kernelDims.x ; ++cur.x)
			{
				if (dimScale.x*SQR(cur.x-mid.x)+dimScale.y*SQR(cur.y-mid.y)+dimScale.z*SQR(cur.z-mid.z)<=1)
					kernel[kernelDims.linearAddressAt(cur)] = 1.0f;
			}
		}
	}

	return kernel;
}

