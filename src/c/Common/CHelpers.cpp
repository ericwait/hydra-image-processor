#include "CHelpers.h"

float* createEllipsoidKernel(Vec<size_t> radii, Vec<size_t>& kernelDims)
{
	kernelDims = radii*2 +1;

	float* kernel = new float[kernelDims.product()];
	memset(kernel,0,sizeof(float)*kernelDims.product());

	Vec<int> mid((kernelDims-1)/2);
	Vec<float> dimScale = Vec<float>(1,1,1) / Vec<float>(radii.pwr(2));

	Vec<int> cur(0,0,0);
	for (cur.z=0; cur.z<kernelDims.z; ++cur.z)
	{
		for (cur.y=0; cur.y<kernelDims.y; ++cur.y)
		{
			for (cur.x=0; cur.x<kernelDims.x; ++cur.x)
			{
				Vec<float> tmp = dimScale * Vec<float>((cur-mid).pwr(2));
				if (tmp.x+tmp.y+tmp.z<=1.0f)
				{
					size_t ind = kernelDims.linearAddressAt(cur);
					kernel[kernelDims.linearAddressAt(cur)] = 1.0f;
				}
			}
		}
	}

	return kernel;
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