#include "CHelpers.h"
#include <vector>
#include "Defines.h"

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
					kernel[kernelDims.linearAddressAt(cur)] = 1.0f;
				}
			}
		}
	}

	return kernel;
}

int calcOtsuThreshold(const double* normHistogram, int numBins)
{
	double* omegas = new double[numBins];
	double* mus = new double[numBins];
	double* sigma_b_squared = new double[numBins];
	double maxVal = 0.0;

	std::vector<int> maxs;
	maxs.reserve(numBins);

	omegas[0] = normHistogram[0];
	mus[0] = 0.0;

	double lastOmega = normHistogram[0];
	double lastMu = 0.0;
	for (int i=1; i<numBins; ++i)
	{
		omegas[i] = lastOmega + normHistogram[i];
		mus[i] = lastMu + (i*normHistogram[i]);

		lastOmega = omegas[i];
		lastMu = mus[i];
	}

	for (int i=0; i<numBins; ++i)
	{
		sigma_b_squared[i] = SQR(mus[numBins-1] * omegas[i] - mus[i]) / (omegas[i] * (1.0 - omegas[i]));
		if (maxVal < sigma_b_squared[i])
			maxVal = sigma_b_squared[i];
	}

	for (int i=0; i<numBins; ++i)
	{
		if (sigma_b_squared[i] > maxVal*0.95)
			maxs.push_back(i);
	}

	double mean = 0.0;

	if (maxVal!=0 && !maxs.empty())
	{
		for (std::vector<int>::iterator it=maxs.begin(); it!=maxs.end(); ++it)
		{
			mean += *it;
		}

		mean /= maxs.size();
	}

	delete[] omegas;
	delete[] mus;
	delete[] sigma_b_squared;

	return floor(mean);
}