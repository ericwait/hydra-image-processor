#include "KernelGenerators.h"

#include <functional>

float* createLoG_GausKernels(Vec<double> sigmas, Vec<std::size_t>& dimsOut)
{
	dimsOut.x = (std::size_t)MAX(1.0f, (10 * sigmas.x));
	dimsOut.y = (std::size_t)MAX(1.0f, (10 * sigmas.y));
	dimsOut.z = (std::size_t)MAX(1.0f, (10 * sigmas.z));

	dimsOut.x = (dimsOut.x % 2 == 0) ? (dimsOut.x + 1) : (dimsOut.x);
	dimsOut.y = (dimsOut.y % 2 == 0) ? (dimsOut.y + 1) : (dimsOut.y);
	dimsOut.z = (dimsOut.z % 2 == 0) ? (dimsOut.z + 1) : (dimsOut.z);

	Vec<double> mid = Vec<double>(dimsOut) / 2.0 - 0.5;

	float* kernelOut = new float[dimsOut.sum()*2];

	Vec<double> sigmaSqr = sigmas.pwr(2);
	Vec<double> sigmaForth = sigmas.pwr(4);

	int loGstride = dimsOut.sum();

	for (int i = 0; i < 3; ++i)
	{
		int stride = 0;
		if (i > 0)
			stride = dimsOut.x;
		if (i > 1)
			stride += dimsOut.y;

		if (sigmas.e[i] == 0)
		{
			kernelOut[stride] = 0.0f;
			kernelOut[stride + loGstride] = 1.0f;
			continue;
		}

		double gaussSum = 0.0;
		for (int j = 0; j < dimsOut.e[i]; ++j)
		{
			double pos = j - mid.e[i];
			double posSqr = SQR(pos);

			double gaussVal = exp(-(posSqr / (2.0 * sigmaSqr.e[i])));
			double logVal = (posSqr / sigmaForth.e[i] - 1.0 / sigmaSqr.e[i]) * gaussVal;

			// Multiply by sigma^2 to get a scale-invariant LoG
			double scaleInvVal = sigmaSqr.e[i] * logVal;

			gaussSum += gaussVal;

			kernelOut[j + stride] = scaleInvVal;
			kernelOut[j + stride + loGstride] = gaussVal;
		}

		double sumVal = 0.0;
		for (int j = 0; j < dimsOut.e[i]; ++j)
		{
			kernelOut[j + stride] /= (float)gaussSum;
			sumVal += kernelOut[j + stride];
			kernelOut[j + stride + loGstride] /= (float)gaussSum;
		}

		for (int j = 0; j < dimsOut.e[i]; ++j)
		{
			kernelOut[j + stride] -= (float)sumVal;
		}
	}

	return kernelOut;
}
