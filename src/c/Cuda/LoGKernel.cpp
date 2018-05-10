#include "KernelGenerators.h"

#include <functional>

float* createLoG_GausKernels(Vec<double> sigmas, Vec<size_t>& dimsOut)
{
	const double PI = std::atan(1.0) * 4;

	dimsOut.x = (size_t)MAX(1.0f, (10 * sigmas.x));
	dimsOut.y = (size_t)MAX(1.0f, (10 * sigmas.y));
	dimsOut.z = (size_t)MAX(1.0f, (10 * sigmas.z));

	dimsOut.x = (dimsOut.x % 2 == 0) ? (dimsOut.x + 1) : (dimsOut.x);
	dimsOut.y = (dimsOut.y % 2 == 0) ? (dimsOut.y + 1) : (dimsOut.y);
	dimsOut.z = (dimsOut.z % 2 == 0) ? (dimsOut.z + 1) : (dimsOut.z);

	Vec<double> mid = Vec<double>(dimsOut) / 2.0 - 0.5;

	float* kernelOut = new float[dimsOut.sum()*2];

	Vec<double> sigmaSqr = sigmas.pwr(2);
	Vec<double> oneOverSigSqr = Vec<double>(1.0) / sigmaSqr;
	Vec<double> twoSigmaSqr = sigmaSqr * 2;
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
			double gauss = exp(-(posSqr / twoSigmaSqr.e[i]));
			double logVal = (posSqr / sigmaForth.e[i] - oneOverSigSqr.e[i])*gauss;
			kernelOut[j + stride] = (float)logVal;
			kernelOut[j + stride + loGstride] = gauss;
			gaussSum += gauss;
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

	//bool is3d = sigmas != Vec<double>(0.0);
	//
	//if (is3d)
	//{
	//	// LaTeX form of LoG
	//	// $\frac{\Big(\frac{(x-\mu_x)^2}{\sigma_x^4}-\frac{1}{\sigma_x^2}+\frac{(y-\mu_y)^2}{\sigma_y^4}-\frac{1}{\sigma_y^2}+\frac{(z-\mu_z)^2}{\sigma_z^4}-\frac{1}{\sigma_z^2}\Big)\exp\Big(-\frac{(x-\mu_x)^2}{2\sigma_x^2}-\frac{(y-\mu_y)^2}{2\sigma_y^2}-\frac{(z-\mu)^2}{2\sigma_z^2}\Big)}{(2\pi)^{\frac{3}{2}}\sigma_x\sigma_y\sigma_z}$
	//	Vec<double> sigma4th = sigmas.pwr(4);
	//	double subtractor = (Vec<double>(1.0f, 1.0f, 1.0f) / sigmaSqr).sum();
	//	double denominator = pow(2.0*PI, 3.0 / 2.0)*sigmas.product();

	//	for (int i =0; i<3; ++i)
	//	{
	//		size_t startOffset = 0;
	//		if (i > 0)
	//			startOffset += dimsOut.x;

	//		if (i > 1)
	//			startOffset += dimsOut.y;

	//		for (int j = 0; j<dimsOut.e[i]; ++j)
	//		{
	//			int firstOther = 0;
	//			if (j == 0)
	//				firstOther = 1;
	//			int secondOther = 2;
	//			if (j == 2)
	//				secondOther = 1;

	//			double firstAdditive = -1.0 / sigmaSqr.e[firstOther];
	//			double secondAdditive = -1.0 / sigmaSqr.e[secondOther];

	//			double muSub = SQR(j - mid.e[i]);
	//			double muSubSigSqr = muSub / (2 * sigmaSqr.e[i]);
	//			double additive = muSub / sigma4th.e[i] - 1.0 / sigmaSqr.e[i];

	//			double posVal = ((additive + firstAdditive + secondAdditive) * exp(-muSubSigSqr)) / denominator;
	//			kernelOut[startOffset + j] = (float)posVal;
	//		}

	//	}
	//}
	//else
	//{
	//	// LaTeX form of LoG
	//	// $\frac{-1}{\pi\sigma_x^2\sigma_y^2}\Bigg(1-\frac{(x-\mu_x)^2}{2\sigma_x^2}-\frac{(y-\mu_y)^2}{2\sigma_y^2}\Bigg)\exp\Bigg(-\frac{(x-\mu_x)^2}{2\sigma_x^2}-\frac{(y-\mu_y)^2}{2\sigma_y^2}\Bigg)$
	//	// figure out which dim is zero
	//	double sigProd = 1.0;
	//	if (sigmas.x != 0)
	//		sigProd *= sigmas.x;
	//	if (sigmas.y != 0)
	//		sigProd *= sigmas.y;
	//	if (sigmas.z != 0)
	//		sigProd *= sigmas.z;

	//	double denominator = -PI*sigProd;

	//	for (int i=0; i<2; ++i)
	//	{
	//		size_t startOffset = i*dimsOut.e[0];

	//		for (int j = 0; j < dimsOut.e[i]; ++j)
	//		{
	//			double gaussComp = SQR(j - mid.e[i]) / twoSigmaSqr.e[i];
	//			double posVal = ((1.0 - gaussComp)*exp(-gaussComp)) / denominator;
	//			kernelOut[startOffset + j] = (float)posVal;
	//		}
	//	}
	//}

	return kernelOut;
}
