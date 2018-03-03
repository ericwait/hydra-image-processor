#include "KernelGenerators.h"

#include <functional>

Vec<size_t> createLoGKernel(Vec<float> sigma, float** kernelOut, size_t& kernSize)
{
	const double PI = std::atan(1.0) * 4;

	Vec<size_t> kernelDims = Vec<size_t>(sigma*10.0f);

	// make odd
	kernelDims.x = (kernelDims.x != 0 && kernelDims.x % 2 == 0) ? (kernelDims.x + 1) : (kernelDims.x);
	kernelDims.y = (kernelDims.y != 0 && kernelDims.y % 2 == 0) ? (kernelDims.y + 1) : (kernelDims.y);
	kernelDims.z = (kernelDims.z != 0 && kernelDims.z % 2 == 0) ? (kernelDims.z + 1) : (kernelDims.z);

	Vec<int> center(Vec<float>(kernelDims - 1) / 2.0f);
	kernSize = 0;
	if (kernelDims.x > 0)
		kernSize = kernelDims.x;

	if (kernelDims.y > 0)
	{
		if (kernSize > 0)
			kernSize *= kernelDims.y;
		else
			kernSize = kernelDims.y;
	}

	if (kernelDims.z > 0)
	{
		if (kernSize > 0)
			kernSize *= kernelDims.z;
		else
			kernSize = kernelDims.z;
	}

	*kernelOut = new float[kernSize];
	float* kernel = *kernelOut;

	for (int i = 0; i < kernSize; ++i)
		kernel[i] = 0.12345f;

	bool is3d = sigma != Vec<float>(0.0f, 0.0f, 0.0f);

	Vec<double> sigmaSqr = sigma.pwr(2);
	Vec<double> twoSigmaSqr = sigmaSqr * 2;

	if (is3d)
	{
		// LaTeX form of LoG
		// $\frac{\Big(\frac{(x-\mu_x)^2}{\sigma_x^4}-\frac{1}{\sigma_x^2}+\frac{(y-\mu_y)^2}{\sigma_y^4}-\frac{1}{\sigma_y^2}+\frac{(z-\mu_z)^2}{\sigma_z^4}-\frac{1}{\sigma_z^2}\Big)\exp\Big(-\frac{(x-\mu_x)^2}{2\sigma_x^2}-\frac{(y-\mu_y)^2}{2\sigma_y^2}-\frac{(z-\mu)^2}{2\sigma_z^2}\Big)}{(2\pi)^{\frac{3}{2}}\sigma_x\sigma_y\sigma_z}$
		Vec<double> sigma4th = sigma.pwr(4);
		double subtractor = (Vec<double>(1.0f, 1.0f, 1.0f) / sigmaSqr).sum();
		double denominator = pow(2.0*PI, 3.0 / 2.0)*sigma.product();

		Vec<int> curPos(0, 0, 0);
		for (curPos.z = 0; curPos.z < kernelDims.z || curPos.z == 0; ++curPos.z)
		{
			double zMuSub = SQR(curPos.z - center.z);
			double zMuSubSigSqr = zMuSub / (2 * sigmaSqr.z);
			double zAdditive = zMuSub / sigma4th.z - 1.0 / sigmaSqr.z;
			for (curPos.y = 0; curPos.y < kernelDims.y || curPos.y == 0; ++curPos.y)
			{
				double yMuSub = SQR(curPos.y - center.y);
				double yMuSubSigSqr = yMuSub / (2 * sigmaSqr.y);
				double yAdditive = yMuSub / sigma4th.y - 1.0 / sigmaSqr.y;
				for (curPos.x = 0; curPos.x < kernelDims.x || curPos.x == 0; ++curPos.x)
				{
					double xMuSub = SQR(curPos.x - center.x);
					double xMuSubSigSqr = xMuSub / (2 * sigmaSqr.x);
					double xAdditive = xMuSub / sigma4th.x - 1.0 / sigmaSqr.x;

					kernel[kernelDims.linearAddressAt(curPos)] = ((xAdditive + yAdditive + zAdditive)*exp(-xMuSubSigSqr - yMuSubSigSqr - zMuSubSigSqr)) / denominator;
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
		if (sigma.x != 0)
			sigProd *= sigma.x;
		if (sigma.y != 0)
			sigProd *= sigma.y;
		if (sigma.z != 0)
			sigProd *= sigma.z;

		double denominator = -PI*sigProd;

		Vec<double> gaussComp(0);
		Vec<int> curPos(0, 0, 0);
		for (curPos.z = 0; curPos.z < kernelDims.z || curPos.z == 0; ++curPos.z)
		{
			if (sigma.z != 0)
			{
				gaussComp.z = SQR(curPos.z - center.z) / twoSigmaSqr.z;
			}
			for (curPos.y = 0; curPos.y < kernelDims.y || curPos.y == 0; ++curPos.y)
			{
				if (sigma.y != 0)
				{
					gaussComp.y = SQR(curPos.y - center.y) / twoSigmaSqr.y;
				}
				for (curPos.x = 0; curPos.x < kernelDims.x || curPos.x == 0; ++curPos.x)
				{
					if (sigma.x != 0)
					{
						gaussComp.x = SQR(curPos.x - center.x) / twoSigmaSqr.x;
					}
					double gauss = gaussComp.sum();
					kernel[kernelDims.linearAddressAt(curPos)] = ((1.0 - gauss)*exp(-gauss)) / denominator;
				}
			}
		}
	}

	kernelDims.x = (0 == kernelDims.x) ? (1) : (kernelDims.x);
	kernelDims.y = (0 == kernelDims.y) ? (1) : (kernelDims.y);
	kernelDims.z = (0 == kernelDims.z) ? (1) : (kernelDims.z);

	return kernelDims;
}
