#include "KernelGenerators.h"

#include <functional>

float* createLoGKernel(Vec<double> sigmas, Vec<size_t>& dimsOut)
{
	const double PI = std::atan(1.0) * 4;

	dimsOut.x = (size_t)MAX(1.0f, (10 * sigmas.x));
	dimsOut.y = (size_t)MAX(1.0f, (10 * sigmas.y));
	dimsOut.z = (size_t)MAX(1.0f, (10 * sigmas.z));

	dimsOut.x = (dimsOut.x % 2 == 0) ? (dimsOut.x + 1) : (dimsOut.x);
	dimsOut.y = (dimsOut.y % 2 == 0) ? (dimsOut.y + 1) : (dimsOut.y);
	dimsOut.z = (dimsOut.z % 2 == 0) ? (dimsOut.z + 1) : (dimsOut.z);

	Vec<double> mid = Vec<double>(dimsOut) / 2.0 - 0.5;

	float* kernelOut = new float[dimsOut.product()];

	bool is3d = sigmas != Vec<double>(0.0f, 0.0f, 0.0f);

	Vec<double> sigmaSqr = sigmas.pwr(2);
	Vec<double> twoSigmaSqr = sigmaSqr * 2;

	if (is3d)
	{
		// LaTeX form of LoG
		// $\frac{\Big(\frac{(x-\mu_x)^2}{\sigma_x^4}-\frac{1}{\sigma_x^2}+\frac{(y-\mu_y)^2}{\sigma_y^4}-\frac{1}{\sigma_y^2}+\frac{(z-\mu_z)^2}{\sigma_z^4}-\frac{1}{\sigma_z^2}\Big)\exp\Big(-\frac{(x-\mu_x)^2}{2\sigma_x^2}-\frac{(y-\mu_y)^2}{2\sigma_y^2}-\frac{(z-\mu)^2}{2\sigma_z^2}\Big)}{(2\pi)^{\frac{3}{2}}\sigma_x\sigma_y\sigma_z}$
		Vec<double> sigma4th = sigmas.pwr(4);
		double subtractor = (Vec<double>(1.0f, 1.0f, 1.0f) / sigmaSqr).sum();
		double denominator = pow(2.0*PI, 3.0 / 2.0)*sigmas.product();

		Vec<int> curPos(0, 0, 0);
		for (curPos.z = 0; curPos.z < dimsOut.z; ++curPos.z)
		{
			double zMuSub = SQR(curPos.z - mid.z);
			double zMuSubSigSqr = zMuSub / (2 * sigmaSqr.z);
			double zAdditive = zMuSub / sigma4th.z - 1.0 / sigmaSqr.z;
			for (curPos.y = 0; curPos.y < dimsOut.y; ++curPos.y)
			{
				double yMuSub = SQR(curPos.y - mid.y);
				double yMuSubSigSqr = yMuSub / (2 * sigmaSqr.y);
				double yAdditive = yMuSub / sigma4th.y - 1.0 / sigmaSqr.y;
				for (curPos.x = 0; curPos.x < dimsOut.x; ++curPos.x)
				{
					double xMuSub = SQR(curPos.x - mid.x);
					double xMuSubSigSqr = xMuSub / (2 * sigmaSqr.x);
					double xAdditive = xMuSub / sigma4th.x - 1.0 / sigmaSqr.x;

					kernelOut[dimsOut.linearAddressAt(curPos)] = ((xAdditive + yAdditive + zAdditive)*exp(-xMuSubSigSqr - yMuSubSigSqr - zMuSubSigSqr)) / denominator;
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
		if (sigmas.x != 0)
			sigProd *= sigmas.x;
		if (sigmas.y != 0)
			sigProd *= sigmas.y;
		if (sigmas.z != 0)
			sigProd *= sigmas.z;

		double denominator = -PI*sigProd;

		Vec<double> gaussComp(0);
		Vec<int> curPos(0, 0, 0);
		for (curPos.z = 0; curPos.z < dimsOut.z; ++curPos.z)
		{
			if (sigmas.z != 0)
			{
				gaussComp.z = SQR(curPos.z - mid.z) / twoSigmaSqr.z;
			}
			for (curPos.y = 0; curPos.y < dimsOut.y; ++curPos.y)
			{
				if (sigmas.y != 0)
				{
					gaussComp.y = SQR(curPos.y - mid.y) / twoSigmaSqr.y;
				}
				for (curPos.x = 0; curPos.x < dimsOut.x; ++curPos.x)
				{
					if (sigmas.x != 0)
					{
						gaussComp.x = SQR(curPos.x - mid.x) / twoSigmaSqr.x;
					}
					double gauss = gaussComp.sum();
					kernelOut[dimsOut.linearAddressAt(curPos)] = ((1.0 - gauss)*exp(-gauss)) / denominator;
				}
			}
		}
	}

	return kernelOut;
}
