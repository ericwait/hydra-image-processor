#include "CudaKernels.cuh"

__device__ DevicePixelType* SubDivide(DevicePixelType* pB, DevicePixelType* pE)
{
	DevicePixelType* pPivot = --pE;
	const DevicePixelType pivot = *pPivot;

	while (pB < pE)
	{
		if (*pB > pivot)
		{
			--pE;
			DevicePixelType temp = *pB;
			*pB = *pE;
			*pE = temp;
		} else
			++pB;
	}

	DevicePixelType temp = *pPivot;
	*pPivot = *pE;
	*pE = temp;

	return pE;
}

__device__ void SelectElement(DevicePixelType* pB, DevicePixelType* pE, size_t k)
{
	while (true)
	{
		DevicePixelType* pPivot = SubDivide(pB, pE);
		size_t n = pPivot - pB;

		if (n == k)
			break;

		if (n > k)
			pE = pPivot;
		else
		{
			pB = pPivot + 1;
			k -= (n + 1);
		}
	}
}

__device__ DevicePixelType cudaFindMedian(DevicePixelType* vals, int numVals)
{
	SelectElement(vals,vals+numVals, numVals/2);
	return vals[numVals/2];
}

