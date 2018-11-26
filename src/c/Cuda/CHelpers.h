#pragma once

#include "Vec.h"
#include <memory.h>
#include <complex>

#ifdef IMAGE_PROCESSOR_DLL
#ifdef IMAGE_PROCESSOR_INTERNAL
#define IMAGE_PROCESSOR_API __declspec(dllexport)
#else
#define IMAGE_PROCESSOR_API __declspec(dllimport)
#endif // IMAGE_PROCESSOR_INTERNAL
#else
#define IMAGE_PROCESSOR_API
#endif // IMAGE_PROCESSOR_DLL

IMAGE_PROCESSOR_API float* createEllipsoidKernel(Vec<std::size_t> radii, Vec<std::size_t>& kernelDims);

IMAGE_PROCESSOR_API int calcOtsuThreshold(const double* normHistogram, int numBins);
