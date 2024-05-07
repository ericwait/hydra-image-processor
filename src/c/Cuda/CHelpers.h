#pragma once

#ifdef _WIN32
    #define DLL_EXPORT_API __declspec(dllexport)
    #define DLL_IMPORT_API __declspec(dllimport)
#else
    #define DLL_EXPORT_API __attribute__((__visibility("default")))
    #define DLL_IMPORT_API 
#endif 

#ifdef IMAGE_PROCESSOR_DLL
    #ifdef IMAGE_PROCESSOR_EXPORT
        #define IMAGE_PROCESSOR_API DLL_EXPORT_API
    #else
        #define IMAGE_PROCESSOR_API DLL_IMPORT_API
    #endif // IMAGE_PROCESSOR_EXPORT
#else
    #define IMAGE_PROCESSOR_API
#endif // IMAGE_PROCESSOR_DLL

#include "Vec.h"
#include <memory.h>
#include <complex>

IMAGE_PROCESSOR_API float* createEllipsoidKernel(Vec<std::size_t> radii, Vec<std::size_t>& kernelDims);

IMAGE_PROCESSOR_API int calcOtsuThreshold(const double* normHistogram, int numBins);
