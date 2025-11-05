/**
 * @file CHelpers.h
 * @brief Helper functions and macros for the CUDA image processing library
 *
 * This file contains platform-specific DLL export/import macros and utility
 * functions for image processing operations such as kernel creation and
 * threshold calculation.
 */

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

/**
 * @brief Creates an ellipsoid-shaped convolution kernel
 *
 * Generates a 3D ellipsoid kernel with the specified radii. The kernel values
 * are set to 1.0 inside the ellipsoid and 0.0 outside.
 *
 * @param radii The radii of the ellipsoid in x, y, and z dimensions
 * @param kernelDims Output parameter that receives the dimensions of the created kernel
 * @return Pointer to the newly allocated kernel array. Caller is responsible for deallocation.
 */
IMAGE_PROCESSOR_API float* createEllipsoidKernel(Vec<std::size_t> radii, Vec<std::size_t>& kernelDims);

/**
 * @brief Calculates the Otsu threshold from a normalized histogram
 *
 * Implements Otsu's method for automatic threshold selection. This method
 * finds the threshold that minimizes the intra-class variance of the
 * thresholded image.
 *
 * @param normHistogram Pointer to the normalized histogram array
 * @param numBins Number of bins in the histogram
 * @return The optimal threshold bin index
 */
IMAGE_PROCESSOR_API int calcOtsuThreshold(const double* normHistogram, int numBins);
