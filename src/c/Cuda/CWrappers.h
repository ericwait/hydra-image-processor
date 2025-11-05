/**
 * @file CWrappers.h
 * @brief C API wrappers for CUDA image processing operations
 *
 * This file provides the main C API interface for the CUDA-accelerated image
 * processing library. It includes device management functions and auto-generated
 * wrapper functions for various image processing operations.
 */

#pragma once

#ifdef _WIN32
    #define DLL_EXPORT_API __declspec(dllexport)
    #define DLL_IMPORT_API __declspec(dllimport)
#else
    #define DLL_EXPORT_API __attribute__((visibility("default")))
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


#include <limits>
#include <algorithm>

#include "Vec.h"
#include "ImageView.h"
#include "CudaDeviceStats.h"

#include "CWrapperAutogen.h"

/**
 * @brief Clears and resets the current CUDA device
 *
 * Frees all memory allocations and resets the device state. Should be called
 * when cleaning up or when switching between devices.
 */
IMAGE_PROCESSOR_API void clearDevice();

/**
 * @brief Gets the number of available CUDA devices
 *
 * @return The number of CUDA-capable devices in the system
 */
IMAGE_PROCESSOR_API int deviceCount();

/**
 * @brief Retrieves statistics for all CUDA devices
 *
 * Allocates and fills an array of DevStats structures containing information
 * about each CUDA device in the system.
 *
 * @param stats Output parameter that receives a pointer to the allocated DevStats array.
 *              Caller is responsible for deallocation.
 * @return The number of devices for which statistics were retrieved
 */
IMAGE_PROCESSOR_API int deviceStats(DevStats** stats);

/**
 * @brief Retrieves memory statistics for all CUDA devices
 *
 * Returns available memory information for each device in the system.
 *
 * @param stats Output parameter that receives a pointer to the allocated memory statistics array.
 *              Caller is responsible for deallocation.
 * @return The number of devices for which memory statistics were retrieved
 */
IMAGE_PROCESSOR_API int memoryStats(std::size_t** stats);


/// Example wrapper header calls 
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<bool> imageIn, ImageView<bool> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<char> imageIn, ImageView<char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<unsigned char> imageIn, ImageView<unsigned char> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<short> imageIn, ImageView<short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<unsigned short> imageIn, ImageView<unsigned short> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<int> imageIn, ImageView<int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<unsigned int> imageIn, ImageView<unsigned int> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<float> imageIn, ImageView<float> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
//IMAGE_PROCESSOR_API void fooFilter(const ImageView<double> imageIn, ImageView<double> imageOut, ImageView<float> kernel, int numIterations = 1, int device = -1);
