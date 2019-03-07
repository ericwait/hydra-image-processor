#pragma once
#include "Vec.h"
#include "CudaDeviceStats.h"
#include "ImageView.h"

#include <limits>
#include <algorithm>


#ifdef IMAGE_PROCESSOR_DLL
#ifdef IMAGE_PROCESSOR_INTERNAL
#define IMAGE_PROCESSOR_API __declspec(dllexport)
#else
#define IMAGE_PROCESSOR_API __declspec(dllimport)
#endif // IMAGE_PROCESSOR_INTERNAL
#else
#define IMAGE_PROCESSOR_API
#endif // IMAGE_PROCESSOR_DLL


#include "CWrapperAutogen.h"


IMAGE_PROCESSOR_API void clearDevice();

IMAGE_PROCESSOR_API int deviceCount();
IMAGE_PROCESSOR_API int deviceStats(DevStats** stats);
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
