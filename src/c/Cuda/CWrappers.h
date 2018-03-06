#pragma once
#include "Vec.h"
#include "CudaDeviceStats.h"
#include <limits>

#ifdef IMAGE_PROCESSOR_DLL
#ifdef IMAGE_PROCESSOR_INTERNAL
#define IMAGE_PROCESSOR_API __declspec(dllexport)
#else
#define IMAGE_PROCESSOR_API __declspec(dllimport)
#endif // IMAGE_PROCESSOR_INTERNAL
#else
#define IMAGE_PROCESSOR_API
#endif // IMAGE_PROCESSOR_DLL

IMAGE_PROCESSOR_API void clearDevice();

IMAGE_PROCESSOR_API int deviceCount();
IMAGE_PROCESSOR_API int deviceStats(DevStats** stats);
IMAGE_PROCESSOR_API int memoryStats(size_t** stats);
