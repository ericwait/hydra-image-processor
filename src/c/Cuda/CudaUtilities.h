/**
 * @file CudaUtilities.h
 * @brief CUDA kernel launch utilities and error handling
 *
 * Provides utility functions for determining optimal kernel launch parameters,
 * error handling macros, and block/thread calculations for CUDA operations.
 */

#pragma once

#include "Vec.h"
#include "Defines.h"
#include "ImageView.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>
#include <vector>
#include <cuda_occupancy.h>

/**
 * @brief Calculates maximum threads per block for a kernel with dynamic shared memory
 *
 * Uses CUDA occupancy API to determine optimal thread block size considering
 * dynamic shared memory requirements.
 *
 * @tparam T Kernel function type
 * @tparam U Shared memory calculation functor type
 * @param func The kernel function pointer
 * @param f Functor that calculates dynamic shared memory size based on block size
 * @param threadLimit Maximum number of threads per block to consider (0 = no limit)
 * @return Recommended maximum number of threads per block
 */
template <typename T, typename U>
int getKernelMaxThreadsSharedMem(T func, U f, int threadLimit = 0)
{
	int blockSizeMax = 0;
	int minGridSize = 0;

	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSizeMax, func, f, threadLimit);

	return blockSizeMax;
}

/**
 * @brief Calculates maximum threads per block for a kernel without dynamic shared memory
 *
 * Uses CUDA occupancy API to determine optimal thread block size.
 *
 * @tparam T Kernel function type
 * @param func The kernel function pointer
 * @param threadLimit Maximum number of threads per block to consider (0 = no limit)
 * @return Recommended maximum number of threads per block
 */
template <typename T>
int getKernelMaxThreads(T func, int threadLimit=0)
{
    int blockSizeMax = 0;
    int minGridSize = 0;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeMax, func, 0, threadLimit);

    return blockSizeMax;
}

/**
 * @brief Handles CUDA errors by throwing an exception with detailed information
 *
 * @param err The CUDA error code to check
 * @param file Source file where the error occurred
 * @param line Line number where the error occurred
 * @throws std::runtime_error if err != cudaSuccess
 */
static void HandleError( cudaError_t err, const char *file, int line )
{
	if (err != cudaSuccess)
	{
		char errorMessage[255];
		sprintf(errorMessage, "%s in %s at line %d\n", cudaGetErrorString( err ),	file, line );
		throw std::runtime_error(errorMessage);
	}
}

/**
 * @brief Macro to check CUDA errors and throw exceptions with file/line info
 *
 * Usage: HANDLE_ERROR(cudaMemcpy(...));
 */
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#ifdef _DEBUG
/**
 * @brief Debug-mode macro to synchronize and check for kernel errors
 *
 * In debug builds, synchronizes the device and checks for kernel launch errors.
 * In release builds, this is a no-op.
 */
#define DEBUG_KERNEL_CHECK() { cudaThreadSynchronize(); HandleError( cudaPeekAtLastError(), __FILE__, __LINE__ ); }
#else
#define DEBUG_KERNEL_CHECK() {}
#endif // _DEBUG

/**
 * @brief Calculates optimal block and thread dimensions for a 3D CUDA kernel
 *
 * Determines the grid (blocks) and block (threads) dimensions for launching
 * a CUDA kernel over a 3D domain.
 *
 * @param dims The 3D dimensions of the problem domain
 * @param maxThreads Maximum threads per block allowed
 * @param blocks Output parameter receiving the grid dimensions
 * @param threads Output parameter receiving the block dimensions
 */
void calcBlockThread(const Vec<std::size_t>& dims, std::size_t maxThreads, Vec<unsigned int>& blocks, Vec<unsigned int>& threads);
