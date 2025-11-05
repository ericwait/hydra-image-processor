/**
 * @file CudaDeviceStats.h
 * @brief CUDA device information and memory management functions
 *
 * This file provides structures and functions for querying CUDA device
 * capabilities, memory statistics, and verifying available memory.
 */

#pragma once
#include <string>

/**
 * @brief Structure containing CUDA device statistics and capabilities
 *
 * Holds comprehensive information about a CUDA device including compute
 * capability, memory sizes, and processing capabilities.
 */
struct DevStats
{
    std::string name;           ///< Device name string
    int major;                  ///< Major compute capability version
    int minor;                  ///< Minor compute capability version

    std::size_t constMem;       ///< Total constant memory in bytes
    std::size_t sharedMem;      ///< Shared memory per block in bytes
    std::size_t totalMem;       ///< Total global memory in bytes

    bool tccDriver;             ///< True if using Tesla Compute Cluster (TCC) driver
    int mpCount;                ///< Number of multiprocessors
    int threadsPerMP;           ///< Maximum threads per multiprocessor

    int warpSize;               ///< Warp size in threads
    int maxThreads;             ///< Maximum threads per block
};

/**
 * @brief Retrieves statistics for all CUDA devices in the system
 *
 * Allocates and fills an array of DevStats structures for all available
 * CUDA devices.
 *
 * @param stats Output parameter that receives a pointer to the allocated DevStats array.
 *              Caller is responsible for deallocation.
 * @return The number of CUDA devices found
 */
int cDeviceStats(DevStats** stats);

/**
 * @brief Gets the available memory on a specific CUDA device
 *
 * Queries the specified device for free and total memory.
 *
 * @param device The device index to query (use -1 for current device)
 * @param totalOut Optional output parameter that receives the total memory size
 * @return The amount of free memory available in bytes
 */
std::size_t memoryAvailable(int device, std::size_t* totalOut = NULL);

/**
 * @brief Checks if sufficient memory is available on a device
 *
 * Verifies that the requested amount of memory is available on the specified
 * device. Can optionally throw an exception if insufficient memory is found.
 *
 * @param needed The amount of memory required in bytes
 * @param device The device index to check (use -1 for current device)
 * @param throws If true, throws an exception when insufficient memory is available
 * @return True if sufficient memory is available, false otherwise
 */
bool checkFreeMemory(std::size_t needed, int device, bool throws = false);

