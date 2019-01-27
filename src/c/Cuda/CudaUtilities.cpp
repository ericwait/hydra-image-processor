#include "CudaUtilities.h"

#include <cuda_runtime.h>
#include <limits>

#pragma optimize("",off)

// Integral implementation of base-2 logarithm function
// NOTE: Returns bit index of highest on bit in mask
int ilog2(std::size_t mask)
{
	// NOTE: Mathematically this should be -Inf
	if ( mask == 0 )
		return -1;

	// Uses binary search for fast query
	const int total_bits = std::numeric_limits<std::size_t>::digits;
	std::size_t bitwidth = total_bits >> 1;
	std::size_t bitoffset = total_bits >> 1;

	// NOTE: depth 6 in binary search -> 2^6 or up to 64-bit numerical support
	for ( std::size_t d = 0; d < 6; ++d )
	{
		std::size_t bm = (((((std::size_t)1) << bitwidth) - 1) << bitoffset);
		std::size_t tl = (mask & bm);

		bitwidth = bitwidth >> 1;
		bitoffset = (tl) ? (bitoffset + bitwidth) : (bitoffset - bitwidth);

		if ( bitwidth == 0 )
			return (int)((tl) ? (bitoffset) : (bitoffset-1));
	}

	// NOTE: Mathematically this should be -Inf
	return -1;
}

//// Linear search base-2 integer logarithm function
//int ilog2(std::size_t mask)
//{
//	if ( mask == 0 )
//		return -1;
//
//	const int total_bits = std::numeric_limits<std::size_t>::digits;
//	for ( std::size_t i = total_bits-1; i >= 0; --i )
//		if ( (mask & (((std::size_t)1) << i)) != 0 )
//			return i;
//
//	return -1;
//}

void calcBlockThread(const Vec<std::size_t>& dims, std::size_t maxThreads, Vec<unsigned int>& blocks, Vec<unsigned int>& threads)
{
	if (dims.z <= 1)
	{
		if (dims.y <= 1)
		{
			if (dims.x < maxThreads)
			{
				threads.x = (unsigned int)dims.x;
				threads.y = 1;
				threads.z = 1;
			} 
			else
			{
				threads.x = (unsigned int)maxThreads;
				threads.y = 1;
				threads.z = 1;
			}
		}
		else
		{
			if (dims.x*dims.y < maxThreads)
			{
				threads.x = (unsigned int)dims.x;
				threads.y = (unsigned int)dims.y;
				threads.z = 1;
			} 
			else
			{
				int dim = (unsigned int)sqrt((double)maxThreads);

				threads.x = dim;
				threads.y = dim;
				threads.z = 1;
			}
		}
	}
	else
	{
		if(dims.x*dims.y*dims.z < maxThreads)
		{
			threads.x = (unsigned int)dims.x;
			threads.y = (unsigned int)dims.y;
			threads.z = (unsigned int)dims.z;
		}
		else
		{
			int index = ilog2(maxThreads);

			int dim = index/3;
			threads.x = 1 << MAX(dim,(int)index - 2*dim);
			threads.y = 1 << dim;
			threads.z = 1 << MIN(dim,(int)index - 2*dim);
		}
	}

	blocks.x = (unsigned int)ceil((float)dims.x/threads.x);
	blocks.y = (unsigned int)ceil((float)dims.y/threads.y);
	blocks.z = (unsigned int)ceil((float)dims.z/threads.z);
}
