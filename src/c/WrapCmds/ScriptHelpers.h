#pragma once

#include "../Cuda/Vec.h"
#include "../Cuda/ImageDimensions.cuh"

#if defined(PY_BUILD)
 #include "PyTypes.h"
#elif defined(MEX_BUILD)
 #include "MexTypes.h"
#endif

#include <cstddef>
#include <cstring>
#include <algorithm>

namespace Script
{
	inline ImageDimensions setupDims(const ArrayType* im)
	{
		ImageDimensions dimsOut;

		dimsOut.dims = Vec<std::size_t>(1);
		dimsOut.chan = 1;
		dimsOut.frame = 1;

		std::size_t numDims = ArrayInfo::getNDims(im);
		const DimType* DIMS = ArrayInfo::getDims(im);

		for ( int i=0; i < std::min<std::size_t>(numDims, 3); ++i )
			dimsOut.dims.e[i] = (std::size_t) DIMS[i];

		if ( numDims > 3 )
			dimsOut.chan = (unsigned int)DIMS[3];

		if ( numDims > 4 )
			dimsOut.frame = (unsigned int)DIMS[4];

		return std::move(dimsOut);
	}


	template <typename T>
	void setupInputPointers(const ArrayType* imageIn, ImageDimensions& dims, T** image)
	{
		dims = setupDims(imageIn);
		*image = ArrayInfo::getData<T>(imageIn);
	}

	template <typename T>
	void setupOutputPointers(ArrayType** imageOut, ImageDimensions& dims, T** image)
	{
		DimType outDims[5];
		for ( int i = 0; i < 3; ++i )
			outDims[i] = dims.dims.e[i];

		outDims[3] = dims.chan;
		outDims[4] = dims.frame;

		*imageOut = createArray<T>(5, outDims);
		*image = ArrayInfo::getData<T>(*imageOut);

		std::memset(*image, 0, sizeof(T)*dims.getNumElements());
	}

	template <typename T>
	void setupImagePointers(const ArrayType* imageIn, T** image, ImageDimensions& dims, ArrayType** argOut = nullptr, T** imageOut = nullptr)
	{
		setupInputPointers(imageIn, dims, image);
		if ( argOut != nullptr && imageOut != nullptr )
			setupOutputPointers(argOut, dims, imageOut);
	}
};
