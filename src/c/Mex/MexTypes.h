#pragma once

#include <mex.h>

#include <cstddef>
#include <cstdint>

namespace Script
{
	typedef mwSize	DimType;
	typedef mxArray ArrayType;
	typedef mxArray ObjectType;

	// Simple template-specialization map for C++ to mex types
	BEGIN_TYPE_MAP(mxClassID)
		TYPE_MAPPING(bool, mxLOGICAL_CLASS)
		TYPE_MAPPING(int8_t, mxINT8_CLASS)
		TYPE_MAPPING(int16_t, mxINT16_CLASS)
		TYPE_MAPPING(int32_t, mxINT32_CLASS)
		TYPE_MAPPING(uint8_t, mxUINT8_CLASS)
		TYPE_MAPPING(uint16_t, mxUINT16_CLASS)
		TYPE_MAPPING(uint32_t, mxUINT32_CLASS)
		TYPE_MAPPING(float, mxSINGLE_CLASS)
		TYPE_MAPPING(double, mxDOUBLE_CLASS)
	END_TYPE_MAP(mxClassID)

	// Helper functions for array allocation
	template <typename T>
	ArrayType* createArray(int ndim, DimType* dims)
	{
		return mxCreateNumericArray(ndim, dims, ID_FROM_TYPE(T), mxREAL);
	}

	template <>
	inline ArrayType* createArray<bool>(int ndim, DimType* dims)
	{
		return mxCreateLogicalArray(ndim, dims);
	}

	namespace ArrayInfo
	{
		inline std::size_t getNDims(const ArrayType* im){ return mxGetNumberOfDimensions(im); }
		inline const DimType* getDims(const ArrayType* im){ return mxGetDimensions(im); }

		template <typename T>
		T* getData(const ArrayType* im) { return (T*)mxGetData(im); }
	};
};
