#pragma once

#include <mex.h>

#include <cstddef>

namespace Script
{
	typedef mwSize	DimType;
	typedef mxArray ArrayType;
	typedef mxArray ObjectType;

	// Simple template-specialization map for C++ to mex types
	template <typename T> struct TypeMap {};
	template <> struct TypeMap<bool> { static const mxClassID typeId = mxLOGICAL_CLASS; };
	template <> struct TypeMap<char> { static const mxClassID typeId = mxINT8_CLASS; };
	template <> struct TypeMap<short> { static const mxClassID typeId = mxINT16_CLASS; };
	template <> struct TypeMap<int> { static const mxClassID typeId = mxINT32_CLASS; };
	template <> struct TypeMap<unsigned char> { static const mxClassID typeId = mxUINT8_CLASS; };
	template <> struct TypeMap<unsigned short> { static const mxClassID typeId = mxUINT16_CLASS; };
	template <> struct TypeMap<unsigned int> { static const mxClassID typeId = mxUINT32_CLASS; };
	template <> struct TypeMap<float> { static const mxClassID typeId = mxSINGLE_CLASS; };
	template <> struct TypeMap<double> { static const mxClassID typeId = mxDOUBLE_CLASS; };

	// Helper functions for array allocation
	template <typename T>
	ArrayType* createArray(int ndim, DimType* dims)
	{
		return mxCreateNumericArray(ndim, dims, TypeMap<T>::typeId, mxREAL);
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
