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
	BEGIN_TYPE_MAP(mxClassID,Matlab)
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
	ArrayType* createArray(const DimInfo& info)
	{
		return mxCreateNumericArray(info.dims.size(), info.dims.data(), ID_FROM_TYPE(T), mxREAL);
	}

	template <>
	inline ArrayType* createArray<bool>(const DimInfo& info)
	{
		return mxCreateLogicalArray(info.dims.size(), info.dims.data());
	}

	namespace ArrayInfo
	{
		inline bool isColumnMajor(const ArrayType* im){return true;}
		inline bool isContiguous(const ArrayType* im){return true;}

		inline std::size_t getNDims(const ArrayType* im){ return mxGetNumberOfDimensions(im); }
		inline DimType getDim(const ArrayType* im, int idim){ return mxGetDimensions(im)[idim]; }

		template <typename T>
		T* getData(const ArrayType* im) { return (T*)mxGetData(im); }
	};
};
