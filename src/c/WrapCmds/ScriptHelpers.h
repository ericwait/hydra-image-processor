#pragma once

#include "../Cuda/Vec.h"
#include "../Cuda/ImageDimensions.cuh"
#include "../Cuda/ImageView.h"

#include <cstddef>
#include <cstring>
#include <vector>
#include <algorithm>


#define BEGIN_TYPE_MAP(EnumType)					\
	typedef EnumType IdType;						\
	template <typename T> struct TypeToIdMap {};	\
	template <typename T> struct TypeNameMap {};	

#define TYPE_MAPPING(Type,TypeID)															\
	template <> struct TypeToIdMap<Type> {static constexpr const IdType typeId = TypeID;};	\
	template <> struct TypeNameMap<Type> {static constexpr const char name[] = #Type;};

#define END_TYPE_MAP(EnumType)

#define TYPE_FROM_ID(TypeID) IdToTypeMap<TypeID>::type
#define ID_FROM_TYPE(Type) TypeToIdMap<Type>::typeId

namespace Script
{
	struct DimInfo
	{
		bool contiguous;
		bool columnMajor;
		std::vector<std::size_t> dims;
	};

	inline DimInfo maxDims(const DimInfo& infoA, const DimInfo& infoB)
	{
		DimInfo outInfo = (infoA.dims.size() > infoB.dims.size()) ? infoA : infoB;

		std::size_t sharedSize = std::min(infoA.dims.size(), infoB.dims.size());
		for ( int i=0; i < sharedSize; ++i )
			outInfo.dims[i] = std::max(infoA.dims[i], infoB.dims[i]);

		return outInfo;
	}

	inline ImageDimensions makeImageDims(const DimInfo& info)
	{
		ImageDimensions dimsOut(Vec<std::size_t>(1), 1, 1);

		int nsdims = std::min<int>(3, (int)info.dims.size());
		for ( int i=0; i < nsdims; ++i )
			dimsOut.dims.e[i] = info.dims[i];

		if ( info.dims.size() >= 4 )
			dimsOut.chan = (unsigned int)info.dims[3];

		if ( info.dims.size() >= 5 )
			dimsOut.frame = (unsigned int)info.dims[4];

		return dimsOut;
	}
};


#if defined(PY_BUILD)
 #include "PyTypes.h"
#elif defined(MEX_BUILD)
 #include "MexTypes.h"
#endif

namespace Script
{
	inline DimInfo getDimInfo(const ArrayType* im)
	{
		DimInfo info;
		info.contiguous = ArrayInfo::isContiguous(im);
		info.columnMajor = ArrayInfo::isColumnMajor(im);

		std::size_t ndims = ArrayInfo::getNDims(im);
		info.dims.resize(ndims);

		// Load dimensions in reverse order if row-major
		int offset = (info.columnMajor) ? (0) : ((int)ndims-1);
		int m = (info.columnMajor) ? (1) : (-1);

		for ( int i=0; i < ndims; ++i )
			info.dims[i] = ArrayInfo::getDim(im, m*i + offset);

		return info;
	}

	template <typename T>
	ImageView<T> wrapInputImage(const ArrayType* imageIn, const DimInfo& info)
	{
		return ImageView<T>(ArrayInfo::getData<T>(imageIn), makeImageDims(info));
	}

	template <typename T>
	ImageView<T> createOutputImage(ArrayType** imageOut, const DimInfo& info)
	{
		*imageOut = createArray<T>(info);
		return wrapInputImage<T>(*imageOut, info);
	}
};
