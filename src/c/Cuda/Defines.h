#pragma once
#include <cstddef>

//Percent of memory that can be used on the device
const double MAX_MEM_AVAIL = 0.95;

#define NUM_BINS (256)
#define MAX_KERNEL_DIM (25)
#define CONST_KERNEL_NUM_EL (MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM)

#define SQR(x) ((x)*(x))
#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

#define mat_to_c(x) ((x)-1)
#define c_to_mat(x) ((x)+1)

#define SIGN(x) (((x)>0) ? (1) : (((x)<0.000001 || (x)>-0.00001) ? (0) : (-1)))
#define CLAMP(val,minVal,maxVal) ((val>=maxVal) ? (maxVal) : ((val<=minVal) ? (minVal) : (val)))

enum ReductionMethods
{
	REDUC_MEAN, REDUC_MEDIAN, REDUC_MIN, REDUC_MAX, REDUC_GAUS
};

// Helper macros to clean up SFINAE code a little
// TODO: Prefer to use type aliases and templated constexpr, but doesn't mix well with top-level macros
#define IS_BOOL(Type) std::is_same<Type,bool>::value
// Convenience macro for getting implicit binary operation output type
#define BINOP_TYPE(TypeA,TypeB) typename std::common_type<TypeA,TypeB>::type
// Check for non-narrowing (valid) implicit conversions from SrcType -> DstType
#define NON_NARROWING(SrcType,DstType) std::is_same<BINOP_TYPE(SrcType,DstType),DstType>::value

#define SIGN_MATCH(TypeA,TypeB) ((std::is_unsigned<TypeA>::value && std::is_unsigned<TypeB>::value) || (std::is_signed<TypeA>::value && std::is_signed<TypeB>::value))
#define INT_MATCH(Type) (std::is_integral<Type>::value && !IS_BOOL(Type))
#define INT_SGN_MATCH(SrcType,DstType) (INT_MATCH(SrcType) && SIGN_MATCH(SrcType,DstType))
#define FLOAT_MATCH(Type) (std::is_floating_point<Type>::value)

//template <typename T> constexpr bool is_bool = std::is_same<T, bool>::value;
//
//template <typename SrcType, typename DstType>
//constexpr bool non_narrowing = std::is_same<typename std::common_type<SrcType, DstType>::type, DstType>::value;

// These are the outer-most routines for trying to clean up SFINAE conditions
#define ENABLE_CHK_T(...) typename std::enable_if<__VA_ARGS__, std::nullptr_t>::type
#define ENABLE_CHK(...) ENABLE_CHK_T(__VA_ARGS__) = nullptr
