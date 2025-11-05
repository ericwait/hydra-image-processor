/**
 * @file Defines.h
 * @brief Common constants, macros, and type traits for the image processor
 *
 * This file contains:
 * - Global constants for memory management and array sizes
 * - Mathematical and utility macros
 * - SFINAE (Substitution Failure Is Not An Error) helper macros for template metaprogramming
 * - Enumerations for algorithm options
 */

#pragma once
#include <cstddef>

/// @brief Maximum fraction of device memory that can be used (95%)
const double MAX_MEM_AVAIL = 0.95;

/// @brief Number of histogram bins for image processing operations
#define NUM_BINS (256)

/// @brief Maximum dimension for constant memory kernels
#define MAX_KERNEL_DIM (25)

/// @brief Total number of elements in the maximum constant memory kernel
#define CONST_KERNEL_NUM_EL (MAX_KERNEL_DIM*MAX_KERNEL_DIM*MAX_KERNEL_DIM)

/// @brief Square of a value
#define SQR(x) ((x)*(x))

/// @brief Maximum of two values
#define MAX(x,y) (((x)>(y))?(x):(y))

/// @brief Minimum of two values
#define MIN(x,y) (((x)<(y))?(x):(y))

/// @brief Convert MATLAB 1-based index to C 0-based index
#define mat_to_c(x) ((x)-1)

/// @brief Convert C 0-based index to MATLAB 1-based index
#define c_to_mat(x) ((x)+1)

/// @brief Returns the sign of a value (-1, 0, or 1)
#define SIGN(x) (((x)>0) ? (1) : (((x)<0.000001 || (x)>-0.00001) ? (0) : (-1)))

/// @brief Clamps a value between minimum and maximum bounds
#define CLAMP(val,minVal,maxVal) ((val>=maxVal) ? (maxVal) : ((val<=minVal) ? (minVal) : (val)))

/**
 * @brief Enumeration of reduction methods for dimensional reduction operations
 */
enum ReductionMethods
{
	REDUC_MEAN,     ///< Mean average reduction
	REDUC_MEDIAN,   ///< Median value reduction
	REDUC_MIN,      ///< Minimum value reduction
	REDUC_MAX,      ///< Maximum value reduction
	REDUC_GAUS      ///< Gaussian-weighted reduction
};

// ============================================================================
// SFINAE Helper Macros
// ============================================================================
// These macros help clean up template metaprogramming code using SFINAE
// (Substitution Failure Is Not An Error) to enable/disable templates based
// on type traits.

/// @brief Checks if a type is bool
#define IS_BOOL(Type) std::is_same<Type,bool>::value

/// @brief Gets the implicit binary operation output type for two types
#define BINOP_TYPE(TypeA,TypeB) typename std::common_type<TypeA,TypeB>::type

/// @brief Checks for non-narrowing (valid) implicit conversions from SrcType to DstType
#define NON_NARROWING(SrcType,DstType) std::is_same<BINOP_TYPE(SrcType,DstType),DstType>::value

/// @brief Checks if two types have matching signedness (both signed or both unsigned)
#define SIGN_MATCH(TypeA,TypeB) ((std::is_unsigned<TypeA>::value && std::is_unsigned<TypeB>::value) || (std::is_signed<TypeA>::value && std::is_signed<TypeB>::value))

/// @brief Checks if a type is an integer (excluding bool)
#define INT_MATCH(Type) (std::is_integral<Type>::value && !IS_BOOL(Type))

/// @brief Checks if both types are integers with matching signedness
#define INT_SGN_MATCH(SrcType,DstType) (INT_MATCH(SrcType) && SIGN_MATCH(SrcType,DstType))

/// @brief Checks if a type is floating-point
#define FLOAT_MATCH(Type) (std::is_floating_point<Type>::value)

/// @brief Checks if a type is numeric (integer or floating-point, including bool)
#define NUMERIC_MATCH(Type) (std::is_arithmetic<Type>::value)

/// @brief Checks if a type is numeric but not bool
#define NUMERIC_NONBOOL(Type) (std::is_arithmetic<Type>::value && !IS_BOOL(Type))

/// @brief Checks if two types are the same
#define IS_SAME(TypeA,TypeB) (std::is_same<TypeA,TypeB>::value)

/// @brief SFINAE enable_if helper that returns a type
#define ENABLE_CHK_T(...) typename std::enable_if<__VA_ARGS__, std::nullptr_t>::type

/// @brief SFINAE enable_if helper for template parameters
#define ENABLE_CHK(...) ENABLE_CHK_T(__VA_ARGS__) = nullptr
