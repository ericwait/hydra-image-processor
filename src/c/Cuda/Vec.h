#ifndef HOST_VEC_ONCE
#define HOST_VEC_ONCE
#define DEVICE_PREFIX
#define INCLUDE_VEC
#define VEC_THIS_CLASS Vec
#define VEC_EXTERN_CLASS DeviceVec
#elif defined(DEVICE_VEC) && !defined(DEVICE_VEC_ONCE)
#define DEVICE_VEC_ONCE
#define DEVICE_PREFIX __device__
#define INCLUDE_VEC
#define VEC_THIS_CLASS DeviceVec
#define VEC_EXTERN_CLASS Vec
#endif

#ifdef INCLUDE_VEC

#include "Defines.h"

template<typename T> class VEC_EXTERN_CLASS;

template<typename T>
class VEC_THIS_CLASS
{
public:
	T x;
	T y;
	T z;

	DEVICE_PREFIX VEC_THIS_CLASS(){x=0; y=0; z=0;};

	template<typename U>
	DEVICE_PREFIX VEC_THIS_CLASS(VEC_THIS_CLASS<U> other)
	{
		this->x = static_cast<T>(other.x);
		this->y = static_cast<T>(other.y);
		this->z = static_cast<T>(other.z);
	}


	DEVICE_PREFIX VEC_THIS_CLASS(T x, T y, T z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	// Negates each element
	DEVICE_PREFIX VEC_THIS_CLASS<T> operator- () const
	{
		VEC_THIS_CLASS<T> outVec;
		outVec.x = -x;
		outVec.y = -y;
		outVec.z = -z;

		return outVec;
	}

	// Adds each element by adder
	template<typename d>
	DEVICE_PREFIX VEC_THIS_CLASS<T> operator+ (d adder) const
	{
		VEC_THIS_CLASS<T> outVec;
		outVec.x = (T)(x+adder);
		outVec.y = (T)(y+adder);
		outVec.z = (T)(z+adder);

		return outVec;
	}

	// Subtracts each element by subtractor
	template<typename d>
	DEVICE_PREFIX VEC_THIS_CLASS<T> operator- (d subtractor) const
	{
		VEC_THIS_CLASS<T> outVec;
		outVec.x = (T)(x-subtractor);
		outVec.y = (T)(y-subtractor);
		outVec.z = (T)(z-subtractor);

		return outVec;
	}

	// Divides each element by divisor
	template<typename d>
	DEVICE_PREFIX VEC_THIS_CLASS<T> operator/ (d divisor) const
	{
		VEC_THIS_CLASS<T> outVec;
		outVec.x = (T)(x/divisor);
		outVec.y = (T)(y/divisor);
		outVec.z = (T)(z/divisor);

		return outVec;
	}

	// Multiplies each element by mult
	template<typename d>
	DEVICE_PREFIX VEC_THIS_CLASS<T> operator* (d mult) const
	{
		VEC_THIS_CLASS<T> outVec;
		outVec.x = (T)(x*mult);
		outVec.y = (T)(y*mult);
		outVec.z = (T)(z*mult);

		return outVec;
	}

	// Raises each element to the pwr
	template<typename d>
	DEVICE_PREFIX VEC_THIS_CLASS<T> pwr (d pw) const
	{
		VEC_THIS_CLASS<T> outVec;
		outVec.x = (T)(pow((double)x,pw));
		outVec.y = (T)(pow((double)y,pw));
		outVec.z = (T)(pow((double)z,pw));

		return outVec;
	}

	// Returns the product of x*y*z
	DEVICE_PREFIX T product() const
	{
		return x*y*z;
	}

	// Returns the sum of x+y+z
	DEVICE_PREFIX T sum() const
	{
		return x+y+z;
	}

	// Returns the max value of x,y,z
	DEVICE_PREFIX T maxValue() const
	{
		return (x>y) ? ((x>z)?(x):(z)) : ((y>z)?(y):(z));
	}

	// Returns the min value of x,y,z
	DEVICE_PREFIX T minValue() const
	{
		return (x<y) ? ((x<z)?(x):(z)) : ((y<z)?(y):(z));
	}

	DEVICE_PREFIX static VEC_THIS_CLASS<T> min(VEC_THIS_CLASS<T> a, VEC_THIS_CLASS<T> b)
	{
		VEC_THIS_CLASS<T> outVec;
		outVec.x = MIN(a.x, b.x);
		outVec.y = MIN(a.y, b.y);
		outVec.z = MIN(a.z, b.z);

		return outVec;
	}

	DEVICE_PREFIX static VEC_THIS_CLASS<T> max(VEC_THIS_CLASS<T> a, VEC_THIS_CLASS<T> b)
	{
		VEC_THIS_CLASS<T> outVec;
		outVec.x = MAX(a.x, b.x);
		outVec.y = MAX(a.y, b.y);
		outVec.z = MAX(a.z, b.z);

		return outVec;
	}

	DEVICE_PREFIX VEC_THIS_CLASS<T> saturate(VEC_THIS_CLASS<T> maxVal)
	{
		VEC_THIS_CLASS<T> outVec;
		outVec.x = (x<maxVal.x) ? (x) : (maxVal.x);
		outVec.y = (y<maxVal.y) ? (y) : (maxVal.y);
		outVec.z = (z<maxVal.z) ? (z) : (maxVal.z);

		return outVec;
	}

	DEVICE_PREFIX VEC_THIS_CLASS<T> clamp(VEC_THIS_CLASS<T> minVal, VEC_THIS_CLASS<T> maxVal)
	{
		VEC_THIS_CLASS<T> outVec;
		outVec.x = (x<maxVal.x) ? ((x>minVal.x) ? (x) : (minVal.x)) : (maxVal.x);
		outVec.y = (y<maxVal.y) ? ((x>minVal.y) ? (y) : (minVal.y)) : (maxVal.y);
		outVec.z = (z<maxVal.z) ? ((x>minVal.z) ? (z) : (minVal.z)) : (maxVal.z);

		return outVec;
	}

#define EXTERN_TYPE VEC_THIS_CLASS
#include "VecFuncs.h"
#undef EXTERN_TYPE

#define EXTERN_TYPE VEC_EXTERN_CLASS
#include "VecFuncs.h"
#undef EXTERN_TYPE
};

#undef INCLUDE_VEC
#undef VEC_THIS_CLASS
#undef VEC_EXTERN_CLASS
#undef DEVICE_PREFIX
#endif