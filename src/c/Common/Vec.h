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
DEVICE_PREFIX VEC_THIS_CLASS<T> pow (d pwr) const
{
	VEC_THIS_CLASS<T> outVec;
	outVec.x = (T)(pow(x,pwr));
	outVec.y = (T)(pow(y,pwr));
	outVec.z = (T)(pow(z,pwr));

	return outVec;
}

// Returns the product of x*y*z
DEVICE_PREFIX T product() const
{
	return x*y*z;
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