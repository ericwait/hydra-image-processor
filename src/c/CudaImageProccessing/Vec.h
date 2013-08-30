#pragma once

#include "Defines.h"

template<typename T>
class Vec
{
public:
	T x;
	T y;
	T z;

	Vec(){x=0; y=0; z=0;};

	Vec(T x, T y, T z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	Vec<T> operator+ (Vec<T> other) const
	{
		Vec<T> outVec;
		outVec.x = x + other.x;
		outVec.y = y + other.y;
		outVec.z = z + other.z;

		return outVec;
	}

	Vec<T> operator- (Vec<T> other) const
	{
		Vec<T> outVec;
		outVec.x = x - other.x;
		outVec.y = y - other.y;
		outVec.z = z - other.z;

		return outVec;
	}

	Vec<T> operator- () const
	{
		Vec<T> outVec;
		outVec.x = -x;
		outVec.y = -y;
		outVec.z = -z;

		return outVec;
	}

	Vec<T>& operator= (const Vec<T> inVec)
	{
		x = inVec.x;
		y = inVec.y;
		z = inVec.z;

		return *this;
	}

	// Are all the values less then the passed in values
	bool operator< (const Vec<T> inVec)
	{
		return x<inVec.x && y<inVec.y && z<inVec.z;
	}

	// Are all the values greater then the passed in values
	bool operator> (const Vec<T> inVec)
	{
		return x>inVec.x && y>inVec.y && z>inVec.z;
	}

	bool operator== (const Vec<T> inVec)
	{
		return x==inVec.x && y==inVec.y && z==inVec.z;
	}

	// Returns the product of x*y*z
	size_t product() const
	{
		return x*y*z;
	}

	// Returns the linear memory map if this is the dimensions and the passed in Vec is the coordinate
	template<typename K>
	size_t linearAddressAt(Vec<K> coordinate) const
	{
		return x + y*coordinate.x + z*coordinate.y*coordinate.x;
	}

	double EuclideanDistanceTo(Vec<T> other)
	{
		return sqrt((double)(SQR(x-other.x) + SQR(y-other.y) + SQR(z-other.z)));
	}

	// Returns the max value of x,y,z
	T maxValue() const
	{
		return (x>y) ? ((x>z)?(x):(z)) : ((y>z)?(y):(z));
	}

	// Returns the min value of x,y,z
	T minValue() const
	{
		return (x<y) ? ((x<z)?(x):(z)) : ((y<z)?(y):(z));
	}
};
