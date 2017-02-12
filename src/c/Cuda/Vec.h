////////////////////////////////////////////////////////////////////////////////
//Copyright 2014 Andrew Cohen, Eric Wait, and Mark Winter
//This file is part of LEVER 3-D - the tool for 5-D stem cell segmentation,
//tracking, and lineaging. See http://bioimage.coe.drexel.edu 'software' section
//for details. LEVER 3-D is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by the Free
//Software Foundation, either version 3 of the License, or (at your option) any
//later version.
//LEVER 3-D is distributed in the hope that it will be useful, but WITHOUT ANY
//WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
//A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//You should have received a copy of the GNU General Public License along with
//LEVer in file "gnu gpl v3.txt".  If not, see  <http://www.gnu.org/licenses/>.
////////////////////////////////////////////////////////////////////////////////

#ifndef INCLUDE_VEC
#define INCLUDE_VEC

#include "Defines.h"

#include <type_traits>

#undef min
#undef max

#ifdef __CUDACC__
#define MIXED_PREFIX __host__ __device__
#else
#define MIXED_PREFIX
#endif


template<typename T>
class Vec
{
public:
	union
	{
		T e[3];
		struct  
		{
			T x;
			T y;
			T z;
		};
	};

	MIXED_PREFIX Vec() : x(0),y(0),z(0){}
	
	MIXED_PREFIX Vec(T val)
		: x(val), y(val), z(val)
	{}

	template<typename U>
	MIXED_PREFIX Vec(const Vec<U>& other)
		:	x(static_cast<T>(other.x)),
			y(static_cast<T>(other.y)),
			z(static_cast<T>(other.z))
	{}


	MIXED_PREFIX Vec(T x, T y, T z)
		: x(x), y(y), z(z)
	{}

	// Negates each element
	MIXED_PREFIX Vec<T> operator- () const
	{
		Vec<T> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = -e[i];

		return outVec;
	}

	// Adds each element by adder
	template<typename U>
	MIXED_PREFIX Vec<typename std::common_type<T, U>::type> operator+ (U adder) const
	{
		Vec<typename std::common_type<T, U>::type> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = e[i] + adder;

		return outVec;
	}

	// Subtracts each element by subtractor
	template<typename U>
	MIXED_PREFIX Vec<typename std::common_type<T, U>::type> operator- (U subtractor) const
	{
		Vec<typename std::common_type<T, U>::type> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = e[i] - subtractor;

		return outVec;
	}

	// Divides each element by divisor
	template<typename U>
	MIXED_PREFIX Vec<typename std::common_type<T, U>::type> operator/ (U divisor) const
	{
		Vec<typename std::common_type<T, U>::type> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = e[i] / divisor;

		return outVec;
	}

	// Multiplies each element by mult
	template<typename U>
	MIXED_PREFIX Vec<typename std::common_type<T, U>::type> operator* (U mult) const
	{
		Vec<typename std::common_type<T, U>::type> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = e[i] * mult;

		return outVec;
	}

	// Raises each element to the pwr
	template<typename U>
	MIXED_PREFIX Vec<typename std::common_type<T, U>::type> pwr (U pw) const
	{
		Vec<typename std::common_type<T, U>::type> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = pow((double)e[i], pw);

		return outVec;
	}

	// Returns the product of x*y*z
	MIXED_PREFIX T product() const
	{
		return x*y*z;
	}

	// Returns the sum of x+y+z
	MIXED_PREFIX T sum() const
	{
		return x+y+z;
	}

	// Returns the max value of x,y,z
	MIXED_PREFIX T maxValue() const
	{
		return (x>y) ? ((x>z)?(x):(z)) : ((y>z)?(y):(z));
	}

	// Returns the min value of x,y,z
	MIXED_PREFIX T minValue() const
	{
		return (x<y) ? ((x<z)?(x):(z)) : ((y<z)?(y):(z));
	}

	MIXED_PREFIX static Vec<T> min(Vec<T> a, Vec<T> b)
	{
		Vec<T> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = MIN(a.e[i], b.e[i]);

		return outVec;
	}

	MIXED_PREFIX static Vec<T> max(Vec<T> a, Vec<T> b)
	{
		Vec<T> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = MAX(a.e[i], b.e[i]);

		return outVec;
	}

	MIXED_PREFIX Vec<T> saturate(Vec<T> maxVal)
	{
		Vec<T> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = (e[i] < maxVal.e[i]) ? (e[i]) : (maxVal.e[i]);

		return outVec;
	}

	MIXED_PREFIX Vec<T> clamp(Vec<T> minVal, Vec<T> maxVal)
	{
		Vec<T> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = (e[i] < maxVal.e[i]) ? ((e[i] > minVal.e[i]) ? (e[i]) : (minVal.e[i])) : (maxVal.e[i]);

		return outVec;
	}

	template<typename U>
	MIXED_PREFIX Vec<size_t> coordAddressOf(U idx)const
	{
		Vec<size_t> vecOut = Vec<size_t>(0,0,0);
		if(x==0 && y==0 && z==0)
			throw runtime_error("Not a valid vector to index into!");

		if(x==0)
		{
			if(y==0)
			{
				vecOut.z = idx;
			} else
			{
				vecOut.z = idx/y;
				vecOut.y = idx - vecOut.z*y;
			}
		} else
		{
			if(y==0)
			{
				vecOut.z = idx/x;
				vecOut.x = idx - vecOut.z*x;
			} else
			{
				vecOut.z = idx/(x*y);
				idx -= vecOut.z*x*y;
				vecOut.y = idx/x;
				vecOut.x = idx - vecOut.y*x;
			}
		}
		return vecOut;
	}

	template <typename U>
	MIXED_PREFIX Vec& operator= (const Vec<U>& other)
	{
		for ( int i=0; i < 3; ++i )
			e[i] = other.e[i];

		return *this;
	}

	template <typename U>
	MIXED_PREFIX Vec<typename std::common_type<T,U>::type> operator+ (const Vec<U>& other) const
	{
		Vec<typename std::common_type<T, U>::type> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = e[i] + other.e[i];

		return outVec;
	}

	template <typename U>
	MIXED_PREFIX Vec<typename std::common_type<T, U>::type> operator- (const Vec<U>& other) const
	{
		Vec<typename std::common_type<T, U>::type> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = e[i] - other.e[i];

		return outVec;
	}

	template <typename U>
	MIXED_PREFIX Vec<typename std::common_type<T, U>::type> operator/ (const Vec<U>& other) const
	{
		Vec<typename std::common_type<T, U>::type> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = e[i] / other.e[i];

		return outVec;
	}

	template <typename U>
	MIXED_PREFIX Vec<typename std::common_type<T, U>::type>  operator* (const Vec<U>& other) const
	{
		Vec<typename std::common_type<T, U>::type> outVec;
		for ( int i=0; i < 3; ++i )
			outVec.e[i] = e[i] * other.e[i];

		return outVec;
	}

	template <typename U>
	MIXED_PREFIX Vec<T>& operator+= (const Vec<U>& other)
	{
		for ( int i=0; i < 3; ++i )
			e[i] += other.e[i];

		return *this;
	}

	template <typename U>
	MIXED_PREFIX Vec<T>& operator-= (const Vec<U>& other)
	{
		for ( int i=0; i < 3; ++i )
			e[i] -= other.e[i];

		return *this;
	}

	// Are all the values less then the passed in values
	MIXED_PREFIX bool operator< (const Vec<T>& inVec) const
	{
		return x<inVec.x && y<inVec.y && z<inVec.z;
	}

	MIXED_PREFIX bool operator<= (const Vec<T>& inVec) const
	{
		return x<=inVec.x && y<=inVec.y && z<=inVec.z;
	}

	// Are all the values greater then the passed in values
	MIXED_PREFIX bool operator>(const Vec<T>& inVec) const
	{
		return x>inVec.x && y>inVec.y && z>inVec.z;
	}

	MIXED_PREFIX bool operator>= (const Vec<T>& inVec) const
	{
		return x>=inVec.x && y>=inVec.y && z>=inVec.z;
	}

	MIXED_PREFIX bool operator== (const Vec<T>& inVec) const
	{
		return x==inVec.x && y==inVec.y && z==inVec.z;
	}

	MIXED_PREFIX bool operator!= (const Vec<T>& inVec) const
	{
		return x!=inVec.x||y!=inVec.y||z!=inVec.z;
	}

	// Returns the linear memory map if this is the dimensions and the passed in Vec is the coordinate
	MIXED_PREFIX size_t linearAddressAt(const Vec<T>& coordinate) const
	{
        return coordinate.x+coordinate.y*x+coordinate.z*y*x;
	}

	MIXED_PREFIX double EuclideanDistanceTo(const Vec<T>& other)
	{
		return sqrt((double)(SQR(x-other.x)+SQR(y-other.y)+SQR(z-other.z)));
	}

	MIXED_PREFIX double lengthSqr()
	{
		return SQR(x) + SQR(y) + SQR(z);
	}

	MIXED_PREFIX double length()
	{
		return sqrt(lengthSqr());
	}

	MIXED_PREFIX Vec<double> normal()
	{
		return ((*this) / length());
	}

    MIXED_PREFIX Vec<T> ceil() const
    {
        Vec<T> out;
        for(int i = 0; i<3; ++i)
        {
            out.e[i] = ::ceil(e[i]);
        }

        return out;
    }

	template <typename U, typename V>
	static MIXED_PREFIX Vec<typename std::common_type<U,V>::type> cross(const Vec<U>& a, const Vec<V>& b)
	{
		Vec<typename std::common_type<U, V>::type> o;

		o.x = a.y*b.z - a.z*b.y;
		o.y = -(a.x*b.z - a.z*b.x);
		o.z = a.x*b.y - a.y*b.x;

		return o;
	}

	template <typename U, typename V>
	static MIXED_PREFIX typename std::common_type<U,V>::type dot(const Vec<U>& a, const Vec<V>& b)
	{
		return a.x*b.x + a.y*b.y + a.z*b.z;
	}
};



#endif
