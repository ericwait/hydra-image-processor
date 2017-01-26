#ifndef INCLUDE_VEC
#define INCLUDE_VEC

#ifdef __CUDACC__
#define MIXED_PREFIX __host__ __device__
#else
#define MIXED_PREFIX
#endif

#include "Defines.h"

#include <type_traits>

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

	MIXED_PREFIX Vec(){x=0; y=0; z=0;};
    
    MIXED_PREFIX Vec(T val)
    {
        this->x = val;
        this->y = val;
        this->z = val;
    }

	template<typename U>
	MIXED_PREFIX Vec(const Vec<U>& other)
	{
		this->x = static_cast<T>(other.x);
		this->y = static_cast<T>(other.y);
		this->z = static_cast<T>(other.z);
	}


	MIXED_PREFIX Vec(T x, T y, T z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	// Negates each element
	MIXED_PREFIX Vec<T> operator- () const
	{
		Vec<T> outVec;
		outVec.x = -x;
		outVec.y = -y;
		outVec.z = -z;

		return outVec;
	}

	// Adds each element by adder
	template<typename U>
	MIXED_PREFIX Vec<typename std::common_type<T, U>::type> operator+ (U adder) const
	{
        Vec<typename std::common_type<T, U>::type> outVec;
		outVec.x = x+adder;
		outVec.y = y+adder;
		outVec.z = z+adder;

		return outVec;
	}

	// Subtracts each element by subtractor
	template<typename U>
	MIXED_PREFIX Vec<typename std::common_type<T, U>::type> operator- (U subtractor) const
	{
		Vec<typename std::common_type<T, U>::type> outVec;
		outVec.x = x-subtractor;
		outVec.y = y-subtractor;
		outVec.z = z-subtractor;

		return outVec;
	}

	// Divides each element by divisor
	template<typename U>
	MIXED_PREFIX Vec<typename std::common_type<T, U>::type> operator/ (U divisor) const
	{
		Vec<typename std::common_type<T, U>::type> outVec;
		outVec.x = x/divisor;
		outVec.y = y/divisor;
		outVec.z = z/divisor;

		return outVec;
	}

	// Multiplies each element by mult
	template<typename U>
	MIXED_PREFIX Vec<typename std::common_type<T, U>::type> operator* (U mult) const
	{
		Vec<typename std::common_type<T, U>::type> outVec;
		outVec.x = x*mult;
		outVec.y = y*mult;
		outVec.z = z*mult;

		return outVec;
	}

	// Raises each element to the pwr
	template<typename U>
	MIXED_PREFIX Vec<typename std::common_type<T, U>::type> pwr (U pw) const
	{
		Vec<typename std::common_type<T, U>::type> outVec;
		outVec.x = pow((double)x,pw);
		outVec.y = pow((double)y,pw);
		outVec.z = pow((double)z,pw);

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
		outVec.x = MIN(a.x, b.x);
		outVec.y = MIN(a.y, b.y);
		outVec.z = MIN(a.z, b.z);

		return outVec;
	}

	MIXED_PREFIX static Vec<T> max(Vec<T> a, Vec<T> b)
	{
		Vec<T> outVec;
		outVec.x = MAX(a.x, b.x);
		outVec.y = MAX(a.y, b.y);
		outVec.z = MAX(a.z, b.z);

		return outVec;
	}

	MIXED_PREFIX Vec<T> saturate(Vec<T> maxVal)
	{
		Vec<T> outVec;
		outVec.x = (x<maxVal.x) ? (x) : (maxVal.x);
		outVec.y = (y<maxVal.y) ? (y) : (maxVal.y);
		outVec.z = (z<maxVal.z) ? (z) : (maxVal.z);

		return outVec;
	}

	MIXED_PREFIX Vec<T> clamp(Vec<T> minVal, Vec<T> maxVal)
	{
		Vec<T> outVec;
		outVec.x = (x<maxVal.x) ? ((x>minVal.x) ? (x) : (minVal.x)) : (maxVal.x);
		outVec.y = (y<maxVal.y) ? ((y>minVal.y) ? (y) : (minVal.y)) : (maxVal.y);
		outVec.z = (z<maxVal.z) ? ((z>minVal.z) ? (z) : (minVal.z)) : (maxVal.z);

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
        this->x = other.x;
        this->y = other.y;
        this->z = other.z;

        return *this;
    }

    template <typename U>
    MIXED_PREFIX Vec<typename std::common_type<T,U>::type> operator+ (const Vec<U>& other) const
    {
        Vec<typename std::common_type<T, U>::type> outVec;
        outVec.x = x+other.x;
        outVec.y = y+other.y;
        outVec.z = z+other.z;

        return outVec;
    }

    template <typename U>
    MIXED_PREFIX Vec<typename std::common_type<T, U>::type> operator- (const Vec<U>& other) const
    {
        Vec<typename std::common_type<T, U>::type> outVec;
        outVec.x = x-other.x;
        outVec.y = y-other.y;
        outVec.z = z-other.z;

        return outVec;
    }

    template <typename U>
    MIXED_PREFIX Vec<typename std::common_type<T, U>::type> operator/ (const Vec<U>& other) const
    {
        Vec<typename std::common_type<T, U>::type> outVec;
        outVec.x = x/other.x;
        outVec.y = y/other.y;
        outVec.z = z/other.z;

        return outVec;
    }

    template <typename U>
    MIXED_PREFIX Vec<typename std::common_type<T, U>::type>  operator* (const Vec<U>& other) const
    {
        Vec<typename std::common_type<T, U>::type> outVec;
        outVec.x = x * other.x;
        outVec.y = y * other.y;
        outVec.z = z * other.z;

        return outVec;
    }

    template <typename U>
    MIXED_PREFIX Vec<T>& operator+= (const Vec<U>& other)
    {
        x += other.x;
        y += other.y;
        z += other.z;

        return *this;
    }

    template <typename U>
    MIXED_PREFIX Vec<T>& operator-= (const Vec<U>& other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;

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
};

#endif
