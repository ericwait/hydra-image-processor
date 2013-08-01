#ifndef VEC_H
#define VEC_H

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

	size_t product() const
	{
		return x*y*z;
	}

	double EuclideanDistanceTo(Vec<T> other)
	{
		return sqrt(SQR((double)x-other.x)+SQR((double)y-other.y)+SQR((double)z-other.z));
	}
};

#endif