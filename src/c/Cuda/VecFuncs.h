#ifdef EXTERN_TYPE

DEVICE_PREFIX VEC_THIS_CLASS(const EXTERN_TYPE<T>& other)
{
	this->x = other.x;
	this->y = other.y;
	this->z = other.z;
}

DEVICE_PREFIX VEC_THIS_CLASS& operator= (const EXTERN_TYPE<T>& other)
{
	this->x = other.x;
	this->y = other.y;
	this->z = other.z;

	return *this;
}

DEVICE_PREFIX VEC_THIS_CLASS<T> operator+ (const EXTERN_TYPE<T>& other) const
{
	VEC_THIS_CLASS<T> outVec;
	outVec.x = x + other.x;
	outVec.y = y + other.y;
	outVec.z = z + other.z;

	return outVec;
}

DEVICE_PREFIX VEC_THIS_CLASS<T> operator- (const EXTERN_TYPE<T>& other) const
{
	VEC_THIS_CLASS<T> outVec;
	outVec.x = x - other.x;
	outVec.y = y - other.y;
	outVec.z = z - other.z;

	return outVec;
}

DEVICE_PREFIX VEC_THIS_CLASS<T> operator/ (const EXTERN_TYPE<T>& other) const
{
	VEC_THIS_CLASS<T> outVec;
	outVec.x = x / other.x;
	outVec.y = y / other.y;
	outVec.z = z / other.z;

	return outVec;
}

DEVICE_PREFIX VEC_THIS_CLASS<T> operator* (const EXTERN_TYPE<T>& other) const
{
	VEC_THIS_CLASS<T> outVec;
	outVec.x = x * other.x;
	outVec.y = y * other.y;
	outVec.z = z * other.z;

	return outVec;
}

// Adds each element by other
template<typename d>
DEVICE_PREFIX VEC_THIS_CLASS<T>& operator+= (const EXTERN_TYPE<T>& other) const
{
	x += other.x;
	y += other.y;
	z += other.z;

	return this;
}

// Subtracts each element by other
template<typename d>
DEVICE_PREFIX VEC_THIS_CLASS<T>& operator-= (const EXTERN_TYPE<T>& other) const
{
	x -= other.x;
	y -= other.y;
	z -= other.z;

	return this;
}

// Are all the values less then the passed in values
DEVICE_PREFIX bool operator< (const EXTERN_TYPE<T>& inVec)
{
	return x<inVec.x && y<inVec.y && z<inVec.z;
}

DEVICE_PREFIX bool operator<= (const EXTERN_TYPE<T>& inVec)
{
	return x<=inVec.x && y<=inVec.y && z<=inVec.z;
}

// Are all the values greater then the passed in values
DEVICE_PREFIX bool operator> (const EXTERN_TYPE<T>& inVec)
{
	return x>inVec.x && y>inVec.y && z>inVec.z;
}

DEVICE_PREFIX bool operator>= (const EXTERN_TYPE<T>& inVec)
{
	return x>=inVec.x && y>=inVec.y && z>=inVec.z;
}

DEVICE_PREFIX bool operator== (const EXTERN_TYPE<T>& inVec)
{
	return x==inVec.x && y==inVec.y && z==inVec.z;
}

DEVICE_PREFIX bool operator!= (const EXTERN_TYPE<T>& inVec)
{
	return x!=inVec.x || y!=inVec.y || z!=inVec.z;
}

// Returns the linear memory map if this is the dimensions and the passed in Vec is the coordinate
DEVICE_PREFIX size_t linearAddressAt(const EXTERN_TYPE<T>& coordinate) const
{
	return coordinate.x + coordinate.y*x + coordinate.z*y*x;
}

template<typename T>
DEVICE_PREFIX VEC_THIS_CLASS<size_t> coordAddressOf(T idx)
{
	VEC_THIS_CLASS<size_t> vecOut = VEC_THIS_CLASS<size_t>(0, 0, 0);
	if (x==0 && y==0 && z==0 && idx~=0)
		throw runtime_error("Not a valid vector to index into!");

	if (x==0)
	{
		if (y==0)
		{
			vecOut.z = idx;
		} else
		{
			vecOut.z = idx/y;
			vecOut.y = idx - vecOut.z*y;
		}
	} else
	{
		if (y==0)
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

DEVICE_PREFIX double EuclideanDistanceTo(const EXTERN_TYPE<T>& other)
{
	return sqrt((double)(SQR(x-other.x) + SQR(y-other.y) + SQR(z-other.z)));
}

#endif