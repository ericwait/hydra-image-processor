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
DEVICE_PREFIX size_t linearAddressAt(const EXTERN_TYPE<T>& coordinate, bool columnMajor=false) const
{
	if (columnMajor)
		return coordinate.y + coordinate.x*y + coordinate.z*x*y;

	return coordinate.x + coordinate.y*x + coordinate.z*y*x;
}

DEVICE_PREFIX double EuclideanDistanceTo(const EXTERN_TYPE<T>& other)
{
	return sqrt((double)(SQR(x-other.x) + SQR(y-other.y) + SQR(z-other.z)));
}

#endif