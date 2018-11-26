#pragma once

template <typename T>
void toDevice(T** dst, T* src, std::size_t length)
{
	*dst = src;
}

template <typename T, typename U>
void toDevice(T** dst, U* src, std::size_t length)
{
	T* temp = new T[length];
	for (std::size_t i = 0; i < length; ++i)
		temp[i] = (T)(src[i]);

	*dst = temp;
}

template <typename T>
void fromDevice(T** dst, T** src, std::size_t length) 
{
	*dst = *src;
}

template <typename T, typename U>
void fromDevice(T** dst, U* src, std::size_t length)
{
	*dst = new T[length]; 
}

template <typename T>
void copyBuffer(T** dst, T** src, std::size_t length)
{
	*dst = *src; 
}

template <typename T, typename U>
void copyBuffer(T** dst, U** src, std::size_t length)
{
	for (std::size_t i = 0; i < length; ++i)
		(*dst)[i] = (T)((*src)[i]);

	U* toDelete = *src;
	delete[] toDelete;
}

template <typename T>
void cleanBuffer(T** buff, T* src) {}

template <typename T, typename U>
void cleanBuffer(T** buff, U* src)
{
	T* toDelete = *buff;
	delete[] toDelete;
}
