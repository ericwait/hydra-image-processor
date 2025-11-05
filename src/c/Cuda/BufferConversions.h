/**
 * @file BufferConversions.h
 * @brief Template functions for buffer type conversions and memory management
 *
 * This file provides template functions for converting between different buffer types,
 * managing memory transfers, and cleaning up allocated buffers. These are used for
 * type conversions when interfacing with different scripting languages.
 */

#pragma once

/**
 * @brief Copies buffer pointer when types match (no conversion needed)
 *
 * @tparam T The buffer element type
 * @param dst Output parameter that receives the source pointer
 * @param src Source buffer pointer
 * @param length Number of elements in the buffer (unused in this specialization)
 */
template <typename T>
void toDevice(T** dst, T* src, std::size_t length)
{
	*dst = src;
}

/**
 * @brief Converts and copies buffer when types differ
 *
 * Allocates a new buffer and converts each element from type U to type T.
 *
 * @tparam T The destination buffer element type
 * @tparam U The source buffer element type
 * @param dst Output parameter that receives the pointer to the new converted buffer
 * @param src Source buffer pointer
 * @param length Number of elements to convert
 */
template <typename T, typename U>
void toDevice(T** dst, U* src, std::size_t length)
{
	T* temp = new T[length];
	for (std::size_t i = 0; i < length; ++i)
		temp[i] = (T)(src[i]);

	*dst = temp;
}

/**
 * @brief Transfers buffer pointer when types match (no conversion needed)
 *
 * @tparam T The buffer element type
 * @param dst Output parameter that receives the source pointer
 * @param src Pointer to source buffer pointer
 * @param length Number of elements in the buffer (unused in this specialization)
 */
template <typename T>
void fromDevice(T** dst, T** src, std::size_t length)
{
	*dst = *src;
}

/**
 * @brief Allocates destination buffer for type conversion
 *
 * Allocates a new buffer of type T. The actual conversion is expected to be
 * performed by the caller after this allocation.
 *
 * @tparam T The destination buffer element type
 * @tparam U The source buffer element type
 * @param dst Output parameter that receives the pointer to the new allocated buffer
 * @param src Source buffer pointer (unused in this implementation)
 * @param length Number of elements to allocate
 */
template <typename T, typename U>
void fromDevice(T** dst, U* src, std::size_t length)
{
	*dst = new T[length];
}

/**
 * @brief Copies buffer pointer when types match (no conversion needed)
 *
 * @tparam T The buffer element type
 * @param dst Output parameter that receives the source pointer
 * @param src Pointer to source buffer pointer
 * @param length Number of elements in the buffer (unused in this specialization)
 */
template <typename T>
void copyBuffer(T** dst, T** src, std::size_t length)
{
	*dst = *src;
}

/**
 * @brief Converts and copies buffer when types differ, then cleans up source
 *
 * Converts each element from type U to type T and deletes the source buffer.
 *
 * @tparam T The destination buffer element type
 * @tparam U The source buffer element type
 * @param dst Pointer to destination buffer pointer (must be pre-allocated)
 * @param src Pointer to source buffer pointer (will be deleted)
 * @param length Number of elements to convert
 */
template <typename T, typename U>
void copyBuffer(T** dst, U** src, std::size_t length)
{
	for (std::size_t i = 0; i < length; ++i)
		(*dst)[i] = (T)((*src)[i]);

	U* toDelete = *src;
	delete[] toDelete;
}

/**
 * @brief No-op cleanup when buffer types match
 *
 * @tparam T The buffer element type
 * @param buff The buffer pointer (unused)
 * @param src The source pointer (unused)
 */
template <typename T>
void cleanBuffer(T** buff, T* src) {}

/**
 * @brief Cleans up temporary buffer created during type conversion
 *
 * Deletes the buffer that was allocated for type conversion.
 *
 * @tparam T The buffer element type to delete
 * @tparam U The source buffer element type
 * @param buff Pointer to the buffer to delete
 * @param src The original source pointer (unused)
 */
template <typename T, typename U>
void cleanBuffer(T** buff, U* src)
{
	T* toDelete = *buff;
	delete[] toDelete;
}
