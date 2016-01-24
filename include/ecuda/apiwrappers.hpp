/*
Copyright (c) 2014-2015, Scott Zuyderduyn
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
*/

//----------------------------------------------------------------------------
// apiwrappers.hpp
//
// Wrappers around CUDA C API functions.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_APIWRAPPERS_HPP
#define ECUDA_APIWRAPPERS_HPP

#include "global.hpp"
#include "allocators.hpp" // for host_allocator

#include <vector>

namespace ecuda {

///
/// \brief Wrapper around CUDA API function cudaMemcpy.
///
/// Copies a contiguous block of memory holding count elements of type T to another
/// contiguous block of memory.
///
/// \param dest Pointer to destination memory.
/// \param src Pointer to source memory.
/// \param count Number of elements to copy.
/// \param kind Type of transfer (cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyHostToDevice)
/// \return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
///
template<typename T>
inline cudaError_t cudaMemcpy( T* dest, const T* src, const size_t count, cudaMemcpyKind kind )
{
	return ::cudaMemcpy( reinterpret_cast<void*>(dest), reinterpret_cast<const void*>(src), sizeof(T)*count, kind );
}

///
/// \brief Wrapper around CUDA API function cudaMemcpy2D.
///
/// Copies a matrix of width*height elements of type T from a contiguous memory block
/// with a given pitch (in bytes) to another contiguous memory block with a given
/// pitch (in bytes).
///
/// \param dest Pointer to destination memory.
/// \param dpitch Pitch (in bytes) of destination memory.
/// \param src Pointer to source memory.
/// \param spitch Pitch (in bytes) of source memory.
/// \param width Width of matrix.
/// \param height Height of matrix.
/// \param kind Type of transfer (cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyHostToDevice)
/// \return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
///
template<typename T>
inline cudaError_t cudaMemcpy2D( T* dest, const size_t dpitch, const T* src, const size_t spitch, const size_t width, const size_t height, cudaMemcpyKind kind )
{
	return ::cudaMemcpy2D( reinterpret_cast<void*>(dest), dpitch, reinterpret_cast<const void*>(src), spitch, width*sizeof(T), height, kind );
}

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

///
/// \brief Checks if each byte of a (possibly) multibyte value are the same.
///
/// This is used to see if a multibyte value is represented by a concatentation of
/// the same single byte value.
///
/// \param value the value to check the byte equality status of
/// \return true if each byte in the value is equal
///
template<typename T>
bool is_equal_bytes( const T& value )
{
	const char* p = reinterpret_cast<const char*>(&value);
	const char* q = p; ++q;
	for( int i = 1; i < sizeof(T); ++i, ++q ) if( *p != *q ) return false;
	return true;
}

} // namespace impl
/// \endcond

///
/// \brief Re-implementation of CUDA API function cudaMemset that enforces a single-byte value.
///
/// This implementation simply calls the CUDA API cudaMemset function since the value argument
/// is explicitly stated as single byte.
///
/// \param devPtr Pointer to device memory.
/// \param value Value to set for each element.
/// \param count The number of elements to set.
/// \return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
///
inline cudaError_t cudaMemset( char* devPtr, const char& value, const size_t count )
{
	return ::cudaMemset( static_cast<void*>(devPtr), static_cast<int>(value), count );
}

///
/// \brief Re-implementation of CUDA API function cudaMemset that allows for any data type.
///
/// The CUDA API cudaMemset function allows only a single-byte value to be specified. This
/// implementation allows any arbitrary data type and value to be specified. The function
/// checks if value is represented by a single byte or, if multibyte, that each byte in the
/// value is the same. If this true, the CUDA API cudaMemset function can be used. If not,
/// then a staging block of host memory is first filled with the value and then copied to
/// the device memory. Thus, this function is more general but keep in mind that there
/// will be a performance hit if the provided value is not represented by a concatentation
/// of the same single byte.
///
/// \param devPtr Pointer to device memory.
/// \param value Value to set for each element.
/// \param count The number of elements to set.
/// \return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
///
template<typename T>
inline cudaError_t cudaMemset( T* devPtr, const T& value, const size_t count )
{
	//TODO: may want to implement logic to limit the size of the staging memory, and do the fill in chunks if count is too large
	if( impl::is_equal_bytes(value) ) {
		return cudaMemset( reinterpret_cast<char*>(devPtr), *reinterpret_cast<const char*>(&value), count*sizeof(T) );
	}
	std::vector< T, host_allocator<T> > v( count, value );
	return cudaMemcpy<T>( devPtr, &v.front(), count, cudaMemcpyHostToDevice );
}

///
/// \brief Re-implementation of CUDA API function cudaMemset2D that enforces a single-byte value.
///
/// This implementation simply calls the CUDA API cudaMemset2D function since the value argument
/// is explicitly stated as single byte.
///
/// \param devPtr Pointer to 2D device memory.
/// \param pitch Pitch in bytes of 2D device memory.
/// \param value Value to set for each element.
/// \param width Width of matrix.
/// \param height Height of matrix.
/// \return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
///
inline cudaError_t cudaMemset2D( char* devPtr, const size_t pitch, const char& value, const size_t width, const size_t height )
{
	return ::cudaMemset2D( static_cast<void*>(devPtr), pitch, static_cast<int>(value), width, height );
}

///
/// \brief Re-implementation of CUDA API function cudaMemset2D that allows for any data type.
///
/// The CUDA API cudaMemset2D function allows only a single-byte value to be specified. This
/// implementation allows any arbitrary data type and value to be specified. The function
/// checks if value is represented by a single byte or, if multibyte, that each byte in the
/// value is the same. If this true, the CUDA API cudaMemset2D function can be used. If not,
/// then a staging block of host memory is first filled with the value and then copied to
/// the device memory. Thus, this function is more general but keep in mind that there
/// will be a performance hit if the provided value is not represented by a concatentation
/// of the same single byte.
///
/// \param devPtr Pointer to 2D device memory.
/// \param pitch Pitch in bytes of 2D device memory.
/// \param value Value to set for each element.
/// \param width Width of matrix.
/// \param height Height of matrix.
/// \return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
///
template<typename T>
cudaError_t cudaMemset2D( T* devPtr, const size_t pitch, const T& value, const size_t width, const size_t height )
{
	if( impl::is_equal_bytes(value) ) {
		return cudaMemset2D( reinterpret_cast<char*>(devPtr), pitch, *reinterpret_cast<const char*>(&value), width*sizeof(T), height );
	}
	std::vector< T, host_allocator<T> > v( width, value );
	char* charPtr = reinterpret_cast<char*>(devPtr);
	for( std::size_t i = 0; i < height; ++i, charPtr += pitch ) {
		const cudaError_t rc = cudaMemcpy<T>( reinterpret_cast<T*>(charPtr), &v.front(), width, cudaMemcpyHostToDevice );
		if( rc != cudaSuccess ) return rc;
	}
	return cudaSuccess;
}

template<typename T>
inline cudaError_t cudaMemcpyToSymbol( T* dest, const T* src, size_t count=1, size_t offset=0, enum cudaMemcpyKind kind=cudaMemcpyHostToDevice )
{
	return ::cudaMemcpyToSymbol( reinterpret_cast<const char*>(dest), reinterpret_cast<const void*>(src), count*sizeof(T), offset, kind );
}

template<typename T>
inline cudaError_t cudaMemcpyToSymbol( T& dest, const T& src, enum cudaMemcpyKind kind=cudaMemcpyHostToDevice )
{
	return ::ecuda::cudaMemcpyToSymbol( &dest, &src, 1, 0, kind );
}

/*
 * This is here because of a bizarre compiler bug in nvcc 5.5.
 * If __threadfence() is called inline (i.e. the at() methods of
 * each container), then nvcc complains about not knowing about it.
 *
 * Example compiler message: error: there are no arguments to ‘__threadfence’ that depend on a template parameter, so a declaration of ‘__threadfence’ must be available
 *
 * In CUDA >=6.0 the same code compiles fine. If we do below and just wrap the __threadfence()
 * call in it's own function then it works in all versions.
 */
inline __DEVICE__ void threadfence() { __threadfence(); }

} // namespace ecuda

#endif
