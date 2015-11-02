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
// Wrappers around CUDA API functions.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_APIWRAPPERS_HPP
#define ECUDA_APIWRAPPERS_HPP

#include "global.hpp"
#include "allocators.hpp"

#ifdef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
#include <algorithm>
#endif

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
inline cudaError_t cudaMemcpy( T* dest, const T* src, const size_t count, cudaMemcpyKind kind ) {
	#ifdef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
	std::copy( src, src+count, dest );
	return true;
	#else
	return cudaMemcpy( reinterpret_cast<void*>(dest), reinterpret_cast<const void*>(src), sizeof(T)*count, kind );
	#endif
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
inline cudaError_t cudaMemcpy2D( T* dest, const size_t dpitch, const T* src, const size_t spitch, const size_t width, const size_t height, cudaMemcpyKind kind ) {
	#ifdef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
	const char* csrc = reinterpret_cast<const char*>(src);
	char* cdest = reinterpret_cast<char*>(dest);
	for( size_t i = 0; i < height; ++i, csrc += spitch, cdest += dpitch ) {
		std::copy( csrc, csrc+(width*sizeof(T)), cdest );
	}
	return true;
	#else
	return cudaMemcpy2D( reinterpret_cast<void*>(dest), dpitch, reinterpret_cast<const void*>(src), spitch, width*sizeof(T), height, kind );
	#endif
}


///
/// \brief Re-implementation of CUDA API function cudaMemset that allows for any data type.
///
/// The CUDA API cudaMemset function allows only a single-byte value to be specified. This
/// implementation allows any arbitrary data type and value to be specified. However, the
/// underlying call is to cudaMemcpy since a staging block of memory is first filled with the
/// value and then transfered to the device. Thus, this function is more general but takes
/// some unspecified performance hit.
///
/// \param devPtr Pointer to device memory.
/// \param value Value to set for each element.
/// \param count The number of elements to set.
/// \return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
///
template<typename T>
inline cudaError_t cudaMemset( T* devPtr, const T& value, const size_t count ) {
	#ifdef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
	std::fill( devPtr, devPtr+count, value );
	return true;
	#else
	//TODO: may want to implement logic to limit the size of the staging memory, and do the fill in chunks if count is too large
	std::vector< T, host_allocator<T> > v( count, value );
	return cudaMemcpy<T>( devPtr, &v.front(), count, cudaMemcpyHostToDevice );
	#endif
}

///
/// \brief Re-implementation of CUDA API function cudaMemset2D that allows for any data type.
///
/// The CUDA API cudaMemset2D function allows only a single-byte value to be specified. This
/// implementation allows any arbitrary data type and value to be specified. However, the
/// underlying call is to cudaMemcpy since a staging block of memory is first filled with the
/// value and then transfered to the device. Thus, this function is more general but takes
/// some unspecified performance hit.
///
/// \param devPtr Pointer to 2D device memory.
/// \param pitch Pitch in bytes of 2D device memory.
/// \param value Value to set for each element.
/// \param width Width of matrix.
/// \param height Height of matrix.
/// \return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
///
template<typename T>
cudaError_t cudaMemset2D( T* devPtr, const size_t pitch, const T& value, const size_t width, const size_t height ) {
	#ifdef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
	for( size_t i = 0; i < height; ++i ) {
		std::fill( devPtr, devPtr+width, value );
		devPtr = reinterpret_cast<T*>(const_cast<char*>(devPtr)+pitch);
	}
	return true;
	#else
	std::vector< T, host_allocator<T> > v( width, value );
	char* charPtr = reinterpret_cast<char*>(devPtr);
	for( std::size_t i = 0; i < height; ++i, charPtr += pitch ) {
		const cudaError_t rc = cudaMemcpy<T>( reinterpret_cast<T*>(charPtr), &v.front(), width, cudaMemcpyHostToDevice );
		if( rc != cudaSuccess ) return rc;
	}
	return cudaSuccess;
	#endif
}

} // namespace ecuda

#endif
