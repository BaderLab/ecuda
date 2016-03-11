/*
Copyright (c) 2015, Scott Zuyderduyn
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
// impl/host_emulation.hpp
//
// Included when host/CPU-only emulation is desired.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_IMPL_HOST_IMPLEMENTATION_HPP
#define ECUDA_IMPL_HOST_IMPLEMENTATION_HPP

/// \cond DEVELOPER_DOCUMENTATION
///
/// These are CUDA API calls that have been reimplemented to use host memory
/// only. If code isn't compiled by nvcc then these functions are called
/// instead.
///
#ifndef __CUDACC__

#include <algorithm>
#include <memory>
#include <ctime>

#define __global__
#define __device__
#define __host__
#define __constant__

enum cudaError_t
{
	cudaSuccess
};

enum cudaMemcpyKind {
	cudaMemcpyDeviceToDevice,
	cudaMemcpyDeviceToHost,
	cudaMemcpyHostToDevice
};

cudaError_t cudaFree( void* devPtr )
{
	delete [] reinterpret_cast<char*>(devPtr); // TODO: does this work as expected?
	return cudaSuccess;
}

inline cudaError_t cudaFreeHost( void* devPtr ) { return cudaFree( devPtr ); }

void cudaSetDevice( int ) {}

cudaError_t cudaMalloc( void** devPtr, size_t size )
{
	*devPtr = std::allocator<char>().allocate( size );
	return cudaSuccess;
}

#define cudaHostAllocDefault       0x00
#define cudaHostAllocPortable      0x01
#define cudaHostAllocMapped        0x02
#define cudaHostAllocWriteCombined 0x04

inline cudaError_t cudaHostAlloc( void** ptr, size_t size, unsigned flags = 0 ) { return cudaMalloc( ptr, size ); }

cudaError_t cudaMallocPitch( void** devPtr, size_t* pitch, size_t width, size_t height )
{
	*pitch = width;
	*pitch += (*pitch % 16); // add padding to get 128-bit memory alignment (16 bytes)
	if( ( width % *pitch ) == 0 ) {
		++(*pitch); // just add a byte to get some padding
		//std::cerr << "WARNING: Host emulation of cudaMallocPitch allocated the equivalent to a contiguous block and so is a poor test of API logic for pitched memory." << std::endl;
	}
	*devPtr = std::allocator<char>().allocate( (*pitch)*height );
	return cudaSuccess;
}

cudaError_t cudaMemcpy( void* dst, const void* src, size_t count, cudaMemcpyKind )
{
	std::copy( reinterpret_cast<const char*>(src), reinterpret_cast<const char*>(src)+count, reinterpret_cast<char*>(dst) );
	return cudaSuccess;
}

cudaError_t cudaMemcpy2D( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind )
{
	char* pDst = reinterpret_cast<char*>(dst);
	const char* pSrc = reinterpret_cast<const char*>(src);
	for( size_t i = 0; i < height; ++i, pDst += dpitch, pSrc += spitch ) std::copy( pSrc, pSrc+width, pDst );
	return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbol( const void* dest, const void* src, size_t count, size_t offset = 0, cudaMemcpyKind = cudaMemcpyHostToDevice )
{
	char* pDst = const_cast<char*>(reinterpret_cast<const char*>(dest));
	pDst += offset;
	const char* pSrc = reinterpret_cast<const char*>(src);
	std::copy( pSrc, pSrc+count, pDst );
	return cudaSuccess;
}

cudaError_t cudaMemset( void* devPtr, int value, size_t count )
{
	char* p = static_cast<char*>(devPtr);
	for( size_t i = 0; i < count; ++i, ++p ) *p = static_cast<char>(value);
	return cudaSuccess;
}

cudaError_t cudaMemset2D( void* devPtr, size_t pitch, int value, size_t width, size_t height )
{
	char* p = static_cast<char*>(devPtr);
	for( std::size_t i = 0; i < height; ++i ) {
		for( std::size_t j = 0; j < pitch; ++j, ++p ) if( j < width ) *p = static_cast<char>(value);
	}
	return cudaSuccess;
}

namespace impl {

struct cudaEvent
{
	std::clock_t time;
};

} // namespace impl

typedef impl::cudaEvent* cudaEvent_t;

typedef int cudaStream_t;

cudaError_t cudaEventCreate( cudaEvent_t* event )
{
	*event = new impl::cudaEvent;
	return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags( cudaEvent_t* event, unsigned ) { return cudaEventCreate(event); }

cudaError_t cudaEventRecord( cudaEvent_t event, cudaStream_t = 0 )
{
	event->time = std::clock();
	return cudaSuccess;
}

cudaError_t cudaEventQuery( cudaEvent_t ) { return cudaSuccess; }

cudaError_t cudaEventDestroy( cudaEvent_t event )
{
	if( event ) delete event;
	return cudaSuccess;
}

cudaError_t cudaEventSynchronize( cudaEvent_t event ) { return cudaSuccess; }

cudaError_t cudaEventElapsedTime( float* ms, cudaEvent_t start, cudaEvent_t end )
{
	*ms = static_cast<double>( end->time - start->time ) / static_cast<double>(CLOCKS_PER_SEC) * static_cast<double>(1000);
	return cudaSuccess;
}

struct cudaDeviceProp {};

cudaError_t cudaGetDeviceProperties( cudaDeviceProp*, int ) { return cudaSuccess; }

cudaError_t cudaDriverGetVersion( int* driverVersion ) { *driverVersion = 0; return cudaSuccess; }

cudaError_t cudaRuntimeGetVersion( int* runtimeVersion ) { *runtimeVersion = 0; return cudaSuccess; }

cudaError_t cudaGetDeviceCount( int* count ) { *count = 0; return cudaSuccess; }

#endif // __CUDACC__
/// \endcond

#endif // ECUDA_IMPL_HOST_IMPLEMENTATION_HPP
