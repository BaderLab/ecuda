//----------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// array.hpp
// An STL-like structure that resides in video memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ARRAY_HPP
#define ECUDA_ARRAY_HPP

#include <cstddef>
#include <limits>
#include <vector>
#include "iterators.hpp"
//#include <estd/iterators.hpp>
#include "global.hpp"
#include "memory.hpp"

namespace ecuda {

template<typename T>
class array {

public:
	typedef T value_type;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef value_type* pointer;
	typedef const value_type* const_pointer;

	typedef ecuda::RandomAccessIterator< array<T> > iterator;
	typedef const ecuda::RandomAccessIterator< const array<T> > const_iterator;
/*
public:
	struct DevicePayload {
		size_type n;
		device_ptr<T> deviceMemory;
	};
*/
private:
	size_type n;
	device_ptr<T> deviceMemory; // video card memory

public:
	__host__ __device__ array( const size_type n=0, const_reference value = T() ) : n(n) {
		#ifndef __CUDA_ARCH__
		if( n ) {
			CUDA_CALL( cudaMalloc( deviceMemory.alloc_ptr(), n*sizeof(T) ) );
			std::vector<T> v( n, value );
			CUDA_CALL( cudaMemcpy( deviceMemory.get(), &v[0], n*sizeof(T), cudaMemcpyHostToDevice ) );
		}
		#endif
	}
	__host__ array( const array<T>& src ) : n(src.n), deviceMemory(src.deviceMemory) {
		std::cerr << "performing inplace copy" << std::endl;
/*
		if( n ) {
			CUDA_CALL( cudaMalloc( deviceMemory.alloc_ptr(), n*sizeof(T) ) );
			CUDA_CALL( cudaMemcpy( deviceMemory.get(), src.deviceMemory.get(), n*sizeof(T), cudaMemcpyDeviceToDevice ) );
		}
*/
	}
	__host__ array( const std::vector<T>& src ) : n(src.size()) {
		if( n ) {
			CUDA_CALL( cudaMalloc( deviceMemory.alloc_ptr(), n*sizeof(T) ) );
			CUDA_CALL( cudaMemcpy( deviceMemory.get(), &src[0], n*sizeof(T), cudaMemcpyHostToDevice ) );
		}
	}
	/*
	__host__ array( const T* sourcePtr, const size_type n=0 ) : n(n) {
		if( n ) {
			CUDA_CALL( cudaMalloc( deviceMemory.alloc_ptr(), n*sizeof(T) ) );
			CUDA_CALL( cudaMemcpy( deviceMemory.get(), sourcePtr, n*sizeof(T), cudaMemcpyHostToDevice ) );
		}
	}
	*/

	//__device__ array( const DevicePayload& devicePayload ) : n(devicePayload.n), deviceMemory(devicePayload.deviceMemory) {}

	__host__ __device__ virtual ~array() {}

	__device__ reference at( size_type index ) { return deviceMemory[index]; }
	__device__ reference operator[]( size_type index ) { return deviceMemory[index]; }
	__device__ const_reference at( size_type index ) const { return deviceMemory[index]; }
	__device__ const_reference operator[]( size_type index ) const { return deviceMemory[index]; }

	__device__ reference front() { return *deviceMemory; }
	__device__ reference back() { return operator[]( size()-1 ); }
	__device__ const_reference front() const { return *deviceMemory; }
	__device__ const_reference back() const { return operator[]( size()-1 ); }

	__host__ __device__ size_type size() const { return n; }

	__device__ iterator begin() { return iterator(*this); }
	__device__ iterator end() { return iterator(*this,size()); }
	__device__ const_iterator begin() const { return const_iterator(*this); }
	__device__ const_iterator end() const { return const_iterator(*this,size()); }

	__host__ const array<T>& operator>>( std::vector<T>& vector ) const {
		vector.resize( n );
std::cerr << "copy destination = " << deviceMemory.get() << std::endl;
std::cerr << "copying device contents of size " << n << " [" << (n*sizeof(T)) << "] to host vector" << std::endl;
		CUDA_CALL( cudaMemcpy( &vector[0], deviceMemory.get(), n*sizeof(T), cudaMemcpyDeviceToHost ) );
for( size_t i = 0; i < n; ++i ) std::cerr << "IN SITU [" << i << "]=" << vector[i] << std::endl;
		return *this;
	}

	__device__ array<T>& operator=( const array<T>& other ) {
		n = other.n;
		deviceMemory = other.deviceMemory;
		return *this;
	}

	/*
	DevicePayload passToDevice() {
		DevicePayload payload;
		payload.n = n;
		payload.deviceMemory = deviceMemory;
		return payload;
	}
	*/

};

} // namespace ecuda

#endif
