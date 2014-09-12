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
#include <estd/iterators.hpp>
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

	typedef estd::RandomAccessIterator< array<T> > iterator;
	typedef const estd::RandomAccessIterator< const array<T> > const_iterator;

private:
	size_type n;
	device_ptr<T> deviceMemory; // video card memory

public:
	__host__ array( const size_type n=0, const_reference value = T() ) : n(n) {
		if( n ) {
			CUDA_CALL( cudaMalloc( reinterpret_cast<void**>(&deviceMemory.get()), n*sizeof(T) ) );
			std::vector<T> v( n, value );
			CUDA_CALL( cudaMemcpy( deviceMemory.get(), &v[0], n*sizeof(T), cudaMemcpyHostToDevice ) );
		}
	}
	__host__ array( const array<T>& src ) : n(src.n) {
		if( n ) {
			CUDA_CALL( cudaMalloc( reinterpret_cast<void**>(&deviceMemory.get()), n*sizeof(T) ) );
			CUDA_CALL( cudaMemcpy( deviceMemory.get(), src.deviceMemory.get(), n*sizeof(T), cudaMemcpyDeviceToDevice ) );
		}
	}
	__host__ array( const T* sourcePtr, const size_type n=0 ) : n(n) {
		if( n ) {
			CUDA_CALL( cudaMalloc( reinterpret_cast<void**>(&deviceMemory.get()), n*sizeof(T) ) );
			CUDA_CALL( cudaMemcpy( deviceMemory.get(), sourcePtr, n*sizeof(T), cudaMemcpyHostToDevice ) );
		}
	}

	__host__ virtual ~array() {}

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

};

} // namespace ecuda

#endif
