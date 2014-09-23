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
#include "global.hpp"
#include "memory.hpp"

namespace ecuda {

///
/// A video memory-bound array structure.
///
template<typename T>
class array {

public:
	typedef T value_type; //!< cell data type
	typedef std::size_t size_type; //!< index data type
	typedef std::ptrdiff_t difference_type; //!<
	typedef value_type& reference; //!< cell reference type
	typedef const value_type& const_reference; //!< cell const reference type
	typedef value_type* pointer; //!< cell pointer type
	typedef const value_type* const_pointer; //!< cell const pointer type

	typedef ecuda::RandomAccessIterator< array<value_type>, pointer > iterator; //!< iterator type
	typedef ecuda::RandomAccessIterator< const array<value_type>, const_pointer > const_iterator; //!< const iterator type

private:
	size_type n; //!< size of array
	device_ptr<T> deviceMemory; //!< smart point to video card memory

public:
	HOST DEVICE array( const size_type n=0, const_reference value = T() ) : n(n) {
		#ifndef __CUDA_ARCH__
		if( n ) {
			CUDA_CALL( cudaMalloc( deviceMemory.alloc_ptr(), n*sizeof(T) ) );
			std::vector<T> v( n, value );
			CUDA_CALL( cudaMemcpy( deviceMemory.get(), &v[0], n*sizeof(T), cudaMemcpyHostToDevice ) );
		}
		#endif
	}
	HOST array( const array<T>& src ) : n(src.n), deviceMemory(src.deviceMemory) {}
	HOST array( const std::vector<T>& src ) : n(src.size()) {
		if( n ) {
			CUDA_CALL( cudaMalloc( deviceMemory.alloc_ptr(), n*sizeof(T) ) );
			CUDA_CALL( cudaMemcpy( deviceMemory.get(), &src[0], n*sizeof(T), cudaMemcpyHostToDevice ) );
		}
	}

	HOST DEVICE virtual ~array() {}

	DEVICE inline reference at( size_type index ) { return deviceMemory[index]; }
	DEVICE inline reference operator[]( size_type index ) { return deviceMemory[index]; }
	DEVICE inline const_reference at( size_type index ) const { return deviceMemory[index]; }
	DEVICE inline const_reference operator[]( size_type index ) const { return deviceMemory[index]; }

	DEVICE inline reference front() { return *deviceMemory; }
	DEVICE inline reference back() { return operator[]( size()-1 ); }
	DEVICE inline const_reference front() const { return *deviceMemory; }
	DEVICE inline const_reference back() const { return operator[]( size()-1 ); }

	HOST DEVICE inline size_type size() const { return n; }

	DEVICE inline iterator begin() { return iterator(this); }
	DEVICE inline iterator end() { return iterator(this,size()); }
	DEVICE inline const_iterator begin() const { return const_iterator(this); }
	DEVICE inline const_iterator end() const { return const_iterator(this,size()); }

	template<class Alloc>
	HOST const array<T>& operator>>( std::vector<T,Alloc>& vector ) const {
		vector.resize( n );
		CUDA_CALL( cudaMemcpy( &vector[0], deviceMemory.get(), n*sizeof(T), cudaMemcpyDeviceToHost ) );
		return *this;
	}

	HOST array<T>& operator<<( std::vector<T>& vector ) {
		if( size() < vector.size() ) throw std::out_of_range( "ecuda::array is not large enough to fit contents of provided std::vector" );
		CUDA_CALL( cudaMemcpy( deviceMemory.get(), &vector[0], vector.size()*sizeof(T), cudaMemcpyHostToDevice ) );
		return *this;
	}

	DEVICE array<T>& operator=( const array<T>& other ) {
		n = other.n;
		deviceMemory = other.deviceMemory;
		return *this;
	}

};

} // namespace ecuda

#endif
