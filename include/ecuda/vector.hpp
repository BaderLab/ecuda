//----------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// vector.hpp
// An STL-like structure that resides in video memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_VECTOR_HPP
#define ECUDA_VECTOR_HPP

#include <cstddef>
#include <limits>
#include <vector>
#include "iterators.hpp"
#include "global.hpp"
#include "memory.hpp"

namespace ecuda {

///
/// A video memory-bound vector structure.
///
template<typename T>
class vector {

public:
	typedef T value_type; //!< cell data type
	typedef std::size_t size_type; //!< index data type
	typedef std::ptrdiff_t difference_type; //!<
	typedef value_type& reference; //!< cell reference type
	typedef const value_type& const_reference; //!< cell const reference type
	typedef value_type* pointer; //!< cell pointer type
	typedef const value_type* const_pointer; //!< cell const pointer type

	typedef ecuda::RandomAccessIterator< vector<T> > iterator; //!< iterator type
	typedef const ecuda::RandomAccessIterator< const vector<T> > const_iterator; //!< const iterator type

private:
	size_type n; //!< size of array
	size_type m; //!< size of memory allocated
	device_ptr<T> deviceMemory; //!< smart point to video card memory

private:
	HOST void growMemory() {
		deviceMemory newMemory;
		// allocate larger chunk
		CUDA_CALL( cudaMalloc( newMemory.alloc_ptr(), m*2*sizeof(T) ) );
		// copy old data to new chunk
		CUDA_CALL( cudaMemcpy( newMemory.get(), deviceMemory.get(), m*sizeof(T), cudaMemcpyDeviceToDevice ) );
		deviceMemory = newMemory;
		m <<= 1;
	}

public:
	HOST DEVICE vector( size_type n=0, const_reference value = T() ) : n(n) {
		m = 1; while( m < n ) m <<= 1;
		#ifndef __CUDA_ARCH__
		if( n ) {
			CUDA_CALL( cudaMalloc( deviceMemory.alloc_ptr(), m*sizeof(T) ) );
			std::vector<T> v( n, value );
			CUDA_CALL( cudaMemcpy( deviceMemory.get(), &v[0], m*sizeof(T), cudaMemcpyHostToDevice ) );
		}
		#endif
	}
	HOST vector( const vector<T>& src ) : n(src.n), m(src.m), deviceMemory(src.deviceMemory) {}
	HOST vector( const std::vector<T>& src ) : n(src.size()) {
		m = 1; while( m < n ) m <<= 1;
		CUDA_CALL( cudaMalloc( deviceMemory.alloc_ptr(), m*sizeof(T) ) );
		if( n ) CUDA_CALL( cudaMemcpy( deviceMemory.get(), &src[0], n*sizeof(T), cudaMemcpyHostToDevice ) );
	}

	HOST DEVICE virtual ~vector() {}

	DEVICE inline reference at( size_type index ) { return deviceMemory[index]; }
	DEVICE inline reference operator[]( size_type index ) { return deviceMemory[index]; }
	DEVICE inline const_reference at( size_type index ) const { return deviceMemory[index]; }
	DEVICE inline const_reference operator[]( size_type index ) const { return deviceMemory[index]; }

	DEVICE inline reference front() { return *deviceMemory; }
	DEVICE inline reference back() { return operator[]( size()-1 ); }
	DEVICE inline const_reference front() const { return *deviceMemory; }
	DEVICE inline const_reference back() const { return operator[]( size()-1 ); }

	HOST DEVICE inline size_type size() const { return n; }
	HOST DEVICE inline size_type capacity() const { return m; }
	HOST DEVICE bool empty() const { return !n; }

	HOST void push_back( const value_type& v ) {
		if( n == m ) growMemory();
		CUDA_CALL( cudaMemcpy( deviceMemory.get()+(n*sizeof(T)), &v, sizeof(v), cudaMemcpyHostToDevice ) );
		++n;
	}
	HOST DEVICE void pop_back() { if( n ) --n; }

	DEVICE inline iterator begin() { return iterator(*this); }
	DEVICE inline iterator end() { return iterator(*this,size()); }
	DEVICE inline const_iterator begin() const { return const_iterator(*this); }
	DEVICE inline const_iterator end() const { return const_iterator(*this,size()); }

	template<class Alloc>
	HOST const vector<T>& operator>>( std::vector<T,Alloc>& vector ) const {
		vector.resize( n );
		CUDA_CALL( cudaMemcpy( &vector[0], deviceMemory.get(), n*sizeof(T), cudaMemcpyDeviceToHost ) );
		return *this;
	}

	HOST vector<T>& operator<<( std::vector<T>& vector ) {
		if( size() < vector.size() ) throw std::out_of_range( "ecuda::array is not large enough to fit contents of provided std::vector" );
		CUDA_CALL( cudaMemcpy( deviceMemory.get(), &vector[0], vector.size()*sizeof(T), cudaMemcpyHostToDevice ) );
		return *this;
	}

	DEVICE vector<T>& operator=( const vector<T>& other ) {
		n = other.n;
		deviceMemory = other.deviceMemory;
		return *this;
	}

};

} // namespace ecuda

#endif
