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
#include "allocators.hpp"
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

	typedef ecuda::RandomAccessIterator< vector<value_type>, pointer > iterator; //!< iterator type
	typedef ecuda::RandomAccessIterator< const vector<value_type>, const_pointer > const_iterator; //!< const iterator type

private:
	size_type n; //!< size of array
	size_type m; //!< size of memory allocated
	device_ptr<T> deviceMemory; //!< smart point to video card memory

private:
	HOST void growMemory( size_type minimum );

public:
	HOST DEVICE vector( size_type n=0, const_reference value = T() );
	HOST DEVICE vector( const vector<T>& src );
	//HOST vector( const vector<T>& src ) : n(src.n), m(src.m), deviceMemory(src.deviceMemory) {}
	HOST vector( const std::vector<T>& src );
	HOST DEVICE virtual ~vector() {}

	// Iterators:
	HOST DEVICE inline iterator begin() { return iterator(this); }
	HOST DEVICE inline iterator end() { return iterator(this,size()); }
	HOST DEVICE inline const_iterator begin() const { return const_iterator(this); }
	HOST DEVICE inline const_iterator end() const { return const_iterator(this,size()); }

	// Capacity:
	HOST DEVICE inline size_type size() const { return n; }
	HOST DEVICE inline size_type max_size() const { return std::numeric_limits<size_type>::max(); }
	HOST void resize( size_type newSize, value_type& value = value_type() );
	HOST DEVICE inline size_type capacity() const { return m; }
	HOST DEVICE inline bool empty() const { return !n; }
	HOST DEVICE inline void reserve( size_type newSize ) { growMemory(newSize); }

	// Element access:
	DEVICE inline reference operator[]( size_type index ) { return deviceMemory[index]; }
	DEVICE inline const_reference operator[]( size_type index ) const { return deviceMemory[index]; }
	DEVICE inline reference at( size_type index ) { return deviceMemory[index]; }
	DEVICE inline const_reference at( size_type index ) const { return deviceMemory[index]; }
	DEVICE inline reference front() { return *deviceMemory; }
	DEVICE inline reference back() { return operator[]( size()-1 ); }
	DEVICE inline const_reference front() const { return *deviceMemory; }
	DEVICE inline const_reference back() const { return operator[]( size()-1 ); }

	// Modifiers:
	HOST void assign( size_type newSize, value_type& value = value_type() );
	template<class InputIterator>
	HOST void assign( InputIterator first, InputIterator last );
	HOST void push_back( const value_type& v );
	HOST DEVICE inline void pop_back() { if( n ) --n; }

	HOST iterator insert( iterator position, const value_type& val );
	HOST iterator insert( iterator position, const size_type span, const value_type& val );
	template<class InputIterator>
	HOST void insert( iterator position, InputIterator first, InputIterator last );
	HOST iterator erase( iterator position );
	HOST iterator erase( iterator first, iterator last );
	HOST DEVICE void swap( vector& other );
	HOST DEVICE inline void clear() { n = 0; }

	// Conversions from other container types:
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


template<typename T>
HOST void vector<T>::growMemory( size_type minimum ) {
	if( m >= minimum ) return; // no growth neccessary
	size_type m2 = m;
	while( m2 < minimum ) m2 <<= 1;
	device_ptr<T> newMemory;
	// allocate larger chunk
	CUDA_CALL( cudaMalloc( newMemory.alloc_ptr(), m2*sizeof(T) ) );
	// copy old data to new chunk
	CUDA_CALL( cudaMemcpy( newMemory.get(), deviceMemory.get(), m*sizeof(T), cudaMemcpyDeviceToDevice ) );
	deviceMemory = newMemory;
	m = m2;
}

template<typename T>
HOST DEVICE vector<T>::vector( size_type n, const_reference value ) : n(n) {
	m = 1; while( m < n ) m <<= 1;
	#ifndef __CUDA_ARCH__
	if( n ) {
		CUDA_CALL( cudaMalloc( deviceMemory.alloc_ptr(), m*sizeof(T) ) );
		std::vector<T> v( n, value );
		CUDA_CALL( cudaMemcpy( deviceMemory.get(), &v[0], m*sizeof(T), cudaMemcpyHostToDevice ) );
	}
	#endif
}

template<typename T>
HOST DEVICE vector<T>::vector( const vector<T>& src ) : n(src.n), m(src.m) {
	#ifndef __CUDA_ARCH__
	// if on host allocate new memory and copy contents
	if( n ) {
		CUDA_CALL( cudaMalloc( deviceMemory.alloc_ptr(), m*sizeof(T) ) );
		CUDA_CALL( cudaMemcpy( deviceMemory.get(), src.deviceMemory.get(), n*sizeof(T), cudaMemcpyDeviceToDevice ) );
	}
	#else
	// if on device just copy pointer
	deviceMemory = src.deviceMemory;
	#endif
}

template<typename T>
HOST vector<T>::vector( const std::vector<T>& src ) : n(src.size()) {
	m = 1; while( m < n ) m <<= 1;
	CUDA_CALL( cudaMalloc( deviceMemory.alloc_ptr(), m*sizeof(T) ) );
	if( n ) CUDA_CALL( cudaMemcpy( deviceMemory.get(), &src[0], n*sizeof(T), cudaMemcpyHostToDevice ) );
}

template<typename T>
HOST void vector<T>::resize( size_type newSize, value_type& value ) {
	growMemory(newSize); // make sure enough device memory is allocated
	std::vector<T> v( newSize-n, value );
	CUDA_CALL( cudaMemcpy( deviceMemory.get()+n, &v[0], v.size()*sizeof(T), cudaMemcpyHostToDevice ) );
	n = newSize;
}

template<typename T>
HOST vector<T>::iterator vector<T>::insert( vector<T>::iterator position, const value_type& val ) {
	growMemory(n+1); // make sure enough device memory is allocated
	const size_type index = position-begin();
	std::vector< T,HostAllocator<T> > v( n-index+1, T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-insertion region to host
	if( v.size() > 1 ) CUDA_CALL( cudaMemcpy( &v[1], deviceMemory.get()+index, (n-index)*sizeof(T), cudaMemcpyDeviceToHost ) ); // copy post-insertion region to +1 in host vector
	v.front() = val; // add new value to position 0
	CUDA_CALL( cudaMemcpy( deviceMemory.get()+index, &v[0], v.size()*sizeof(T), cudaMemcpyHostToDevice ) ); // copy +1 inserted data back onto device
	++n;
	return position; // since iterators are index based, the return iterator is the same as the one provided
}

template<typename T>
HOST vector<T>::iterator vector<T>::insert( vector<T>::iterator position, const size_type span, const value_type& val ) {
	growMemory(n+span); // make sure enough device memory is allocated
	const size_type index = position-begin();
	std::vector< T,HostAllocator<T> > v( n-index+span, T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-insertion region to host
	if( v.size() > span ) CUDA_CALL( cudaMemcpy( &v[span], deviceMemory.get()+index, (n-index)*sizeof(T), cudaMemcpyDeviceToHost ) ); // copy post-insertion region to +span in host vector
	for( size_type i = 0; i < span; ++i ) v[i] = val; // fill in start region with new value
	CUDA_CALL( cudaMemcpy( deviceMemory.get()+index, &v[0], v.size()*sizeof(T), cudaMemcpyHostToDevice ) ); // copy +span inserted data back onto device
	n += span;
	return position; // since iterators are index based, the return iterator is the same as the one provided
}

template<typename T>
template<class InputIterator>
HOST void vector<T>::insert( vector<T>::iterator position, InputIterator first, InputIterator last ) {
	const std::vector<T> x( first, last );
	growMemory(n+x.size());
	const size_type index = position-begin();
	std::vector< T,HostAllocator<T> > v( n-index+x.size(), T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-insertion region to host
	if( v.size() > x.size() ) CUDA_CALL( cudaMemcpy( &v[x.size()], deviceMemory.get()+index, (n-index)*sizeof(T), cudaMemcpyDeviceToHost ) ); // copy post-insertion region to +span in host vector
	for( size_type i = 0; i < x.size(); ++i ) v[i] = x[i]; // fill in start region with new values
	CUDA_CALL( cudaMemcpy( deviceMemory.get()+index, &v[0], v.size()*sizeof(T), cudaMemcpyHostToDevice ) ); // copy +x.size() inserted data back onto device
	n += x.size();
}

template<typename T>
HOST vector<T>::iterator vector<T>::erase( vector<T>::iterator position ) {
	const size_type index = position-begin();
	std::vector< T,HostAllocator<T> > v( n-index-1, T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-erasure region to host
	CUDA_CALL( cudaMemcpy( &v[0], deviceMemory.get()+(index+1), v.size()*sizeof(T), cudaMemcpyDeviceToHost ) );
	CUDA_CALL( cudaMemcpy( deviceMemory.get()+index, &v[0], v.size()*sizeof(T), cudaMemcpyHostToDevice ) );
	--n;
	return position; // since iterators are index based, the return iterator is the same as the one provided
}

template<typename T>
HOST vector<T>::iterator vector<T>::erase( vector<T>::iterator first, vector<T>::iterator last ) {
	const size_type index1 = first-begin();
	const size_type index2 = last-begin();
	std::vector< T,HostAllocator<T> > v( n-index2, T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-erasure region to host
	CUDA_CALL( cudaMemcpy( &v[0], deviceMemory.get()+index2, v.size()*sizeof(T), cudaMemcpyDeviceToHost ) );
	CUDA_CALL( cudaMemcpy( deviceMemory.get()+index1, &v[0], v.size()*sizeof(T), cudaMemcpyHostToDevice ) );
	n -= index2-index1;
	return last; // since iterators are index based, the return iterator is the same as the one provided
}

template<typename T>
HOST DEVICE void vector<T>::swap( vector<T>& other ) {
	// just swap all members
	{ device_ptr<T> tmp = other.deviceMemory; other.deviceMemory = deviceMemory; deviceMemory = tmp; }
	{ size_type tmp = other.n; other.n = n; n = tmp; }
	{ size_type tmp = other.m; other.m = m; m = tmp; }
}

template<typename T>
HOST void vector<T>::assign( size_type newSize, value_type& value ) {
	growMemory(newSize); // make sure enough device memory is allocated
	std::vector<T> v( newSize, value );
	CUDA_CALL( cudaMemcpy( deviceMemory.get(), &v[0], v.size()*sizeof(T), cudaMemcpyHostToDevice ) );
	n = newSize;
}

template<typename T>
template<class InputIterator>
HOST void vector<T>::assign( InputIterator first, InputIterator last ) {
	std::vector<T> v( first, last );
	growMemory( v.size() ); // make sure enough device memory is allocated
	CUDA_CALL( cudaMemcpy( deviceMemory.get(), &v[0], v.size()*sizeof(T), cudaMemcpyHostToDevice ) );
	n = v.size();
}

template<typename T>
HOST void vector<T>::push_back( const value_type& v ) {
	growMemory(n+1);
	CUDA_CALL( cudaMemcpy( deviceMemory.get()+(n*sizeof(T)), &v, sizeof(v), cudaMemcpyHostToDevice ) );
	++n;
}


} // namespace ecuda

#endif
