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
// vector.hpp
// An STL-like structure that resides in video memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_VECTOR_HPP
#define ECUDA_VECTOR_HPP

#include <cstddef>
#include <iterator>
#include <limits>
#include <vector>

#ifdef __CPP11_SUPPORTED__
#include <initializer_list>
#include <memory>
#include <utility>
#endif

#include "allocators.hpp"
#include "apiwrappers.hpp"
#include "iterators.hpp"
#include "global.hpp"
#include "memory.hpp"

namespace ecuda {

///
/// A video memory-bound vector structure.
///
template< typename T, class Alloc=DeviceAllocator<T> >
class vector {

public:
	typedef T value_type; //!< cell data type
	typedef Alloc allocator_type; //!< allocator type
	typedef std::size_t size_type; //!< unsigned integral type
	typedef std::ptrdiff_t difference_type; //!< signed integer type
	#ifdef __CPP11_SUPPORTED__
	typedef value_type& reference; //!< cell reference type
	typedef const value_type& const_reference; //!< cell const reference type
	typedef typename std::allocator_traits<Alloc>::pointer pointer; //!< cell pointer type
	typedef typename std::allocator_traits<Alloc>::const_pointer const_pointer; //!< cell const pointer type
	#else
	typedef typename Alloc::reference reference; //!< cell reference type
	typedef typename Alloc::const_reference const_reference; //!< cell const reference type
	typedef typename Alloc::pointer pointer; //!< cell pointer type
	typedef typename Alloc::const_pointer const_pointer; //!< cell const pointer type
	#endif

	typedef pointer_iterator<value_type,pointer> iterator; //!< iterator type
	typedef pointer_iterator<const value_type,pointer> const_iterator; //!< const iterator type
	typedef std::reverse_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

private:
	// REMEMBER: n and m altered on device memory won't be reflected on the host object. Don't allow
	//           the device to perform any operations that change their value.
	size_type n; //!< number of elements currently stored
	size_type m; //!< number of elements worth of memory allocated
	device_ptr<T> deviceMemory; //!< smart point to video card memory
	allocator_type allocator;

private:
	HOST void growMemory( size_type minimum );

public:
	HOST explicit vector( const allocator_type& allocator = allocator_type() ) : n(0), m(0), allocator(allocator) {}
	HOST explicit vector( size_type n, const value_type& value, const allocator_type& allocator = allocator_type() );
	HOST explicit vector( size_type n ) : n(n) {
		m = 1; while( m < n ) m <<= 1;
		if( n ) {
			deviceMemory = device_ptr<T>( allocator.allocate( m ) );
			std::vector<T> v( n );
			CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), m, cudaMemcpyHostToDevice ) );
		}
	}
	template<class InputIterator>
	HOST vector( InputIterator begin, InputIterator end, const allocator_type& allocator = allocator_type() ) : allocator(allocator) {
		std::vector<value_type> v( begin, end );
		growMemory( v.size() );
		CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	}
	HOST vector( const vector<T>& src ) : n(src.n), m(src.m), deviceMemory(src.deviceMemory), allocator(src.allocator) {};
	template<class Alloc2>
	HOST vector( const vector<T,Alloc2>& src, const allocator_type& allocator ) : n(src.n), m(src.m), allocator(allocator) {
		if( n ) {
			deviceMemory = device_ptr<T>( this->allocator.allocate( m ) );
			CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), src.deviceMemory.get(), m, cudaMemcpyDeviceToDevice ) );
		}
	}
	#ifdef __CPP11_SUPPORTED__
	HOST DEVICE vector( vector&& src ) : n(src.n), m(src.m), deviceMemory(std::move(src.deviceMemory)), allocator(std::move(src.allocator)) {}
	HOST DEVICE vector( vector&& src, const allocator_type& allocator ) : n(src.n), m(src.m), deviceMemory(std::move(src.deviceMemory)), allocator(allocator) {}
	HOST DEVICE vector( std::initializer_list<value_type> il, const allocator_type& allocator = allocator_type() ) : n(0), m(0), allocator(allocator) {
		std::vector<value_type> v( il );
		growMemory( v.size() );
		CUDA_CALL( cudaMemcpy<value_type>( deviceMemory.get(), v.data(), v.size(), cudaMemcpyHostToDevice ) );
	}
	#endif

	HOST vector( const std::vector<T>& src, const allocator_type& allocator = allocator_type() );
	HOST DEVICE virtual ~vector() {}

	// Iterators:
	HOST DEVICE inline iterator begin() { return iterator(deviceMemory.get()); }
	HOST DEVICE inline iterator end() { return iterator(deviceMemory.get()+size()); }
	HOST DEVICE inline const_iterator begin() const { return const_iterator(deviceMemory.get()); }
	HOST DEVICE inline const_iterator end() const { return const_iterator(deviceMemory.get()+size()); }

	// Capacity:
	HOST DEVICE inline size_type size() const { return n; }
	HOST DEVICE inline size_type max_size() const { return std::numeric_limits<size_type>::max(); }
	HOST void resize( size_type newSize, value_type& value = value_type() );
	HOST DEVICE inline size_type capacity() const { return m; }
	HOST DEVICE inline bool empty() const { return !n; }
	HOST inline void reserve( size_type newSize ) { growMemory(newSize); }

	// Element access:
	DEVICE inline reference operator[]( const size_type index ) { return deviceMemory[index]; }
	DEVICE inline const_reference operator[]( const size_type index ) const { return deviceMemory[index]; }

	/*
	 * Deprecating these function since the STL standard seems to specify that at() accessors
	 * must implement range checking that throws an exception on failure.  Since exceptions are
	 * not supported within a CUDA kernel, this cannot be satisfied.
	 *
	DEVICE inline reference at( const size_type index ) { return deviceMemory[index]; }
	DEVICE inline const_reference at( const size_type index ) const { return deviceMemory[index]; }
	*/

	DEVICE inline reference front() { return *deviceMemory; }
	DEVICE inline reference back() { return operator[]( size()-1 ); }
	DEVICE inline const_reference front() const { return *deviceMemory; }
	DEVICE inline const_reference back() const { return operator[]( size()-1 ); }

	// Modifiers:
	HOST void assign( size_type newSize, value_type& value = value_type() );
	template<class InputIterator>
	HOST void assign( InputIterator first, InputIterator last );
	HOST void push_back( const value_type& v );
	HOST DEVICE inline void pop_back() { if( n ) --n; } // NOTE: if called from device the host instance doesn't change

	HOST iterator insert( iterator position, const value_type& val );
	HOST iterator insert( iterator position, const size_type span, const value_type& val );
	template<class InputIterator>
	HOST void insert( iterator position, InputIterator first, InputIterator last );
	HOST iterator erase( iterator position );
	HOST iterator erase( iterator first, iterator last );
	HOST DEVICE void swap( vector& other );
	HOST DEVICE inline void clear() { n = 0; }

	HOST inline allocator_type get_allocator() const { return allocator_type(); }

	// Conversions from other container types:
	template<class OtherAlloc>
	HOST const vector<T,Alloc>& operator>>( std::vector<T,OtherAlloc>& vector ) const {
		vector.resize( n );
		CUDA_CALL( cudaMemcpy<T>( &vector.front(), deviceMemory.get(), n, cudaMemcpyDeviceToHost ) );
		//CUDA_CALL( cudaMemcpy( &vector[0], deviceMemory.get(), n*sizeof(T), cudaMemcpyDeviceToHost ) );
		return *this;
	}

	HOST vector<T,Alloc>& operator<<( std::vector<T>& vector ) {
		if( size() < vector.size() ) throw std::out_of_range( "ecuda::array is not large enough to fit contents of provided std::vector" );
		CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &vector.front(), vector.size(), cudaMemcpyHostToDevice ) );
		//CUDA_CALL( cudaMemcpy( deviceMemory.get(), &vector[0], vector.size()*sizeof(T), cudaMemcpyHostToDevice ) );
		return *this;
	}

	// critical function used to bridge host->device code
	DEVICE vector<T,Alloc>& operator=( const vector<T>& other ) {
		n = other.n;
		deviceMemory = other.deviceMemory;
		return *this;
	}

};


template<typename T,class Alloc>
HOST void vector<T,Alloc>::growMemory( size_type minimum ) {
	if( m >= minimum ) return; // no growth neccessary
	size_type m2 = m;
	if( !m2 ) m2 = 1; // in case no memory is currently allocated
	while( m2 < minimum ) m2 <<= 1;
	// allocate larger chunk
	device_ptr<T> newMemory( allocator.allocate( m2 ) );
	// copy old data to new chunk
	CUDA_CALL( cudaMemcpy<T>( newMemory.get(), deviceMemory.get(), m, cudaMemcpyDeviceToDevice ) );
	deviceMemory = newMemory;
	m = m2;
}

template<typename T,class Alloc>
HOST vector<T,Alloc>::vector( size_type n, const value_type& value, const allocator_type& allocator ) : n(n), allocator(allocator) {
	m = 1; while( m < n ) m <<= 1;
	#ifndef __CUDA_ARCH__
	if( n ) {
		deviceMemory = device_ptr<T>( this->allocator.allocate( m ) );
		std::vector<T> v( n, value );
		CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), m, cudaMemcpyHostToDevice ) );
	}
	#endif
}

template<typename T,class Alloc>
HOST vector<T,Alloc>::vector( const std::vector<T>& src, const allocator_type& allocator ) : n(src.size()), allocator(allocator) {
	m = 1; while( m < n ) m <<= 1;
	deviceMemory = device_ptr<T>( this->allocator.allocate( m ) );
	if( n ) CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &src.front(), n, cudaMemcpyHostToDevice ) );
}

template<typename T,class Alloc>
HOST void vector<T,Alloc>::resize( size_type newSize, value_type& value ) {
	growMemory(newSize); // make sure enough device memory is allocated
	std::vector<T> v( newSize-n, value );
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+n, &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	n = newSize;
}

template<typename T,class Alloc>
HOST vector<T,Alloc>::iterator vector<T,Alloc>::insert( vector<T,Alloc>::iterator position, const value_type& val ) {
	growMemory(n+1); // make sure enough device memory is allocated
	const size_type index = position-begin();
	std::vector<T,Alloc> v( n-index+1, T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-insertion region to host
	if( v.size() > 1 ) CUDA_CALL( cudaMemcpy<T>( &v[1], deviceMemory.get()+index, (n-index), cudaMemcpyDeviceToHost ) ); // copy post-insertion region to +1 in host vector
	v.front() = val; // add new value to position 0
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+index, &v.front(), v.size(), cudaMemcpyHostToDevice ) ); // copy +1 inserted data back onto device
	++n;
	return position; // since iterators are index based, the return iterator is the same as the one provided
}

template<typename T,class Alloc>
HOST vector<T,Alloc>::iterator vector<T,Alloc>::insert( vector<T,Alloc>::iterator position, const size_type span, const value_type& val ) {
	growMemory(n+span); // make sure enough device memory is allocated
	const size_type index = position-begin();
	std::vector< T, HostAllocator<T> > v( n-index+span, T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-insertion region to host
	if( v.size() > span ) CUDA_CALL( cudaMemcpy<T>( &v[span], deviceMemory.get()+index, (n-index), cudaMemcpyDeviceToHost ) ); // copy post-insertion region to +span in host vector
	for( size_type i = 0; i < span; ++i ) v[i] = val; // fill in start region with new value
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+index, &v.front(), v.size(), cudaMemcpyHostToDevice ) ); // copy +span inserted data back onto device
	n += span;
	return position; // since iterators are index based, the return iterator is the same as the one provided
}

template<typename T,class Alloc>
template<class InputIterator>
HOST void vector<T,Alloc>::insert( vector<T,Alloc>::iterator position, InputIterator first, InputIterator last ) {
	const std::vector<T> x( first, last );
	growMemory(n+x.size());
	const size_type index = position-begin();
	std::vector< T,HostAllocator<T> > v( n-index+x.size(), T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-insertion region to host
	if( v.size() > x.size() ) CUDA_CALL( cudaMemcpy<T>( &v[x.size()], deviceMemory.get()+index, (n-index), cudaMemcpyDeviceToHost ) ); // copy post-insertion region to +span in host vector
	for( size_type i = 0; i < x.size(); ++i ) v[i] = x[i]; // fill in start region with new values
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+index, &v.front(), v.size(), cudaMemcpyHostToDevice ) ); // copy +x.size() inserted data back onto device
	n += x.size();
}

template<typename T,class Alloc>
HOST vector<T,Alloc>::iterator vector<T,Alloc>::erase( vector<T,Alloc>::iterator position ) {
	const size_type index = position-begin();
	std::vector< T,HostAllocator<T> > v( n-index-1, T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-erasure region to host
	CUDA_CALL( cudaMemcpy<T>( &v.front(), deviceMemory.get()+(index+1), v.size(), cudaMemcpyDeviceToHost ) );
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+index, &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	--n;
	return position; // since iterators are index based, the return iterator is the same as the one provided
}

template<typename T,class Alloc>
HOST vector<T,Alloc>::iterator vector<T,Alloc>::erase( vector<T,Alloc>::iterator first, vector<T,Alloc>::iterator last ) {
	const size_type index1 = first-begin();
	const size_type index2 = last-begin();
	std::vector< T,HostAllocator<T> > v( n-index2, T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-erasure region to host
	CUDA_CALL( cudaMemcpy<T>( &v.front(), deviceMemory.get()+index2, v.size(), cudaMemcpyDeviceToHost ) );
	CUDA_CALL( cudaMemcpy<T>( &v.front(), deviceMemory.get()+index2, v.size(), cudaMemcpyDeviceToHost ) );
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+index1, &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	n -= index2-index1;
	return last; // since iterators are index based, the return iterator is the same as the one provided
}

template<typename T,class Alloc>
HOST DEVICE void vector<T,Alloc>::swap( vector<T,Alloc>& other ) {
	// just swap all members
	{ device_ptr<T> tmp = other.deviceMemory; other.deviceMemory = deviceMemory; deviceMemory = tmp; }
	{ size_type tmp = other.n; other.n = n; n = tmp; }
	{ size_type tmp = other.m; other.m = m; m = tmp; }
}

template<typename T,class Alloc>
HOST void vector<T,Alloc>::assign( size_type newSize, value_type& value ) {
	growMemory(newSize); // make sure enough device memory is allocated
	std::vector<T> v( newSize, value );
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	//CUDA_CALL( cudaMemcpy( deviceMemory.get(), &v[0], v.size()*sizeof(T), cudaMemcpyHostToDevice ) );
	n = newSize;
}

template<typename T,class Alloc>
template<class InputIterator>
HOST void vector<T,Alloc>::assign( InputIterator first, InputIterator last ) {
	std::vector<T> v( first, last );
	growMemory( v.size() ); // make sure enough device memory is allocated
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	//CUDA_CALL( cudaMemcpy( deviceMemory.get(), &v[0], v.size()*sizeof(T), cudaMemcpyHostToDevice ) );
	n = v.size();
}

template<typename T,class Alloc>
HOST void vector<T,Alloc>::push_back( const value_type& v ) {
	growMemory(n+1);
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+(n*sizeof(T)), &v, 1, cudaMemcpyHostToDevice ) );
	//CUDA_CALL( cudaMemcpy( deviceMemory.get()+(n*sizeof(T)), &v, sizeof(v), cudaMemcpyHostToDevice ) );
	++n;
}


} // namespace ecuda

#endif
