/*
Copyright (c) 2014, Scott Zuyderduyn
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
// array.hpp
//
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
#include "allocators.hpp"
#include "apiwrappers.hpp"

namespace ecuda {

///
/// A video memory-bound array structure.
///
/// Creates a fixed size array in GPU memory.  Redeclares most of the
/// STL methods on the equivalent std::array.  Methods are prefaced with
/// appropriate keywords to declare them as host and/or device capable.
/// In general: operations requiring memory allocation/deallocation/copying
/// are host only, operations to access the values of specific elements
/// are device only, and general information can be accessed by both.
///
template< typename T, class Alloc=DeviceAllocator<T> >
class array {

public:
	typedef T value_type; //!< cell data type
	typedef Alloc allocator_type; //!< allocator type
	typedef std::size_t size_type; //!< index data type
	typedef std::ptrdiff_t difference_type; //!<
	typedef value_type& reference; //!< cell reference type
	typedef const value_type& const_reference; //!< cell const reference type
	typedef value_type* pointer; //!< cell pointer type
	typedef const value_type* const_pointer; //!< cell const pointer type

	typedef ecuda::RandomAccessIterator< array<value_type>, pointer > iterator; //!< iterator type
	typedef ecuda::RandomAccessIterator< const array<value_type>, const_pointer > const_iterator; //!< const iterator type

private:
	// REMEMBER: n altered on device memory won't be reflected on the host object.
	//           Don't allow the device to perform any operations that change its
	//           value.
	size_type n; //!< size of array
	device_ptr<T> deviceMemory; //!< smart pointer to video card memory

public:
	///
	/// \brief Constructs an array with n elements. Each element is a copy of value.
	/// \param n Initial container size (i.e. the number of elements in the container at construction).
	/// \param value Value to fill the container with.
	///
	HOST array( const size_type n=0, const_reference value = T() );
	///
	/// \brief Constructs an array with a copy of each of the elements in src, in the same order.
	/// \param src Another array object of the same type (with the same class tempalte argument T and Alloc), whose contents are copied.
	///
	HOST array( const array<T>& src ) : n(src.n), deviceMemory(src.deviceMemory) {}
	///
	/// \brief Constructs an array with a copy of each of the elements in src, in the same order.
	/// \param src An STL vectory object of the same type (with the same class tempalte argument T and Alloc), whose contents are copied.
	///
	HOST array( const std::vector<T>& src );
	///
	/// \brief Destructs the array object.
	///
	HOST DEVICE ~array() {}

	DEVICE inline reference at( size_type index ) { return deviceMemory[index]; }
	DEVICE inline reference operator[]( size_type index ) { return deviceMemory[index]; }
	DEVICE inline const_reference at( size_type index ) const { return deviceMemory[index]; }
	DEVICE inline const_reference operator[]( size_type index ) const { return deviceMemory[index]; }

	DEVICE inline reference front() { return *deviceMemory; }
	DEVICE inline reference back() { return operator[]( size()-1 ); }
	DEVICE inline const_reference front() const { return *deviceMemory; }
	DEVICE inline const_reference back() const { return operator[]( size()-1 ); }

	HOST DEVICE inline size_type size() const { return n; }
	HOST DEVICE inline T* data() { return deviceMemory.get(); }
	HOST DEVICE inline const T* data() const { return deviceMemory.get(); }

	HOST DEVICE inline iterator begin() { return iterator(this); }
	HOST DEVICE inline iterator end() { return iterator(this,size()); }
	HOST DEVICE inline const_iterator begin() const { return const_iterator(this); }
	HOST DEVICE inline const_iterator end() const { return const_iterator(this,size()); }

	HOST inline allocator_type get_allocator() const { return allocator_type(); }

	template<class OtherAlloc>
	HOST const array<T,Alloc>& operator>>( std::vector<T,OtherAlloc>& vector ) const {
		vector.resize( n );
		CUDA_CALL( cudaMemcpy<T>( &vector.front(), deviceMemory.get(), n, cudaMemcpyDeviceToHost ) );
		//CUDA_CALL( cudaMemcpy( &vector[0], deviceMemory.get(), n*sizeof(T), cudaMemcpyDeviceToHost ) );
		return *this;
	}

	HOST array<T,Alloc>& operator<<( std::vector<T>& vector ) {
		if( size() < vector.size() ) throw std::out_of_range( "ecuda::array is not large enough to fit contents of provided std::vector" );
		CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &vector.front(), vector.size(), cudaMemcpyHostToDevice ) );
		//CUDA_CALL( cudaMemcpy( deviceMemory.get(), &vector[0], vector.size()*sizeof(T), cudaMemcpyHostToDevice ) );
		return *this;
	}

	// critical function used to bridge host->device code
	DEVICE array<T,Alloc>& operator=( const array<T,Alloc>& other ) {
		n = other.n;
		deviceMemory = other.deviceMemory;
		return *this;
	}

};


template<typename T,class Alloc>
HOST array<T,Alloc>::array( const size_type n, const_reference value ) : n(n) {
	if( n ) {
		deviceMemory = device_ptr<T>( Alloc().allocate(n) );
		//CUDA_CALL( cudaMalloc( deviceMemory.alloc_ptr(), n*sizeof(T) ) );
		std::vector<T> v( n, value );
		CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), n, cudaMemcpyHostToDevice ) );
		//CUDA_CALL( cudaMemcpy( deviceMemory.get(), &v[0], n*sizeof(T), cudaMemcpyHostToDevice ) );
	}
}

template<typename T,class Alloc>
HOST array<T,Alloc>::array( const std::vector<T>& src ) : n(src.size()) {
	if( n ) {
		deviceMemory = device_ptr<T>( Alloc().allocate(n) );
		//CUDA_CALL( cudaMalloc( deviceMemory.alloc_ptr(), n*sizeof(T) ) );
		CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &src.front(), n, cudaMemcpyHostToDevice ) );
		//CUDA_CALL( cudaMemcpy( deviceMemory.get(), &src[0], n*sizeof(T), cudaMemcpyHostToDevice ) );
	}
}


} // namespace ecuda

#endif
