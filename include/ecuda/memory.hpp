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
// memory.hpp
//
// Smart pointers to device memory and simple homespun unique_ptr in absense
// of C++11 memory library.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MEMORY_HPP
#define ECUDA_MEMORY_HPP

#include <cstddef>
#include "global.hpp"

#ifdef __CPP11_SUPPORTED__
#include <memory>
#endif

namespace ecuda {

///
/// A smart pointer for device memory.
///
/// This class keeps a pointer to allocated device memory and automatically
/// deallocates it when it goes out of scope.  The workings are similar to
/// a C++11 shared_ptr.  Since deallocation can only be done from host code
/// reference counting only occurs within host code.  On the device the pointer
/// is passed around freely without regards to reference counting and will
/// never undergo deallocation.
///
template<typename T>
class device_ptr {

public:
	typedef T element_type; //!< data type represented in allocated memory
	typedef T* pointer; //!< data type pointer
	typedef T& reference; //!< data type reference
	typedef void** allocation_pointer; //!< pointer to pointer used by CUDA API to allocate device memory
	typedef std::size_t size_type; //!< size type for pointer arithmetic and reference counting

private:
	pointer ptr; //!< pointer to device memory
	size_type* shared_count; //!< pointer to reference count

public:
	HOST DEVICE device_ptr() : ptr(NULL) {
		#ifndef __CUDA_ARCH__
		shared_count = new size_type;
		*shared_count = 1;
		#endif
	}
	HOST DEVICE device_ptr( const device_ptr<T>& src ) : ptr(src.ptr), shared_count(src.shared_count) {
		#ifndef __CUDA_ARCH__
		++(*shared_count);
		#endif
	}
	// destroys the smart pointer, if instantiated from the host this will decrement the share count,
	// if instantiated from the device nothing happens, iff. the share count is zero and the smart
	// pointer resides on the host, the underlying device memory will be deallocated.
	HOST DEVICE ~device_ptr() {
		#ifndef __CUDA_ARCH__
		--(*shared_count);
		if( !(*shared_count) and ptr ) {
			CUDA_CALL( cudaFree(ptr) );
			delete shared_count;
		}
		#endif
	}

	// both host and device can get the pointer itself
	HOST DEVICE inline pointer get() const { return ptr; }
	HOST DEVICE inline operator bool() const { return get() != NULL; }

	// only device can dereference the pointer or call for the pointer in the context of acting upon the object
	DEVICE inline reference operator*() const { return *ptr; }
	DEVICE inline pointer   operator->() const { return ptr; }
	DEVICE inline reference operator[]( size_type index ) const { return *(ptr+index); }

	// get a pointer to the pointer to the device memory suitable for use with cudaMalloc...()-style calls
	HOST inline allocation_pointer alloc_ptr() { return reinterpret_cast<void**>(&ptr); }

	// both host and device can do comparisons on the pointer
	HOST DEVICE inline bool operator==( const device_ptr<T>& other ) const { return ptr == other.ptr; }
	HOST DEVICE inline bool operator!=( const device_ptr<T>& other ) const { return ptr != other.ptr; }
	HOST DEVICE inline bool operator< ( const device_ptr<T>& other ) const { return ptr <  other.ptr; }
	HOST DEVICE inline bool operator> ( const device_ptr<T>& other ) const { return ptr >  other.ptr; }
	HOST DEVICE inline bool operator<=( const device_ptr<T>& other ) const { return ptr <= other.ptr; }
	HOST DEVICE inline bool operator>=( const device_ptr<T>& other ) const { return ptr >= other.ptr; }

	HOST DEVICE device_ptr<T>& operator=( const device_ptr<T>& other ) {
		#ifndef __CUDA_ARCH__
		~device_ptr();
		#endif
		ptr = other.ptr;
		#ifndef __CUDA_ARCH__
		shared_count = other.shared_count;
		++(*shared_count);
		#endif
		return *this;
	}

};

#ifdef __CPP11_SUPPORTED__
// some future proofing for the glorious day when
// nvcc will support C++11 and we can just use the
// prepackaged implementations
template<typename T> typedef std::unique_ptr<T> unique_ptr<T>;
template<typename T> typedef std::unique_ptr<T[]> unique_ptr<T[]>;
#else
template<typename T>
class unique_ptr {

public:
	typedef T element_type;
	typedef T* pointer;
	typedef T& reference;

private:
	T* ptr;

public:
	HOST DEVICE unique_ptr( T* ptr=NULL ) : ptr(ptr) {}
	HOST DEVICE ~unique_ptr() {	if( ptr ) delete ptr; }

	HOST DEVICE inline pointer get() const { return ptr; }
	HOST DEVICE inline operator bool() const { return get() != NULL; }
	HOST DEVICE inline reference operator*() const { return *ptr; }
	HOST DEVICE inline pointer operator->() const { return ptr; }

	HOST DEVICE inline bool operator==( const unique_ptr<T>& other ) const { return ptr == other.ptr; }
	HOST DEVICE inline bool operator!=( const unique_ptr<T>& other ) const { return ptr != other.ptr; }
	HOST DEVICE inline bool operator<( const unique_ptr<T>& other ) const { return ptr < other.ptr; }
	HOST DEVICE inline bool operator>( const unique_ptr<T>& other ) const { return ptr > other.ptr; }
	HOST DEVICE inline bool operator<=( const unique_ptr<T>& other ) const { return ptr <= other.ptr; }
	HOST DEVICE inline bool operator>=( const unique_ptr<T>& other ) const { return ptr >= other.ptr; }

};

template<typename T>
class unique_ptr<T[]> {

public:
	typedef T element_type;
	typedef T* pointer;
	typedef T& reference;
	typedef std::size_t size_type;

private:
	T* ptr;

public:
	HOST DEVICE unique_ptr<T[]>( T* ptr=NULL ) : ptr(ptr) {}
	//unique_ptr<T[]>( const unique_ptr<T[]>& src ) : ptr(src.ptr) {}
	HOST DEVICE ~unique_ptr<T[]>() { if( ptr ) delete [] ptr; }

	HOST DEVICE inline pointer get() const { return ptr; }
	HOST DEVICE inline operator bool() const { return get() != NULL; }
	HOST DEVICE inline reference operator[]( const size_type index ) const { return *(ptr+index); }

	HOST DEVICE inline bool operator==( const unique_ptr<T[]>& other ) const { return ptr == other.ptr; }
	HOST DEVICE inline bool operator!=( const unique_ptr<T[]>& other ) const { return ptr != other.ptr; }
	HOST DEVICE inline bool operator<( const unique_ptr<T[]>& other ) const { return ptr < other.ptr; }
	HOST DEVICE inline bool operator>( const unique_ptr<T[]>& other ) const { return ptr > other.ptr; }
	HOST DEVICE inline bool operator<=( const unique_ptr<T[]>& other ) const { return ptr <= other.ptr; }
	HOST DEVICE inline bool operator>=( const unique_ptr<T[]>& other ) const { return ptr >= other.ptr; }

};
#endif

} // namespace ecuda

#endif
