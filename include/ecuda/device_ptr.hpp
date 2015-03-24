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
// device_ptr.hpp
//
// Smart pointer to device memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_DEVICE_PTR_HPP
#define ECUDA_DEVICE_PTR_HPP

//#include <cstddef>
#include "global.hpp"

//#ifdef __CPP11_SUPPORTED__
//#include <memory>
//#endif

namespace ecuda {

/*
template<typename T>
class deleter {
public:
	typedef T element_type;
	typedef T* pointer;
public:
	HOST DEVICE deleter() {}
	HOST DEVICE deleter( const deleter& other ) {}
	HOST DEVICE ~deleter() {}
	HOST inline void operator()( pointer ptr ) { if( ptr ) CUDA_CALL( cudaFree(ptr) ); }
};
*/

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
/// Like a typical smart pointer, this class handles deallocation but allocation
/// is performed elsewhere and the pointer to the allocated memory location is
/// passed to the constructor.
///
template<typename T>
class device_ptr {

public:
	typedef T element_type; //!< data type represented in allocated memory
	typedef T* pointer; //!< data type pointer
	typedef T& reference; //!< data type reference
	typedef std::size_t size_type; //!< size type for pointer arithmetic and reference counting
	typedef std::ptrdiff_t difference_type; //!< signed integer type of the result of subtracting two pointers

private:
	pointer ptr; //!< pointer to device memory
	size_type* shared_count; //!< pointer to reference count

public:
	//HOST DEVICE device_ptr() : ptr(NULL) {
	//	#ifndef __CUDA_ARCH__
	//	shared_count = new size_type;
	//	*shared_count = 1;
	//	#endif
	//}
	HOST DEVICE device_ptr( pointer ptr = pointer() ) : ptr(ptr) {
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

	#ifdef __CPP11_SUPPORTED__
	HOST DEVICE device_ptr( device_ptr<T>&& src ) : ptr(src.ptr), shared_count(src.shared_count) {
		src.ptr = NULL;
		src.shared_count = NULL;
	}
	#endif

	// destroys the smart pointer, if instantiated from the host this will decrement the share count,
	// if instantiated from the device nothing happens, iff. the share count is zero and the smart
	// pointer resides on the host, the underlying device memory will be deallocated.
	HOST DEVICE ~device_ptr() {
		#ifndef __CUDA_ARCH__
		--(*shared_count);
		if( !(*shared_count) ) {
			if( ptr ) CUDA_CALL( cudaFree( ptr ) );
			//deleter<T>()(ptr);
			delete shared_count;
		}
		#endif
	}

	// both host and device can get the pointer itself
	HOST DEVICE inline pointer get() const { return ptr; }
	HOST DEVICE inline operator bool() const { return get() != NULL; }
	HOST DEVICE inline operator pointer() const { return ptr; }

	// only device can dereference the pointer or call for the pointer in the context of acting upon the object
	DEVICE inline reference operator*() const { return *ptr; }
	DEVICE inline pointer   operator->() const { return ptr; }
	DEVICE inline reference operator[]( size_type index ) const { return *(ptr+index); }

	// both host and device can do comparisons on the pointer
	HOST DEVICE inline bool operator==( const device_ptr<T>& other ) const { return ptr == other.ptr; }
	HOST DEVICE inline bool operator!=( const device_ptr<T>& other ) const { return ptr != other.ptr; }
	HOST DEVICE inline bool operator< ( const device_ptr<T>& other ) const { return ptr <  other.ptr; }
	HOST DEVICE inline bool operator> ( const device_ptr<T>& other ) const { return ptr >  other.ptr; }
	HOST DEVICE inline bool operator<=( const device_ptr<T>& other ) const { return ptr <= other.ptr; }
	HOST DEVICE inline bool operator>=( const device_ptr<T>& other ) const { return ptr >= other.ptr; }

	HOST DEVICE inline difference_type operator-( const device_ptr<T>& other ) const { return ptr - other.ptr; }

	HOST device_ptr<T>& operator=( pointer ptr ) {
		~device_ptr();
		this->ptr = ptr;
		shared_count = new size_type;
		*shared_count = 1;
		return *this;
	}

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

} // namespace ecuda

#endif
