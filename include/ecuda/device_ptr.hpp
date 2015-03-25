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

#include "global.hpp"
#include "algorithm.hpp"

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
	size_type* reference_count; //!< pointer to reference count

public:
	///
	/// \brief Default constructor.
	///
	/// \param ptr A pointer to the allocated block of device memory.
	///
	HOST DEVICE device_ptr( pointer ptr = pointer() ) : ptr(ptr) {
		#ifndef __CUDA_ARCH__
		reference_count = new size_type;
		*reference_count = 1;
		#else
		reference_count = nullptr;
		#endif
	}

	///
	/// \brief Copy constructor.
	///
	/// If called from the host, the reference count is incremented. If called from the device,
	/// the underlying pointer is copied but no change to the reference count occurs.
	///
	/// \param src Another device pointer to be used as source to initialize with.
	///
	HOST DEVICE device_ptr( const device_ptr<T>& src ) : ptr(src.ptr), reference_count(src.reference_count) {
		#ifndef __CUDA_ARCH__
		++(*reference_count);
		#endif
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Move constructor.
	///
	/// Constructs the device pointer using move semantics.
	///
	/// \param src Another device pointer whose contents are to be moved.
	///
	HOST DEVICE device_ptr( device_ptr<T>&& src ) : ptr(src.ptr), shared_count(src.shared_count) {
		src.ptr = nullptr;
		#ifndef __CUDA_ARCH__
		src.reference_count = new size_type;
		*(src.reference_count) = 1;
		#else
		src.reference_count = nullptr;
		#endif
	}
	#endif

	///
	/// \brief Destructor.
	///
	/// Destructs the device pointer. If called from host code, the reference count is decremented.
	/// If the reference count becomes zero, the device memory is freed. If called from the device
	/// the object is destroyed but nothing happens to the underlying pointer or reference count.
	///
	HOST DEVICE ~device_ptr() {
		#ifndef __CUDA_ARCH__
		--(*reference_count);
		if( !(*reference_count) ) {
			if( ptr ) CUDA_CALL( cudaFree( ptr ) );
			//deleter<T>()(ptr);
			delete reference_count;
		}
		#endif
	}

	///
	/// \brief Exchanges the contents of *this and other.
	///
	/// \param other device pointer to exchange the contents with
	///
	HOST DEVICE inline void swap( device_ptr& other ) __NOEXCEPT__ {
		#ifdef __CUDA_ARCH__
		ecuda::swap( ptr, other.ptr );
		ecuda::swap( reference_count, other.reference_count );
		#else
		std::swap( ptr, other.ptr );
		std::swap( reference_count, other.reference_count );
		#endif
	}

	HOST DEVICE inline void reset() __NOEXCEPT__ { device_ptr().swap(*this); }
	template<typename U> HOST DEVICE inline void reset( U* p ) __NOEXCEPT__ { device_ptr<T>(p).swap(*this); }

	HOST DEVICE inline pointer get() const { return ptr; }
	DEVICE inline reference operator*() const { return *ptr; }
	DEVICE inline pointer   operator->() const { return ptr; }
	HOST inline size_type use_count() const __NOEXCEPT__ { return reference_count ? *reference_count : 0; }
	HOST inline bool unique() const __NOEXCEPT__ { return use_count() == 1; }
	HOST DEVICE inline operator bool() const { return get() != nullptr; }
	HOST DEVICE inline operator pointer() const { return ptr; }
	template<typename U>
	bool owner_before( const device_ptr<U>& other ) const {
		if( ptr == other.ptr ) return false;
		if( !ptr ) return true;
		if( !other.ptr ) return false;
		return ptr < other.ptr;
	}

	// only device can dereference the pointer or call for the pointer in the context of acting upon the object
	//DEVICE inline reference operator[]( size_type index ) const { return *(ptr+index); }

	// both host and device can do comparisons on the pointer
	HOST DEVICE inline bool operator==( const device_ptr<T>& other ) const { return ptr == other.ptr; }
	HOST DEVICE inline bool operator!=( const device_ptr<T>& other ) const { return ptr != other.ptr; }
	HOST DEVICE inline bool operator< ( const device_ptr<T>& other ) const { return ptr <  other.ptr; }
	HOST DEVICE inline bool operator> ( const device_ptr<T>& other ) const { return ptr >  other.ptr; }
	HOST DEVICE inline bool operator<=( const device_ptr<T>& other ) const { return ptr <= other.ptr; }
	HOST DEVICE inline bool operator>=( const device_ptr<T>& other ) const { return ptr >= other.ptr; }

	HOST DEVICE inline difference_type operator-( const device_ptr<T>& other ) const { return ptr - other.ptr; }

	HOST device_ptr<T>& operator=( pointer p ) {
		~device_ptr();
		ptr = p;
		reference_count = new size_type;
		*reference_count = 1;
		return *this;
	}

	HOST DEVICE device_ptr<T>& operator=( const device_ptr<T>& other ) {
		#ifndef __CUDA_ARCH__
		~device_ptr();
		#endif
		ptr = other.ptr;
		#ifndef __CUDA_ARCH__
		reference_count = other.reference_count;
		++(*reference_count);
		#endif
		return *this;
	}

};

} // namespace ecuda

#endif
