//----------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// memory.hpp
// Smart pointers to device memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MEMORY_HPP
#define ECUDA_MEMORY_HPP

#include <cstddef>
#include "global.hpp"

#if __cplusplus >= 201103L
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

	// only device can dereference the pointer
	DEVICE inline reference operator*() const { return *ptr; }
	DEVICE inline pointer   operator->() const { return ptr; }
	DEVICE inline reference operator[]( size_type index ) const { return *(ptr+index); }

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

#if __cplusplus >= 201103L
// some future proofing for the glorious day when
// nvcc will support C++11 and we can just use the
// prepackaged implementations
template<typename T> typedef std::unique_ptr<T> unique_ptr<T>
template<typename T> typedef std::unique_ptr<T[]> unique_ptr<T[]>
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
	unique_ptr( T* ptr=NULL ) : ptr(ptr) {}
	~unique_ptr() {	if( ptr ) delete ptr; }

	inline pointer get() const { return ptr; }
	inline operator bool() const { return get() != NULL; }
	inline reference operator*() const { return *ptr; }
	inline pointer operator->() const { return ptr; }

	inline bool operator==( const unique_ptr<T>& other ) const { return ptr == other.ptr; }
	inline bool operator!=( const unique_ptr<T>& other ) const { return ptr != other.ptr; }
	inline bool operator<( const unique_ptr<T>& other ) const { return ptr < other.ptr; }
	inline bool operator>( const unique_ptr<T>& other ) const { return ptr > other.ptr; }
	inline bool operator<=( const unique_ptr<T>& other ) const { return ptr <= other.ptr; }
	inline bool operator>=( const unique_ptr<T>& other ) const { return ptr >= other.ptr; }

};

template<typename T>
class unique_ptr<T[]> {

public:
	typedef T element_type;
	typedef T* pointer;
	typedef T& reference;
	typedef std::size_t size_type;

private:
	T ptr;

public:
	unique_ptr<T[]>( T ptr=NULL ) : ptr(ptr) {}
	~unique_ptr<T[]>() { if( &ptr ) delete [] ptr; }

	inline pointer get() const { return ptr; }
	inline operator bool() const { return get() != NULL; }
	inline reference operator[]( const size_type index ) const { return *(ptr+index); }

	inline bool operator==( const unique_ptr<T[]>& other ) const { return ptr == other.ptr; }
	inline bool operator!=( const unique_ptr<T[]>& other ) const { return ptr != other.ptr; }
	inline bool operator<( const unique_ptr<T[]>& other ) const { return ptr < other.ptr; }
	inline bool operator>( const unique_ptr<T[]>& other ) const { return ptr > other.ptr; }
	inline bool operator<=( const unique_ptr<T[]>& other ) const { return ptr <= other.ptr; }
	inline bool operator>=( const unique_ptr<T[]>& other ) const { return ptr >= other.ptr; }

};
#endif

} // namespace ecuda

#endif
