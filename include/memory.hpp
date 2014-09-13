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
	typedef T element_type;
	typedef T* pointer;
	typedef void** allocation_pointer;
	typedef T& reference;
	typedef std::size_t size_type;

private:
	pointer ptr;
	size_type* shared_count;

public:
	__host__ __device__ device_ptr() : ptr(NULL) {
		#ifndef __CUDA_ARCH__
		shared_count = new size_type;
		*shared_count = 1;
		#endif
	}
	__host__ __device__ device_ptr( const device_ptr<T>& src ) : ptr(src.ptr), shared_count(src.shared_count) {
		#ifndef __CUDA_ARCH__
		++(*shared_count);
		#endif
	}
	__host__ __device__ ~device_ptr() {
		#ifndef __CUDA_ARCH__
		--(*shared_count);
		if( !(*shared_count) and ptr ) {
			CUDA_CALL( cudaFree(ptr) );
			delete shared_count;
		}
		#endif
	}

	// both host and device can get the pointer itself
	__host__ __device__ pointer get() const { return ptr; }
	__host__ __device__ operator bool() const { return get() != NULL; }

	// only device can dereference the pointer
	__device__ reference operator*() const { return *ptr; }
	__device__ pointer   operator->() const { return ptr; }
	__device__ reference operator[]( size_type index ) const { return *(ptr+index); }

	__host__ allocation_pointer alloc_ptr() { return reinterpret_cast<void**>(&ptr); }

	// both host and device can do comparisons on the pointer
	__host__ __device__ bool operator==( const device_ptr<T>& other ) const { return ptr == other.ptr; }
	__host__ __device__ bool operator!=( const device_ptr<T>& other ) const { return ptr != other.ptr; }
	__host__ __device__ bool operator< ( const device_ptr<T>& other ) const { return ptr <  other.ptr; }
	__host__ __device__ bool operator> ( const device_ptr<T>& other ) const { return ptr >  other.ptr; }
	__host__ __device__ bool operator<=( const device_ptr<T>& other ) const { return ptr <= other.ptr; }
	__host__ __device__ bool operator>=( const device_ptr<T>& other ) const { return ptr >= other.ptr; }

	__host__ __device__ device_ptr<T>& operator=( const device_ptr<T>& other ) {
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

/*
template<typename T>
class unique_ptr {

public:
	typedef T element_type;
	typedef T* pointer;
	typedef T& reference;

private:
	T* ptr;

public:
	unique_ptr( T* ptr=NULL ) : ptr(ptr) {
		std::cerr << "assigning smart pointer" << std::endl;
		std::cerr << "ptr=" << &ptr << std::endl;
		std::cerr << "complete" << std::endl;
	}
	~unique_ptr() {	if( ptr ) CUDA_CALL( cudaFree(ptr) ); }

	__host__ __device__ pointer get() const { return ptr; }
	__host__ __device__ operator bool() const { return get() != NULL; }
	__host__ __device__ reference operator*() const { return *ptr; }
	__host__ __device__ pointer operator->() const { return ptr; }

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
	unique_ptr<T[]>( T ptr=NULL ) : ptr(ptr) { 
		std::cerr << "Assigning smart pointer" << std::endl;
		std::cerr << "Address=" << ptr << std::endl; 
		std::cerr << "Complete" << std::endl;
	}
	~unique_ptr<T[]>() { if( &ptr ) CUDA_CALL( cudaFree(&ptr) ); }

	__host__ __device__ pointer get() const { return ptr; }
	__host__ __device__ operator bool() const { return get() != NULL; }
	__host__ __device__ reference operator[]( const size_type index ) const { return *(ptr+index); }

	inline bool operator==( const unique_ptr<T[]>& other ) const { return ptr == other.ptr; }
	inline bool operator!=( const unique_ptr<T[]>& other ) const { return ptr != other.ptr; }
	inline bool operator<( const unique_ptr<T[]>& other ) const { return ptr < other.ptr; }
	inline bool operator>( const unique_ptr<T[]>& other ) const { return ptr > other.ptr; }
	inline bool operator<=( const unique_ptr<T[]>& other ) const { return ptr <= other.ptr; }
	inline bool operator>=( const unique_ptr<T[]>& other ) const { return ptr >= other.ptr; }

};
*/

} // namespace ecuda

#endif
