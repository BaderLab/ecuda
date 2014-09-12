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

template<typename T>
class device_ptr {
public:
	typedef T element_type;
	typedef T* pointer;
	typedef void** doublepointer;
	typedef T& reference;
	typedef std::size_t size_type;

private:
	pointer ptr;
	size_type* shared_count;

public:
	device_ptr() : ptr(NULL), shared_count(new size_type) { *shared_count = 0; }
	device_ptr( const device_ptr<T>& src ) : ptr(src.ptr), reference_count(src.shared_count) { ++(*shared_count); }
	~device_ptr() {
		--(*shared_count);
		if( !(*shared_count) and ptr ) {
			CUDA_CALL( cudaFree(ptr) );
			delete shared_count;
		}
	}

	__host__ __device__ pointer get() const { return ptr; }
	__host__ __device__ operator bool() const { return get() != NULL; }
	__device__ reference operator*() const { return *ptr; }
	__device__ pointer   operator->() const { return ptr; }
	__device__ reference operator[]( size_type index ) const { return *(ptr+index); }

	__host__ doublepointer addressof() { return reinterpret_cast<void**>(&ptr); }

	__host__ __device__ bool operator==( const device_ptr<T>& other ) const { return ptr == other.ptr; }
	__host__ __device__ bool operator!=( const device_ptr<T>& other ) const { return ptr != other.ptr; }
	__host__ __device__ bool operator< ( const device_ptr<T>& other ) const { return ptr <  other.ptr; }
	__host__ __device__ bool operator> ( const device_ptr<T>& other ) const { return ptr >  other.ptr; }
	__host__ __device__ bool operator<=( const device_ptr<T>& other ) const { return ptr <= other.ptr; }
	__host__ __device__ bool operator>=( const device_ptr<T>& other ) const { return ptr >= other.ptr; }

	__host__ __device__ device_ptr<T>& operator=( const device_ptr<T>& other ) {
		~device_ptr();
		ptr = other.ptr;
		shared_count = other.shared_count;
		++(*shared_count);
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
