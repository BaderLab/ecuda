//----------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// allocators.hpp
// STL-compatible memory allocators.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ARRAY_HPP
#define ECUDA_ARRAY_HPP

#include <stdexcept>

namespace ecuda {

template<typename T>
class HostAllocator {
public:
	typedef T value_type;
	typedef T* pointer;
	typedef T& reference;
	typedef const T* const_pointer;
	typedef const T& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	template<typename U> struct rebind { typedef HostAllocator<U> other; };
public:
	HostAllocator() throw() {}
	HostAllocator( const HostAllocator& alloc ) throw() {}
	template<typename U>
	HostAllocator( const HostAllocator<U>& alloc ) throw() {}
	~HostAllocator() throw() {}
	pointer address( reference x ) { return &x; }
	const_pointer address( const_reference x ) const { return &x; }
	pointer allocate( size_type n, std::allocator<void>::const_pointer hint = 0 ) {
		pointer ptr = NULL;
		const cudaError_t result = cudaMallocHost( reinterpret_cast<void**>(&ptr), n*sizeof(T) );
		if( result != cudaSuccess ) throw std::bad_alloc();
		return ptr;
	}
	void deallocate( pointer ptr, size_type /*n*/ ) {
		if( ptr ) cudaFreeHost( reinterpret_cast<void*>(ptr) );
	}
	size_type max_size() const throw() { return std::numeric_limits<size_type>::max(); }
	void construct( pointer ptr, const_reference val ) { new ((void*)ptr) value_type (val);	}
	void destroy( pointer ptr ) { ptr->~value_type(); }
};

//template<typename T>
//class host_vector : public std::vector<T> {
//public:
//	host_array() : std::vector<T>( HostAllocator() ) {}
//	host_array( std::vector<T>::size_type n, const std::vector<T>::value_type& val = std::vector<T>::value_type() ) : std::vector<T>( n, val, HostAllocator() ) {}
//
//}

} // namespace ecuda

#endif
