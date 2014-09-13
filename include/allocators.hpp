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
#ifndef ECUDA_ALLOCATORS_HPP
#define ECUDA_ALLOCATORS_HPP

#include <stdexcept>

namespace ecuda {

///
/// An STL allocator that utilizes page-locked host memory.
///
/// e.g. std::vector<int>( HostAllocator<int>() ) would instantiate a vector
///      whose underlying contents would be stored in page-locked host memory.
///
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

} // namespace ecuda

#endif
