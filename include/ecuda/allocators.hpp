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
// allocators.hpp
// STL-compatible memory allocators.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ALLOCATORS_HPP
#define ECUDA_ALLOCATORS_HPP

#include <limits>
#include <stdexcept>

namespace ecuda {

///
/// An STL allocator that utilizes page-locked host memory.
///
/// Page-locked or "pinned" memory makes copying memory from the GPU (device)
/// to the CPU (host) faster.  Using STL containers with this allocator makes
/// them better at acting as "staging" points when moving data from the
/// device memory to the host memory.
///
/// e.g. std::vector< int, HostAllocator<int> >( HostAllocator<int>() ) would
///      instantiate a vector whose underlying contents would be stored in
///      page-locked host memory.  Then a call to, for example:
///        ecuda::vector<int> deviceVector(1000);
///        // do work on device vector using the GPU...
///        std::vector< int, ecuda::HostAllocator<int> > hostVector( 1000, HostAllocator<int>() );
///        deviceVector >> hostVector; // copy results from device to host
///        // do work on the host vector...
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
