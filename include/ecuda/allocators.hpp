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
// allocators.hpp
//
// STL-compatible memory allocators using CUDA memory allocation routines.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ALLOCATORS_HPP
#define ECUDA_ALLOCATORS_HPP

#include <limits>
#include <stdexcept>

#include "global.hpp"
#include "padded_ptr.hpp"

namespace ecuda {

///
/// \brief An STL allocator for page-locked host memory.
///
/// The implementation uses the CUDA API functions cudaMallocHost and
/// cudaFreeHost.
///
/// Page-locked or "pinned" memory makes copying memory from the GPU (device)
/// to the CPU (host) faster.  Using STL containers with this allocator makes
/// them better at acting as "staging" points when moving data from the
/// device memory to the host memory. This is used internally to optimize
/// host <=> device transfers that involve any kind of temporary staging memory,
/// but can be used effectively by an end-user of the library as well.
///
/// For example:
/// \code{.cpp}
/// std::vector< int, host_allocator<int> > v;
/// \endcode
/// This would instantiate a vector whose underlying contents would be stored in
/// page-locked host memory.  Then a call to, for example:
/// \code{.cpp}
/// ecuda::vector<int> deviceVector(1000);
/// // do work on device vector using the GPU...
/// std::vector< int, ecuda::host_allocator<int> > hostVector( 1000 );
/// deviceVector >> hostVector; // copy results from device to host
/// \endcode
/// This would potentially be a faster transfer than one would get using a
/// <tt>std::vector</tt> with the default STL allocator.
///
template<typename T>
class host_allocator {

public:
	typedef T value_type; //!< element type
	typedef T* pointer; //!< pointer to element
	typedef T& reference; //!< reference to element
	typedef const T* const_pointer; //!< pointer to constant element
	typedef const T& const_reference; //!< reference to constant element
	typedef std::size_t size_type; //!< quantities of elements
	typedef std::ptrdiff_t difference_type; //!< difference between two pointers
	/// \cond DEVELOPER_DOCUMENTATION
	template<typename U> struct rebind { typedef host_allocator<U> other; }; //!< its member type U is the equivalent allocator type to allocate elements of type U
	/// \endcond

public:
	///
	/// \brief Constructs a host allocator object.
	///
	host_allocator() throw() {}

	///
	/// \brief Constructs a host allocator object from another host allocator object.
	/// \param alloc Allocator object.
	///
	host_allocator( const host_allocator& alloc ) throw() {}

	///
	/// \brief Constructs a host allocator object from another host allocator object with a different element type.
	/// \param alloc Allocator object.
	///
	template<typename U>
	host_allocator( const host_allocator<U>& alloc ) throw() {}

	///
	/// \brief Destructs the host allocator object.
	///
	~host_allocator() throw() {}

	///
	/// \brief Returns the address of x.
	///
	/// This effectively means returning &x.
	///
	/// \param x Reference to object.
	/// \return A pointer to the object.
	///
	inline pointer address( reference x ) { return &x; }

	///
	/// \brief Returns the address of x.
	///
	/// This effectively means returning &x.
	///
	/// \param x Reference to object.
	/// \return A pointer to the object.
	///
	inline const_pointer address( const_reference x ) const { return &x; }

	///
	/// \brief Allocate block of storage.
	///
	/// Attempts to allocate a block of storage with a size large enough to contain n elements of member type
	/// value_type (as alias of the allocator's template parameter), and returns a pointer to the first element.
	///
	/// The storage is aligned appropriately for object of type value_type, but they are not constructed.
	///
	/// The block of storage is allocated using cudaMallocHost and throws std::bad_alloc if it cannot
	/// allocate the total amount of storage requested.
	///
	/// \param n Number of elements (each of size sizeof(value_type)) to be allocated.
	/// \param hint Either 0 or a value previously obtained by another call to allocate and not
	///             yet freed with deallocate.  For standard memory allocation, a non-zero value may
	///             used as a hint to improve performance by allocating the new block near the one
	///             specified. The address of an adjacent element is often a good choice.
	///             In this case, hint is always ignored since the CUDA host memory allocator
	///             cannot take advantage of it.
	/// \return A pointer to the initial element in the block of storage.
	///
	pointer allocate( size_type n, std::allocator<void>::const_pointer hint = 0 ) {
		pointer ptr = NULL;
		const cudaError_t result = cudaMallocHost( reinterpret_cast<void**>(&ptr), n*sizeof(T) );
		if( result != cudaSuccess ) throw std::bad_alloc();
		return ptr;
	}

	///
	/// \brief Releases a block of storage previously allocated with member allocate and not yet released.
	///
	/// The elements in the array are not destroyed by a call to this member function.
	///
	/// In the default allocator, the block of storage is at some point deallocated using \c \::operator \c delete (either
	/// during the function call, or later).
	///
	/// \param ptr Pointer to a block of storage previously allocated with allocate. pointer is a member type
	///            (defined as an alias of T* in ecuda::host_allocator<T>).
	///
	inline void deallocate( pointer ptr, size_type /*n*/ ) {
		if( ptr ) cudaFreeHost( reinterpret_cast<void*>(ptr) );
	}

	///
	/// \brief Returns the maximum number of elements, each of member type value_type (an alias of allocator's template parameter)
	///        that could potentially be allocated by a call to member allocate.
	///
	/// A call to member allocate with the value returned by this function can still fail to allocate the requested storage.
	///
	/// \return The nubmer of elements that might be allcoated as maximum by a call to member allocate.
	///
	inline size_type max_size() const throw() { return std::numeric_limits<size_type>::max(); }

	///
	/// \brief Constructs an element object on the location pointed by ptr.
	/// \param ptr Pointer to a location with enough storage space to contain an element of type value_type.
	///            pointer is a member type (defined as an alias of T* in ecuda::host_allocator<T>).
	/// \param val Value to initialize the constructed element to.
	///            const_reference is a member type (defined as an alias of T& in ecuda::host_allocator<T>).
	///
	inline void construct( pointer ptr, const_reference val ) { new ((void*)ptr) value_type (val);	}

	///
	/// \brief Destroys in-place the object pointed by ptr.
	///        Notice that this does not deallocate the storage for the element (see member deallocate to release storage space).
	/// \param ptr Pointer to the object to be destroyed.
	///
	inline void destroy( pointer ptr ) { ptr->~value_type(); }

};


///
/// \brief An STL allocator for device memory.
///
/// The implementation uses the CUDA API functions cudaMalloc and cudaFree.
///
template<typename T>
class device_allocator {

public:
	typedef T value_type; //!< element type
	typedef T* pointer; //!< pointer to element
	typedef T& reference; //!< reference to element
	typedef const T* const_pointer; //!< pointer to constant element
	typedef const T& const_reference; //!< reference to constant element
	typedef std::size_t size_type; //!< quantities of elements
	typedef std::ptrdiff_t difference_type; //!< difference between two pointers
	/// \cond DEVELOPER_DOCUMENTATION
	template<typename U> struct rebind { typedef device_allocator<U> other; }; //!< its member type U is the equivalent allocator type to allocate elements of type U
	/// \endcond

public:
	///
	/// \brief Constructs a device allocator object.
	///
	HOST DEVICE device_allocator() throw() {}

	///
	/// \brief Constructs a device allocator object from another device allocator object.
	/// \param alloc Allocator object.
	///
	HOST DEVICE device_allocator( const device_allocator& alloc ) throw() {}

	///
	/// \brief Constructs a device allocator object from another device allocator object with a different element type.
	/// \param alloc Allocator object.
	///
	template<typename U>
	HOST DEVICE device_allocator( const device_allocator<U>& alloc ) throw() {}

	///
	/// \brief Destructs the device allocator object.
	///
	HOST DEVICE ~device_allocator() throw() {}

	///
	/// \brief Returns the address of x.
	///
	/// This effectively means returning &x.
	///
	/// \param x Reference to object.
	/// \return A pointer to the object.
	///
	HOST DEVICE inline pointer address( reference x ) { return &x; }

	///
	/// \brief Returns the address of x.
	///
	/// This effectively means returning &x.
	///
	/// \param x Reference to object.
	/// \return A pointer to the object.
	///
	HOST DEVICE inline const_pointer address( const_reference x ) const { return &x; }

	///
	/// \brief Allocate block of storage.
	///
	/// Attempts to allocate a block of storage with a size large enough to contain n elements of member type
	/// value_type (as alias of the allocator's template parameter), and returns a pointer to the first element.
	///
	/// The storage is aligned appropriately for object of type value_type, but they are not constructed.
	///
	/// The block of storage is allocated using cudaMallocHost and throws std::bad_alloc if it cannot
	/// allocate the total amount of storage requested.
	///
	/// \param n Number of elements (each of size sizeof(value_type)) to be allocated.
	/// \param hint Either 0 or a value previously obtained by another call to allocate and not
	///             yet freed with deallocate.  For standard memory allocation, a non-zero value may
	///             used as a hint to improve performance by allocating the new block near the one
	///             specified. The address of an adjacent element is often a good choice.
	///             In this case, hint is always ignored since the CUDA device memory allocator
	///             cannot take advantage of it.
	/// \return A pointer to the initial element in the block of storage.
	///
	HOST pointer allocate( size_type n, std::allocator<void>::const_pointer hint = 0 ) {
		pointer ptr = NULL;
		const cudaError_t result = cudaMalloc( reinterpret_cast<void**>(&ptr), n*sizeof(T) );
		if( result != cudaSuccess ) throw std::bad_alloc();
		return ptr;
	}

	///
	/// \brief Releases a block of storage previously allocated with member allocate and not yet released.
	///
	/// The elements in the array are not destroyed by a call to this member function.
	///
	/// In the default allocator, the block of storage is at some point deallocated using \c \::operator \c delete (either
	/// during the function call, or later).
	///
	/// \param ptr Pointer to a block of storage previously allocated with allocate. pointer is a member type
	///            (defined as an alias of T* in ecuda::device_allocator<T>).
	///
	HOST inline void deallocate( pointer ptr, size_type /*n*/ ) { if( ptr ) cudaFree( reinterpret_cast<void*>(ptr) ); }

	///
	/// \brief Returns the maximum number of elements, each of member type value_type (an alias of allocator's template parameter)
	///        that could potentially be allocated by a call to member allocate.
	///
	/// A call to member allocate with the value returned by this function can still fail to allocate the requested storage.
	///
	/// \return The nubmer of elements that might be allcoated as maximum by a call to member allocate.
	///
	HOST DEVICE inline size_type max_size() const throw() { return std::numeric_limits<size_type>::max(); }

	///
	/// \brief Constructs an element object on the location pointed by ptr.
	/// \param ptr Pointer to a location with enough storage space to contain an element of type value_type.
	///            pointer is a member type (defined as an alias of T* in ecuda::device_allocator<T>).
	/// \param val Value to initialize the constructed element to.
	///            const_reference is a member type (defined as an alias of T& in ecuda::device_allocator<T>).
	///
	HOST inline void construct( pointer ptr, const_reference val ) {
		CUDA_CALL( cudaMemcpy( reinterpret_cast<void*>(ptr), reinterpret_cast<const void*>(&val), sizeof(val), cudaMemcpyHostToDevice ) );
	}

	///
	/// \brief Destroys in-place the object pointed by ptr.
	///        Notice that this does not deallocate the storage for the element (see member deallocate to release storage space).
	/// \param ptr Pointer to the object to be destroyed.
	///
	HOST inline void destroy( pointer ptr ) { ptr->~value_type(); }

};

///
/// \brief An pseudo-STL allocator for hardware aligned device memory.
///
/// The implementation uses the CUDA API functions cudaMallocPitch and cudaFree.
///
/// This allocator is not precisely to STL specification because the allocation
/// is two dimensional (requires both a width and height parameter, not just
/// length).  Thus, the allocate() method takes an additional parameter.
///
/// Also, although the allocated memory is contiguous, the arrangement of
/// data within this region is not.  After each width number of elements, there are
/// some bytes of "empty" memory.  This is to ensure that each "row" is
/// hardware aligned so that a read/write operation from/to a range of elements is
/// more likely to be accomplished in fewer operations.  As a result, the pointer
/// to the allocation is an ecuda::padded_ptr which stores information about this
/// padding (i.e. the pitch of the 2D memory).  This allows pointer-like operations
/// that traverse memory to be performed transparently since the padding will be
/// taken into account.
///
template<typename T>
class device_pitch_allocator {

public:
	typedef T value_type; //!< element type
	typedef padded_ptr<T,T*,1> pointer; //!< pointer to element
	typedef T& reference; //!< reference to element
	typedef padded_ptr<const T,const T*,1> const_pointer; //!< pointer to constant element
	typedef const T& const_reference; //!< reference to constant element
	typedef std::size_t size_type; //!< quantities of elements
	typedef std::ptrdiff_t difference_type; //!< difference between two pointers
	/// \cond DEVELOPER_DOCUMENTATION
	template<typename U> struct rebind { typedef device_allocator<U> other; }; //!< its member type U is the equivalent allocator type to allocate elements of type U
	/// \endcond

public:
	///
	/// \brief Constructs a device pitched memory allocator object.
	///
	HOST DEVICE device_pitch_allocator() throw() {}

	///
	/// \brief Constructs a device pitched memory allocator object from another host allocator object.
	/// \param alloc Allocator object.
	///
	HOST DEVICE device_pitch_allocator( const device_pitch_allocator& alloc ) throw() {}

	///
	/// \brief Constructs a device pitched memory allocator object from another device pitched memory allocator object with a different element type.
	/// \param alloc Allocator object.
	///
	template<typename U>
	HOST DEVICE device_pitch_allocator( const device_pitch_allocator<U>& alloc ) throw() {}

	///
	/// \brief Destructs the device pitched memory allocator object.
	///
	HOST DEVICE ~device_pitch_allocator() throw() {}

	///
	/// \brief Returns the address of x.
	///
	/// This effectively means returning &x.
	///
	/// \param x Reference to object.
	/// \return A pointer to the object.
	///
	HOST DEVICE inline pointer address( reference x ) { return &x; }

	///
	/// \brief Returns the address of x.
	///
	/// This effectively means returning &x.
	///
	/// \param x Reference to object.
	/// \return A pointer to the object.
	///
	HOST DEVICE inline const_pointer address( const_reference x ) const { return &x; }

	///
	/// \brief Allocate block of storage.
	///
	/// Attempts to allocate a block of storage with a size large enough to contain n elements of member type
	/// value_type (as alias of the allocator's template parameter), and returns a pointer to the first element.
	///
	/// The storage is aligned appropriately for object of type value_type, but they are not constructed.
	///
	/// The block of storage is allocated using cudaMallocHost and throws std::bad_alloc if it cannot
	/// allocate the total amount of storage requested.
	///
	/// \param w Width of the matrix (each of size sizeof(value_type)) to be allocated.
	/// \param h Height of the matrix to be allocated.
	/// \param hint Either 0 or a value previously obtained by another call to allocate and not
	///             yet freed with deallocate.  For standard memory allocation, a non-zero value may
	///             used as a hint to improve performance by allocating the new block near the one
	///             specified. The address of an adjacent element is often a good choice.
	///             In this case, hint is always ignored since the CUDA device pitched memory memory allocator
	///             cannot take advantage of it.
	/// \return A pointer to the initial element in the block of storage.
	///
	HOST pointer allocate( size_type w, size_type h = 1, std::allocator<void>::const_pointer hint = 0 ) const {
		if( !h ) h = 1; // height must be at least 1
		value_type* nakedPtr;
		std::size_t pitch;
		const cudaError_t result = cudaMallocPitch( reinterpret_cast<void**>(&nakedPtr), &pitch, w*sizeof(value_type), h );
		if( result != cudaSuccess ) throw std::bad_alloc();
		return pointer( nakedPtr, w, pitch-w*sizeof(value_type), 0 );
	}

	///
	/// \brief Releases a block of storage previously allocated with member allocate and not yet released.
	///
	/// The elements in the array are not destroyed by a call to this member function.
	///
	/// In the default allocator, the block of storage is at some point deallocated using \c \::operator \c delete (either
	/// during the function call, or later).
	///
	/// \param ptr Pointer to a block of storage previously allocated with allocate. pointer is a member type
	///            (defined as an alias of T* in ecuda::device_pitch_allocator<T>).
	///
	HOST inline void deallocate( pointer ptr, size_type /*n*/ ) { if( ptr ) cudaFree( ptr ); } // reinterpret_cast<void*>(ptr) ); }

	///
	/// \brief Returns the maximum number of elements, each of member type value_type (an alias of allocator's template parameter)
	///        that could potentially be allocated by a call to member allocate.
	///
	/// A call to member allocate with the value returned by this function can still fail to allocate the requested storage.
	///
	/// \return The nubmer of elements that might be allcoated as maximum by a call to member allocate.
	///
	HOST DEVICE inline size_type max_size() const throw() { return std::numeric_limits<size_type>::max(); }

	///
	/// \brief Constructs an element object on the location pointed by ptr.
	/// \param ptr Pointer to a location with enough storage space to contain an element of type value_type.
	///            pointer is a member type (defined as an alias of T* in ecuda::device_pitch_allocator<T>).
	/// \param val Value to initialize the constructed element to.
	///            const_reference is a member type (defined as an alias of T& in ecuda::device_pitch_allocator<T>).
	///
	HOST inline void construct( pointer ptr, const_reference val ) {
		CUDA_CALL( cudaMemcpy( ptr, reinterpret_cast<const void*>(&val), sizeof(val), cudaMemcpyHostToDevice ) );
	}

	///
	/// \brief Destroys in-place the object pointed by ptr.
	///        Notice that this does not deallocate the storage for the element (see member deallocate to release storage space).
	/// \param ptr Pointer to the object to be destroyed.
	///
	HOST inline void destroy( pointer ptr ) { ptr->~value_type(); }

};


} // namespace ecuda

#endif

