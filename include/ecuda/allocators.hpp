/*
Copyright (c) 2014-2016, Scott Zuyderduyn
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
#include "type_traits.hpp"
#include "ptr/common.hpp"
#include "ptr/padded_ptr.hpp"

namespace ecuda {

///
/// \brief Allocator for page-locked host memory.
///
/// Implementation follows the specification of an STL allocator. The main
/// difference is that the CUDA API functions cudaHostAlloc and cudaFreeHost
/// are used internally to allocate/deallocate memory.
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
/// std::vector< int, ecuda::host_allocator<int> > v;
/// \endcode
/// This would instantiate a vector whose underlying contents would be stored in
/// page-locked host memory.  Then a call to, for example:
/// \code{.cpp}
/// ecuda::vector<int> deviceVector(1000);
/// // do work on device vector using the GPU...
/// std::vector< int, ecuda::host_allocator<int> > hostVector( 1000 );
/// ecuda::copy( deviceVector.begin(), deviceVector.end(), hostVector.begin() ); // copy results from device to host
/// \endcode
/// This would potentially be a faster transfer than one would get using a
/// <tt>std::vector</tt> with the default STL allocator.
///
template<typename T,unsigned Flags=cudaHostAllocDefault>
class host_allocator
{

public:
	typedef T                                                   value_type;      //!< element type
	typedef typename ecuda::add_pointer<T>::type                pointer;         //!< pointer to element
	typedef typename ecuda::add_lvalue_reference<T>::type       reference;       //!< reference to element
	typedef typename make_const<pointer>::type                  const_pointer;   //!< pointer to constant element
	typedef typename ecuda::add_lvalue_reference<const T>::type const_reference; //!< reference to constant element
	typedef std::size_t                                         size_type;       //!< quantities of elements
	typedef std::ptrdiff_t                                      difference_type; //!< difference between two pointers
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
	/// value_type, and returns a pointer to the first element.
	///
	/// The storage is aligned appropriately for object of type value_type, but they are not constructed.
	///
	/// The block of storage is allocated using cudaHostAlloc and throws std::bad_alloc if it cannot
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
	pointer allocate( size_type n, std::allocator<void>::const_pointer hint = 0 )
	{
		pointer ptr = NULL;
		const cudaError_t result = cudaHostAlloc( reinterpret_cast<void**>(&ptr), n*sizeof(T), Flags );
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
	inline void deallocate( pointer ptr, size_type )
	{
		typedef typename ecuda::add_pointer<value_type>::type raw_pointer_type;
		default_host_delete<value_type>()( naked_cast<raw_pointer_type>(ptr) );
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
/// \brief Allocator for device memory.
///
/// Implementation follows the specification of an STL allocator. The main
/// difference is that the CUDA API functions cudaMalloc and cudaFree are
/// used internally to allocate/deallocate memory.
///
/// Unlike the standard std::allocator or ecuda::host_allocator, the
/// allocator allocates device memory which is only accessible through device
/// code. Therefore, ecuda::device_allocator cannot be used as a replacement
/// allocator for the standard STL containers (e.g. vector).
///
template<typename T>
class device_allocator
{

public:
	typedef T                                                   value_type;      //!< element type
	typedef typename ecuda::add_pointer<T>::type                pointer;         //!< pointer to element
	typedef typename ecuda::add_lvalue_reference<T>::type       reference;       //!< reference to element
	typedef typename make_const<pointer>::type                  const_pointer;   //!< pointer to constant element
	typedef typename ecuda::add_lvalue_reference<const T>::type const_reference; //!< reference to constant element
	typedef std::size_t                                         size_type;       //!< quantities of elements
	typedef std::ptrdiff_t                                      difference_type; //!< difference between two pointers
	/// \cond DEVELOPER_DOCUMENTATION
	template<typename U> struct rebind { typedef device_allocator<U> other; }; //!< its member type U is the equivalent allocator type to allocate elements of type U
	/// \endcond

public:
	///
	/// \brief Constructs a device allocator object.
	///
	__HOST__ __DEVICE__ device_allocator() throw() {}

	///
	/// \brief Constructs a device allocator object from another device allocator object.
	/// \param alloc Allocator object.
	///
	__HOST__ __DEVICE__ device_allocator( const device_allocator& alloc ) throw() {}

	///
	/// \brief Constructs a device allocator object from another device allocator object with a different element type.
	/// \param alloc Allocator object.
	///
	template<typename U>
	__HOST__ __DEVICE__ device_allocator( const device_allocator<U>& alloc ) throw() {}

	///
	/// \brief Destructs the device allocator object.
	///
	__HOST__ __DEVICE__ ~device_allocator() throw() {}

	///
	/// \brief Returns the address of x.
	///
	/// This effectively means returning &x.
	///
	/// \param x Reference to object.
	/// \return A pointer to the object.
	///
	__HOST__ __DEVICE__ inline pointer address( reference x ) { return &x; }

	///
	/// \brief Returns the address of x.
	///
	/// This effectively means returning &x.
	///
	/// \param x Reference to object.
	/// \return A pointer to the object.
	///
	__HOST__ __DEVICE__ inline const_pointer address( const_reference x ) const { return &x; }

	///
	/// \brief Allocate block of storage.
	///
	/// Attempts to allocate a block of storage with a size large enough to contain n elements of member type
	/// value_type, and returns a pointer to the first element.
	///
	/// The storage is aligned appropriately for object of type value_type, but they are not constructed.
	///
	/// The block of storage is allocated using cudaMalloc and throws std::bad_alloc if it cannot
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
	__HOST__ pointer allocate( size_type n, std::allocator<void>::const_pointer hint = 0 )
	{
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
	__HOST__ inline void deallocate( pointer ptr, size_type n )
	{
		typedef typename ecuda::add_pointer<value_type>::type raw_pointer_type;
		default_device_delete<value_type>()( naked_cast<raw_pointer_type>(ptr) );
	}

	///
	/// \brief Returns the maximum number of elements, each of member type value_type (an alias of allocator's template parameter)
	///        that could potentially be allocated by a call to member allocate.
	///
	/// A call to member allocate with the value returned by this function can still fail to allocate the requested storage.
	///
	/// \return The nubmer of elements that might be allcoated as maximum by a call to member allocate.
	///
	__HOST__ __DEVICE__ inline size_type max_size() const throw() { return std::numeric_limits<size_type>::max(); }

	///
	/// \brief Constructs an element object on the location pointed by ptr.
	/// \param ptr Pointer to a location with enough storage space to contain an element of type value_type.
	///            pointer is a member type (defined as an alias of T* in ecuda::device_allocator<T>).
	/// \param val Value to initialize the constructed element to.
	///            const_reference is a member type (defined as an alias of T& in ecuda::device_allocator<T>).
	///
	__DEVICE__ inline void construct( pointer ptr, const_reference val ); // not supported on device

	///
	/// \brief Destroys in-place the object pointed by ptr.
	///        Notice that this does not deallocate the storage for the element (see member deallocate to release storage space).
	/// \param ptr Pointer to the object to be destroyed.
	///
	__DEVICE__ inline void destroy( pointer ptr ); // not supported on device

};

///
/// \brief Allocator for hardware aligned device memory.
///
/// Implementation follows the specification of an STL allocator. The main
/// difference is that the CUDA API functions cudaMallocPitch and cudaFree
/// are used internally to allocate/deallocate memory.
///
/// Unlike the standard std::allocator or ecuda::host_allocator, the
/// allocator allocates device memory which is only accessible through device
/// code. Therefore, ecuda::device_pitch_allocator cannot be used as a
/// replacement allocator for the standard STL containers (e.g. vector).
///
/// This allocator is NOT strictly compatible with STL specification because the
/// allocated memory is 2D and has padding to align the allocation in hardware
/// memory. The allocator requires both a width and height to specify size,
/// instead of a single length. The allocator uses the ecuda::padded_ptr
/// pointer specialization to store details on the padding of the allocated
/// memory.
///
template<typename T>
class device_pitch_allocator
{

public:
	typedef T                                                   value_type;      //!< element type
	typedef padded_ptr<T,typename ecuda::add_pointer<T>::type>  pointer;         //!< pointer to element
	typedef typename ecuda::add_lvalue_reference<T>::type       reference;       //!< reference to element
	typedef typename make_const<pointer>::type                  const_pointer;   //!< pointer to constant element
	typedef typename ecuda::add_lvalue_reference<const T>::type const_reference; //!< reference to constant element
	typedef std::size_t                                         size_type;       //!< quantities of elements
	typedef std::ptrdiff_t                                      difference_type; //!< difference between two pointers
	/// \cond DEVELOPER_DOCUMENTATION
	template<typename U> struct rebind { typedef device_allocator<U> other; }; //!< its member type U is the equivalent allocator type to allocate elements of type U
	/// \endcond

private:
	template<typename U> struct char_cast;
	template<typename U> struct char_cast<U*>       { char* type;       };
	template<typename U> struct char_cast<const U*> { const char* type; };

public:
	///
	/// \brief Constructs a device pitched memory allocator object.
	///
	__HOST__ __DEVICE__ device_pitch_allocator() throw() {}

	///
	/// \brief Constructs a device pitched memory allocator object from another host allocator object.
	/// \param alloc Allocator object.
	///
	__HOST__ __DEVICE__ device_pitch_allocator( const device_pitch_allocator& alloc ) throw() {}

	///
	/// \brief Constructs a device pitched memory allocator object from another device pitched memory allocator object with a different element type.
	/// \param alloc Allocator object.
	///
	template<typename U>
	__HOST__ __DEVICE__ device_pitch_allocator( const device_pitch_allocator<U>& alloc ) throw() {}

	///
	/// \brief Destructs the device pitched memory allocator object.
	///
	__HOST__ __DEVICE__ ~device_pitch_allocator() throw() {}

	///
	/// \brief Returns the address of x.
	///
	/// This effectively means returning &x.
	///
	/// \param x Reference to object.
	/// \return A pointer to the object.
	///
	__HOST__ __DEVICE__ inline pointer address( reference x ) { return &x; }

	///
	/// \brief Returns the address of x.
	///
	/// This effectively means returning &x.
	///
	/// \param x Reference to object.
	/// \return A pointer to the object.
	///
	__HOST__ __DEVICE__ inline const_pointer address( const_reference x ) const { return &x; }

	///
	/// \brief Allocate block of storage.
	///
	/// Attempts to allocate a block of storage with a size large enough to contain n elements of member type
	/// value_type, and returns a pointer to the first element.
	///
	/// The storage is aligned appropriately for object of type value_type, but they are not constructed.
	///
	/// The block of storage is allocated using cudaMallocPitch and throws std::bad_alloc if it cannot
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
	__HOST__ pointer allocate( size_type w, size_type h, std::allocator<void>::const_pointer hint = 0 )
	{
		typename ecuda::add_pointer<value_type>::type ptr = NULL;
		size_type pitch;
		const cudaError_t result = cudaMallocPitch( reinterpret_cast<void**>(&ptr), &pitch, w*sizeof(value_type), h );
		if( result != cudaSuccess ) throw std::bad_alloc();
		return pointer( ptr, pitch );
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
	__HOST__ inline void deallocate( pointer ptr, size_type n )
	{
		typedef typename ecuda::add_pointer<value_type>::type raw_pointer_type;
		default_device_delete<value_type>()( naked_cast<raw_pointer_type>(ptr) );
	}

	///
	/// \brief Returns the maximum number of elements, each of member type value_type (an alias of allocator's template parameter)
	///        that could potentially be allocated by a call to member allocate.
	///
	/// A call to member allocate with the value returned by this function can still fail to allocate the requested storage.
	///
	/// \return The nubmer of elements that might be allcoated as maximum by a call to member allocate.
	///
	__HOST__ __DEVICE__ inline size_type max_size() const throw() { return std::numeric_limits<size_type>::max(); }

	///
	/// \brief Constructs an element object on the location pointed by ptr.
	/// \param ptr Pointer to a location with enough storage space to contain an element of type value_type.
	///            pointer is a member type (defined as an alias of T* in ecuda::device_pitch_allocator<T>).
	/// \param val Value to initialize the constructed element to.
	///            const_reference is a member type (defined as an alias of T& in ecuda::device_pitch_allocator<T>).
	///
	__DEVICE__ inline void construct( pointer ptr, const_reference val ); // not supported on device

	///
	/// \brief Destroys in-place the object pointed by ptr.
	///        Notice that this does not deallocate the storage for the element (see member deallocate to release storage space).
	/// \param ptr Pointer to the object to be destroyed.
	///
	__DEVICE__ inline void destroy( pointer ptr ); // not supported on device

	///
	/// \brief Returns the address of a given coordinate.
	///
	/// Since pitched memory has padding at each row, the location of (x,y) is not
	/// necessarily offset by width*x+y.
	///
	/// \param ptr
	/// \param x
	/// \param y
	/// \param pitch
	/// \return A pointer to the location.
	///
	__HOST__ __DEVICE__ inline const_pointer address( const_pointer ptr, size_type x, size_type y, size_type pitch ) const
	{
		return reinterpret_cast<const_pointer>( naked_cast<const char*>(ptr) + x*pitch + y*sizeof(value_type) );
	}

	///
	/// \brief Returns the address of a given coordinate.
	///
	/// Since pitched memory has padding at each row, the location of (x,y) is not
	/// necessarily offset by width*x+y.
	///
	/// \param ptr
	/// \param x
	/// \param y
	/// \return A pointer to the location.
	///
	__HOST__ __DEVICE__ inline pointer address( pointer ptr, size_type x, size_type y )
	{
		// TODO: this is not general if this is padded_ptr<T,[some other specialized class]>
		typedef typename ecuda::add_pointer<value_type>::type raw_pointer;
		raw_pointer p = naked_cast<raw_pointer>(ptr);
		typedef typename char_cast<raw_pointer>::type char_pointer;
		char_pointer p2 = reinterpret_cast<char_pointer>(p);
		p2 += ptr.get_pitch() * x;
		p = p2;
		p += y;
		return pointer( p, ptr.get_pitch() );
	}

};

} // namespace ecuda

#endif
