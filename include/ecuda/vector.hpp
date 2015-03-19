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
// vector.hpp
// An STL-like structure that resides in video memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_VECTOR_HPP
#define ECUDA_VECTOR_HPP

#include <cstddef>
#include <iterator>
#include <limits>
#include <vector>

#ifdef __CPP11_SUPPORTED__
#include <initializer_list>
#include <memory>
#include <utility>
#endif

#include "algorithm.hpp"
#include "allocators.hpp"
#include "apiwrappers.hpp"
#include "iterators.hpp"
#include "global.hpp"
#include "memory.hpp"

namespace ecuda {

///
/// A video memory-bound vector structure.
///
template< typename T, class Alloc=DeviceAllocator<T> >
class vector {

public:
	typedef T value_type; //!< cell data type
	typedef Alloc allocator_type; //!< allocator type
	typedef std::size_t size_type; //!< unsigned integral type
	typedef std::ptrdiff_t difference_type; //!< signed integer type
	#ifdef __CPP11_SUPPORTED__
	typedef value_type& reference; //!< cell reference type
	typedef const value_type& const_reference; //!< cell const reference type
	typedef typename std::allocator_traits<Alloc>::pointer pointer; //!< cell pointer type
	typedef typename std::allocator_traits<Alloc>::const_pointer const_pointer; //!< cell const pointer type
	#else
	typedef typename Alloc::reference reference; //!< cell reference type
	typedef typename Alloc::const_reference const_reference; //!< cell const reference type
	typedef typename Alloc::pointer pointer; //!< cell pointer type
	typedef typename Alloc::const_pointer const_pointer; //!< cell const pointer type
	#endif

	typedef pointer_iterator<value_type,pointer> iterator; //!< iterator type
	typedef pointer_iterator<const value_type,pointer> const_iterator; //!< const iterator type
	typedef std::reverse_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

private:
	// REMEMBER: n and m altered on device memory won't be reflected on the host object. Don't allow
	//           the device to perform any operations that change their value.
	size_type n; //!< number of elements currently stored
	size_type m; //!< number of elements worth of memory allocated
	device_ptr<T> deviceMemory; //!< smart point to video card memory
	allocator_type allocator;

private:
	HOST void growMemory( size_type minimum );

public:
	///
	/// \brief Default constructor. Constructs empty container.
	/// \param allocator allocator to use for all memory allocations of this container
	///
	HOST explicit vector( const allocator_type& allocator = allocator_type() ) : n(0), m(0), allocator(allocator) {}

	///
	/// \brief Constructs the container with n copies of elements with value value.
	/// \param n the size of the container
	/// \param value the value to initialize elements of the container with
	/// \param allocator allocator to use for all memory allocations of this container
	///
	HOST explicit vector( size_type n, const value_type& value, const allocator_type& allocator = allocator_type() );

	///
	/// \brief Constructs the container with n default-inserted instances of T. No copies are made.
	/// \param n the size of the container
	///
	HOST explicit vector( size_type n ) : n(n), m(m) {
		if( n ) {
			growMemory( n );
			std::vector<T> v( n );
			CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), n, cudaMemcpyHostToDevice ) );
		}
	}

	///
	/// \brief Constructs the container with the contents of the range [begin,end).
	/// \param begin,end the range to copy the elements from
	/// \param allocator allocator to use for all memory allocations of this container
	///
	template<class InputIterator>
	HOST vector( InputIterator begin, InputIterator end, const allocator_type& allocator = allocator_type() ) : allocator(allocator) {
		std::vector<value_type> v( begin, end );
		growMemory( v.size() );
		CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	}

	///
	/// \brief Copy constructor. Constructs the container with the copy of the contents of the other.
	/// \param src another container to be used as source to initialize the elements of the container with
	///
	HOST vector( const vector<T>& src ) : n(src.n), m(src.m),
		#ifdef __CPP11_SUPPORTED__
		allocator(std::allocator_traits<allocator_type>::select_on_container_copy_construction(src.get_allocator()))
		#else
		allocator(src.allocator)
		#endif
	{
		deviceMemory = device_ptr<T>( this->allocator.allocate(m) );
		CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), src.deviceMemory.get(), m, cudaMemcpyDeviceToDevice ) );
	}

	///
	/// \brief Copy constructor. Constructs the container with the copy of the contents of the other.
	/// \param src another container to be used as source to initialize the elements of the container with
	/// \param allocator allocator to use for all memory allocations of this container
	///
	template<class Alloc2>
	HOST vector( const vector<T,Alloc2>& src, const allocator_type& allocator ) : n(src.n), m(src.m), allocator(allocator) {
		deviceMemory = device_ptr<T>( this->allocator.allocate(m) );
		CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), src.deviceMemory.get(), m, cudaMemcpyDeviceToDevice ) );
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	/// \param src another container to be used as source to initialize the elements of the container with
	///
	HOST DEVICE vector( vector&& src ) : n(src.n), m(src.m), deviceMemory(std::move(src.deviceMemory)), allocator(std::move(src.allocator)) {}

	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	/// \param src another container to be used as source to initialize the elements of the container with
	/// \param allocator allocator to use for all memory allocations of this container
	///
	HOST DEVICE vector( vector&& src, const allocator_type& allocator ) : n(src.n), m(src.m), deviceMemory(std::move(src.deviceMemory)), allocator(allocator) {}

	///
	/// \brief Constructs the container with the contents of the initializer list il.
	/// \param il initializer list to initialize the elements of the container with
	/// \param allocator allocator to use for all memory allocations of this container
	///
	HOST DEVICE vector( std::initializer_list<value_type> il, const allocator_type& allocator = allocator_type() ) : n(0), m(0), allocator(allocator) {
		std::vector<value_type> v( il );
		growMemory( v.size() );
		CUDA_CALL( cudaMemcpy<value_type>( deviceMemory.get(), v.data(), v.size(), cudaMemcpyHostToDevice ) );
	}
	#endif

	//HOST vector( const std::vector<T>& src, const allocator_type& allocator = allocator_type() );

	HOST DEVICE ~vector() {}

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	HOST DEVICE inline iterator begin() { return iterator(deviceMemory.get()); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	HOST DEVICE inline iterator end() { return iterator(deviceMemory.get()+size()); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	HOST DEVICE inline const_iterator begin() const { return const_iterator(deviceMemory.get()); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	HOST DEVICE inline const_iterator end() const { return const_iterator(deviceMemory.get()+size()); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	HOST DEVICE inline reverse_iterator rbegin() { return reverse_iterator(end()); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	HOST DEVICE inline reverse_iterator rend() { return reverse_iterator(begin()); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	HOST DEVICE inline const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	HOST DEVICE inline const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

	///
	/// \brief Returns the number of elements in the container.
	///
	/// \returns The number of elements in the container.
	///
	HOST DEVICE inline size_type size() const { return n; }

	///
	/// \brief Returns the maximum number of elements the container is able to hold due to system
	/// or library implementation limitations.
	///
	/// \returns Maximum number of elements.
	///
	HOST DEVICE __CONSTEXPR__ inline size_type max_size() const { return std::numeric_limits<size_type>::max(); }

	///
	/// \brief Resizes the container to contain newSize elements.
	///
	/// If the current size is greater than newSize, the container is reduced to its first newSize
	/// elements as if by repeatedly calling pop_back().
	///
	/// \param newSize new size of the container
	/// \param value the value to initialize the new elements with
	///
	HOST void resize( size_type newSize, value_type& value = value_type() );

	///
	/// \brief Returns the number of elements that the container has currently allocated space for.
	/// \return Capacity of the currently allocated storage.
	///
	HOST DEVICE inline size_type capacity() const { return m; }

	///
	/// \brief Checks if the container has no elements.
	///
	/// \returns true if the container is empty, false otherwise.
	///
	HOST DEVICE inline bool empty() const { return !n; }

	///
	/// \brief Increase the capacity of the container to a value that's greater or equal to newCapacity.
	///
	/// If newCapacity is greater than the current capacity(), new storage is allocated, otherwise the
	/// method does nothing.
	///
	/// If newCapacity is greater than capacity(), all iterators and references, including the past-the-end
	/// iterator, are invalidated. Otherwise, no iterators or references are invalidated.
	///
	/// \param newCapacity new capacity of the container
	///
	HOST inline void reserve( size_type newCapacity ) { growMemory(newCapacity); }

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	DEVICE inline reference operator[]( const size_type index ) { return deviceMemory[index]; }

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	DEVICE inline const_reference operator[]( const size_type index ) const { return deviceMemory[index]; }

	/*
	 * Deprecating these functions since the STL standard seems to specify that at() accessors
	 * must implement range checking that throws an exception on failure.  Since exceptions are
	 * not supported within a CUDA kernel, this cannot be satisfied.
	 *
	DEVICE inline reference at( const size_type index ) { return deviceMemory[index]; }
	DEVICE inline const_reference at( const size_type index ) const { return deviceMemory[index]; }
	*/

	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
	DEVICE inline reference front() { return *deviceMemory; }

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
	DEVICE inline reference back() { return operator[]( size()-1 ); }

	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
	DEVICE inline const_reference front() const { return *deviceMemory; }

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
	DEVICE inline const_reference back() const { return operator[]( size()-1 ); }

	///
	/// \brief Replaces the contents of the container.
	/// \param newSize the new size of the container
	/// \param value the value to initialize elements of the container with
	///
	HOST void assign( size_type newSize, value_type& value = value_type() );

	///
	/// \brief Replaces the contents of the container with copies of those in the range [begin,end).
	/// \param begin,end the range to copy the elements from
	///
	template<class InputIterator>
	HOST void assign( InputIterator first, InputIterator last );

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Replaces the contents with the elements from the initializer list il.
	/// \param il initializer list to copy the values from
	///
	HOST void assign( std::initializer_list<value_type> il );
	#endif


	///
	/// \brief Appends the given element value to the end of the container.
	/// \param value the value of the element to append
	///
	HOST void push_back( const value_type& value );

	/*
	 * Not implementing this function since a move operation isn't possible because the item
	 * must be copied from host to device memory.
	 *
	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Appends the given element value to the end of the container.
	/// \param value the value of the element to append
	///
	HOST void push_back( value_type&& value );
	#endif
	*/

	///
	/// \brief Removes the last element of the container.
	///
	/// Calling pop_back on an empty container is undefined. No iterators or references
	/// expect for back() and end() are invalidated.
	///
	/// Although this can be called from both the host and device, a call from the
	/// device removes the last element of the container in the calling thread only.
	///
	HOST DEVICE inline void pop_back() { /*if( n )*/ --n; } // NOTE: if called from device the host instance doesn't change

	///
	/// \brief Inserts value before position.
	///
	/// Causes reallocation if the new size() is greater than the old capacity(). If the new
	/// size() is greater than capacity(), all iterators and references are invalidated.
	/// Otherwise, only the iterators and references before the insertion point remain valid.
	/// The past-the-end iterator is also invalidated.
	///
	/// \param position iterator before which the content will be inserted. position may be the end() iterator
	/// \param value element value to insert
	/// \return iterator pointing to the inserted value
	///
	HOST iterator insert( const_iterator position, const value_type& value );

	///
	/// \brief Inserts count copies of the value before position.
	///
	/// Causes reallocation if the new size() is greater than the old capacity(). If the new
	/// size() is greater than capacity(), all iterators and references are invalidated.
	/// Otherwise, only the iterators and references before the insertion point remain valid.
	/// The past-the-end iterator is also invalidated.
	///
	/// \param position iterator before which the content will be inserted. position may be the end() iterator
	/// \param count number of copies of value to insert
	/// \param value element value to insert
	/// \return iterator pointing to the inserted value
	///
	HOST iterator insert( const_iterator position, const size_type count, const value_type& value );

	/*
	 * Not implementing this function since a move operation isn't possible because the item
	 * must be copied from host to device memory.
	 *
	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Inserts count copies of the value before position.
	///
	/// Causes reallocation if the new size() is greater than the old capacity(). If the new
	/// size() is greater than capacity(), all iterators and references are invalidated.
	/// Otherwise, only the iterators and references before the insertion point remain valid.
	/// The past-the-end iterator is also invalidated.
	///
	/// \param position iterator before which the content will be inserted. position may be the end() iterator
	/// \param value element value to insert
	/// \return iterator pointing to the inserted value
	///
	HOST iterator insert( const_iterator position, value_type&& value );
	#endif
	*/

	///
	/// \brief Inserts elements from range [first,last) before position.
	///
	/// Causes reallocation if the new size() is greater than the old capacity(). If the new
	/// size() is greater than capacity(), all iterators and references are invalidated.
	/// Otherwise, only the iterators and references before the insertion point remain valid.
	/// The past-the-end iterator is also invalidated.
	///
	/// \param position iterator before which the content will be inserted. position may be the end() iterator
	/// \param first,last the range of elements to insert, can't be iterators into container for which insert is called
	///
	template<class InputIterator>
	HOST void insert( const_iterator position, InputIterator first, InputIterator last );

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Inserts elements from initializer_list before position.
	///
	/// Causes reallocation if the new size() is greater than the old capacity(). If the new
	/// size() is greater than capacity(), all iterators and references are invalidated.
	/// Otherwise, only the iterators and references before the insertion point remain valid.
	/// The past-the-end iterator is also invalidated.
	///
	/// \param position iterator before which the content will be inserted. position may be the end() iterator
	/// \param il initializer list to insert values from
	///
	HOST inline void insert( const_iterator position, std::initializer_list<value_type> il ) { return insert( position, il.begin(), il.end() ); }
	#endif

	///
	/// \brief Removes the element at position.
	///
	/// Invalidates iterators and references at or after the point of the erase, including
	/// the end() iterator. The iterator position must be valid and dereferencable. Thus the
	/// end() iterator (which is valid, but not dereferencable) cannot be used as a value
	/// for position.
	///
	/// \param position iterator to the element to remove
	/// \returns Iterator following the last removed element. If the iterator position refers
	///          to the last element, the end() iterator is returned.
	///
	HOST iterator erase( const_iterator position );

	///
	/// \brief Removes the elements in the range [first,last).
	///
	/// Invalidates iterators and references at or after the point of the erase, including
	/// the end() iterator. The iterator position must be valid and dereferencable. Thus the
	/// end() iterator (which is valid, but not dereferencable) cannot be used as a value
	/// for position. The iterator first does not need to be dereferencable if first==last:
	/// erasing an empty range is a no-op.
	///
	/// \param first,last range of elements to remove
	/// \returns Iterator following the last removed element. If the iterator position refers
	///          to the last element, the end() iterator is returned.
	///
	HOST iterator erase( const_iterator first, const_iterator last );

	///
	/// \brief Exchanges the contents of the container with those of the other.
	///
	/// Does not invoke any move, copy, or swap operations on individual elements. All iterators
	/// and references remain valid. The past-the-end iterator is invalidated.
	///
	/// Although this can be called from both the host and device, a call from the
	/// device only swaps the contents of the containers in the calling thread only.
	///
	/// \param other container to exchange the contents with
	///
	HOST DEVICE void swap( vector& other );

	///
	/// \brief Removes all elements from the container.
	///
	/// Invalidates all references, pointers, or iterators referring to contained elements.
	/// May invalidate any past-the-end iterators. Leaves the capacity() of the vector unchanged.
	///
	/// Although this can be called from both the host and device, a call from the
	/// device only clears the contents of the container in the calling thread only.
	///
	HOST DEVICE inline void clear() { n = 0; }

	///
	/// \brief Returns the allocator associated with the container.
	/// \returns The associated allocator.
	///
	HOST inline allocator_type get_allocator() const { return allocator_type(); }

	// Conversions from other container types:
	template<class OtherAlloc>
	HOST const vector<T,Alloc>& operator>>( std::vector<T,OtherAlloc>& vector ) const {
		vector.resize( n );
		CUDA_CALL( cudaMemcpy<T>( &vector.front(), deviceMemory.get(), n, cudaMemcpyDeviceToHost ) );
		//CUDA_CALL( cudaMemcpy( &vector[0], deviceMemory.get(), n*sizeof(T), cudaMemcpyDeviceToHost ) );
		return *this;
	}

	HOST vector<T,Alloc>& operator<<( std::vector<T>& vector ) {
		if( size() < vector.size() ) throw std::out_of_range( "ecuda::array is not large enough to fit contents of provided std::vector" );
		CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &vector.front(), vector.size(), cudaMemcpyHostToDevice ) );
		//CUDA_CALL( cudaMemcpy( deviceMemory.get(), &vector[0], vector.size()*sizeof(T), cudaMemcpyHostToDevice ) );
		return *this;
	}

	// critical function used to bridge host->device code
	DEVICE vector<T,Alloc>& operator=( const vector<T>& other ) {
		n = other.n;
		deviceMemory = other.deviceMemory;
		return *this;
	}

};


template<typename T,class Alloc>
HOST void vector<T,Alloc>::growMemory( size_type minimum ) {
	if( m >= minimum ) return; // no growth neccessary
	size_type m2 = m;
	if( !m2 ) m2 = 1; // in case no memory is currently allocated
	while( m2 < minimum ) m2 <<= 1;
	// allocate larger chunk
	device_ptr<T> newMemory( allocator.allocate( m2 ) );
	// copy old data to new chunk
	if( deviceMemory ) CUDA_CALL( cudaMemcpy<T>( newMemory.get(), deviceMemory.get(), m, cudaMemcpyDeviceToDevice ) );
	deviceMemory = newMemory;
	m = m2;
}

template<typename T,class Alloc>
HOST vector<T,Alloc>::vector( size_type n, const value_type& value, const allocator_type& allocator ) : n(n), allocator(allocator) {
	m = 1; while( m < n ) m <<= 1;
	#ifndef __CUDA_ARCH__
	if( n ) {
		deviceMemory = device_ptr<T>( this->allocator.allocate( m ) );
		std::vector<T> v( n, value );
		CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), m, cudaMemcpyHostToDevice ) );
	}
	#endif
}

//template<typename T,class Alloc>
//HOST vector<T,Alloc>::vector( const std::vector<T>& src, const allocator_type& allocator ) : n(src.size()), allocator(allocator) {
//	m = 1; while( m < n ) m <<= 1;
//	deviceMemory = device_ptr<T>( this->allocator.allocate( m ) );
//	if( n ) CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &src.front(), n, cudaMemcpyHostToDevice ) );
//}

template<typename T,class Alloc>
HOST void vector<T,Alloc>::resize( size_type newSize, value_type& value ) {
	growMemory(newSize); // make sure enough device memory is allocated
	std::vector<T> v( newSize-n, value );
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+n, &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	n = newSize;
}

template<typename T,class Alloc>
HOST vector<T,Alloc>::iterator vector<T,Alloc>::insert( vector<T,Alloc>::const_iterator position, const value_type& val ) {
	growMemory(n+1); // make sure enough device memory is allocated
	const size_type index = position-begin();
	std::vector<T,Alloc> v( n-index+1, T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-insertion region to host
	if( v.size() > 1 ) CUDA_CALL( cudaMemcpy<T>( &v[1], deviceMemory.get()+index, (n-index), cudaMemcpyDeviceToHost ) ); // copy post-insertion region to +1 in host vector
	v.front() = val; // add new value to position 0
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+index, &v.front(), v.size(), cudaMemcpyHostToDevice ) ); // copy +1 inserted data back onto device
	++n;
	return position; // since iterators are index based, the return iterator is the same as the one provided
}

template<typename T,class Alloc>
HOST vector<T,Alloc>::iterator vector<T,Alloc>::insert( vector<T,Alloc>::const_iterator position, const size_type span, const value_type& val ) {
	growMemory(n+span); // make sure enough device memory is allocated
	const size_type index = position-begin();
	std::vector< T, HostAllocator<T> > v( n-index+span, T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-insertion region to host
	if( v.size() > span ) CUDA_CALL( cudaMemcpy<T>( &v[span], deviceMemory.get()+index, (n-index), cudaMemcpyDeviceToHost ) ); // copy post-insertion region to +span in host vector
	for( size_type i = 0; i < span; ++i ) v[i] = val; // fill in start region with new value
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+index, &v.front(), v.size(), cudaMemcpyHostToDevice ) ); // copy +span inserted data back onto device
	n += span;
	return position; // since iterators are index based, the return iterator is the same as the one provided
}

template<typename T,class Alloc>
template<class InputIterator>
HOST void vector<T,Alloc>::insert( vector<T,Alloc>::const_iterator position, InputIterator first, InputIterator last ) {
	const std::vector<T> x( first, last );
	growMemory(n+x.size());
	const size_type index = position-begin();
	std::vector< T,HostAllocator<T> > v( n-index+x.size(), T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-insertion region to host
	if( v.size() > x.size() ) CUDA_CALL( cudaMemcpy<T>( &v[x.size()], deviceMemory.get()+index, (n-index), cudaMemcpyDeviceToHost ) ); // copy post-insertion region to +span in host vector
	for( size_type i = 0; i < x.size(); ++i ) v[i] = x[i]; // fill in start region with new values
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+index, &v.front(), v.size(), cudaMemcpyHostToDevice ) ); // copy +x.size() inserted data back onto device
	n += x.size();
}

template<typename T,class Alloc>
HOST vector<T,Alloc>::iterator vector<T,Alloc>::erase( vector<T,Alloc>::const_iterator position ) {
	const size_type index = position-begin();
	std::vector< T,HostAllocator<T> > v( n-index-1, T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-erasure region to host
	CUDA_CALL( cudaMemcpy<T>( &v.front(), deviceMemory.get()+(index+1), v.size(), cudaMemcpyDeviceToHost ) );
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+index, &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	--n;
	return iterator(begin()+index);
}

template<typename T,class Alloc>
HOST vector<T,Alloc>::iterator vector<T,Alloc>::erase( vector<T,Alloc>::const_iterator first, vector<T,Alloc>::const_iterator last ) {
	const size_type index1 = first-begin();
	const size_type index2 = last-begin();
	std::vector< T,HostAllocator<T> > v( n-index2, T(), HostAllocator<T>() ); // allocate pinned memory to transfer post-erasure region to host
	CUDA_CALL( cudaMemcpy<T>( &v.front(), deviceMemory.get()+index2, v.size(), cudaMemcpyDeviceToHost ) );
	CUDA_CALL( cudaMemcpy<T>( &v.front(), deviceMemory.get()+index2, v.size(), cudaMemcpyDeviceToHost ) );
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+index1, &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	n -= index2-index1;
	return iterator(begin()+index1);
}

template<typename T,class Alloc>
HOST DEVICE void vector<T,Alloc>::swap( vector<T,Alloc>& other ) {
	// just swap all members
	#ifdef __CUDA_ARCH__
	ecuda::swap( n, other.n );
	ecuda::swap( m, other.m );
	ecuda::swap( deviceMemory, other.deviceMemory );
	#else
	std::swap( n, other.n );
	std::swap( m, other.m );
	std::swap( deviceMemory, other.deviceMemory );
	#endif
}

template<typename T,class Alloc>
HOST void vector<T,Alloc>::assign( size_type newSize, value_type& value ) {
	growMemory(newSize); // make sure enough device memory is allocated
	std::vector<T> v( newSize, value );
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	//CUDA_CALL( cudaMemcpy( deviceMemory.get(), &v[0], v.size()*sizeof(T), cudaMemcpyHostToDevice ) );
	n = newSize;
}

template<typename T,class Alloc>
template<class InputIterator>
HOST void vector<T,Alloc>::assign( InputIterator first, InputIterator last ) {
	std::vector<T> v( first, last );
	growMemory( v.size() ); // make sure enough device memory is allocated
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	//CUDA_CALL( cudaMemcpy( deviceMemory.get(), &v[0], v.size()*sizeof(T), cudaMemcpyHostToDevice ) );
	n = v.size();
}

#ifdef __CPP11_SUPPORTED__
template<typename T,class Alloc>
HOST void vector<T,Alloc>::assign( std::initializer_list<value_type> il ) {
	std::vector<T> v( il );
	growMemory( v.size() ); // make sure enough device memory is allocated
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	n = v.size();
}
#endif


template<typename T,class Alloc>
HOST void vector<T,Alloc>::push_back( const value_type& v ) {
	growMemory(n+1);
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get()+(n*sizeof(T)), &v, 1, cudaMemcpyHostToDevice ) );
	//CUDA_CALL( cudaMemcpy( deviceMemory.get()+(n*sizeof(T)), &v, sizeof(v), cudaMemcpyHostToDevice ) );
	++n;
}

} // namespace ecuda

#endif
