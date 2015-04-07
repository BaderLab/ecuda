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
#include "device_ptr.hpp"
#include "iterators.hpp"
#include "global.hpp"

#include "views.hpp" //TODO: change this

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION

struct __false_type {};
struct __true_type {};
template<typename T> struct __is_integer { typedef __false_type __type; };
template<> struct __is_integer<bool> { typedef __true_type __type; };
template<> struct __is_integer<char> { typedef __true_type __type; };
template<> struct __is_integer<signed char> { typedef __true_type __type; };
template<> struct __is_integer<unsigned char> { typedef __true_type __type; };
#ifdef _GLIBCXX_USE_WCHAR_T
template<> struct __is_integer<wchar_t> { typedef __true_type __type; };
#endif
#ifdef __CPP11_SUPPORTED__
template<> struct __is_integer<char16_t> { typedef __true_type __type; };
template<> struct __is_integer<char32_t> { typedef __true_type __type; };
#endif
template<> struct __is_integer<short> { typedef __true_type __type; };
template<> struct __is_integer<unsigned short> { typedef __true_type __type; };
template<> struct __is_integer<int> { typedef __true_type __type; };
template<> struct __is_integer<unsigned int> { typedef __true_type __type; };
template<> struct __is_integer<long> { typedef __true_type __type; };
template<> struct __is_integer<unsigned long> { typedef __true_type __type; };
template<> struct __is_integer<long long> { typedef __true_type __type; };
template<> struct __is_integer<unsigned long long> { typedef __true_type __type; };

/// \endcond

///
/// \brief A resizable vector stored in device memory.
///
///
///
///
template< typename T, class Alloc=device_allocator<T> >
class vector :
	private __device_sequence<
		T,
		device_ptr<
			T,
			#ifdef __CPP11_SUPPORTED__
			typename std::allocator_traits<Alloc>::pointer
			#else
			typename Alloc::pointer
			#endif
		>,
		__dimension_contiguous_tag,
		__container_type_base_tag
	>
{

private:
	typedef __device_sequence<
		T,
		device_ptr<
			T,
			#ifdef __CPP11_SUPPORTED__
			typename std::allocator_traits<Alloc>::pointer
			#else
			typename Alloc::pointer
			#endif
		>,
		__dimension_contiguous_tag,
		__container_type_base_tag
	> base_container_type;
	typedef __device_sequence<T,T*,              __dimension_contiguous_tag,__container_type_derived_tag> derived_container_type;

public:
	typedef typename base_container_type::value_type value_type; //!< element data type
	typedef Alloc allocator_type; //!< allocator type
	typedef typename base_container_type::size_type size_type; //!< unsigned integral type
	typedef typename base_container_type::difference_type difference_type; //!< signed integral type
	typedef typename base_container_type::pointer pointer; //!< element pointer type
	typedef typename base_container_type::reference reference; //!< element reference type
	typedef typename base_container_type::const_reference const_reference; //!< element const reference type

	typedef typename base_container_type::iterator iterator; //!< iterator type
	typedef typename base_container_type::const_iterator const_iterator; //!< const iterator type
	typedef typename base_container_type::reverse_iterator reverse_iterator; //!< reverse iterator type
	typedef typename base_container_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

private:
	// REMEMBER: n and m altered on device memory won't be reflected on the host object. Don't allow
	//           the device to perform any operations that change their value.
	size_type n; //!< number of elements currently stored
	//size_type m; //!< number of elements worth of memory allocated
	//device_ptr<value_type> deviceMemory; //!< smart point to video card memory
	allocator_type allocator;

private:
	HOST void growMemory( const size_type minimum ) {
		if( base_container_type::size() >= minimum ) return; // no growth neccessary
		size_type m2 = base_container_type::size();
		if( !m2 ) m2 = 1; // in case no memory is currently allocated
		while( m2 < minimum ) m2 <<= 1;
		// allocate larger chunk
		device_ptr<value_type> newMemory( allocator.allocate( m2 ) );
		base_container_type newBase( newMemory, m2 );
		// copy old data to new chunk
		if( base_container_type::data() ) newBase.copy_range_from( begin(), end(), newBase.begin() );
		base_container_type::operator=( newBase );
	}

	template<class Iterator>
	inline HOST void init( Iterator first, Iterator last, __false_type ) {
		if( iterator_category_traits<typename std::iterator_traits<Iterator>::iterator_category>::is_contiguous ) {
			n = last-first;
			growMemory( n );
			base_container_type::copy_range_from( first, last, base_container_type::begin() );
			return;
		}
		else if( iterator_category_traits<typename std::iterator_traits<Iterator>::iterator_category>::is_device ) {
			throw cuda_error( cudaErrorInvalidDevicePointer, "ecuda::vector::init() cannot initialize with non-contiguous device iterator" );
		}
		else {
			std::vector< value_type, host_allocator<value_type> > v( first, last );
			n = v.size();
			growMemory( n );
			base_container_type::copy_range_from( v.begin(), v.end(), base_container_type::begin() );
		}
	}

	HOST void init( size_type n, const value_type& value, __true_type ) {
		growMemory( n );
		if( n ) base_container_type::fill( value );
		this->n = n;
	}

public:
	///
	/// \brief Default constructor. Constructs empty container.
	/// \param allocator allocator to use for all memory allocations of this container
	///
	HOST explicit vector( const allocator_type& allocator = allocator_type() ) : base_container_type(), n(0), allocator(allocator) {}

	///
	/// \brief Constructs the container with n copies of elements with value value.
	/// \param n the size of the container
	/// \param value the value to initialize elements of the container with
	/// \param allocator allocator to use for all memory allocations of this container
	///
	HOST explicit vector( size_type n, const value_type& value, const allocator_type& allocator = allocator_type() ) : base_container_type(), n(0), allocator(allocator) {
		init( n, value, __true_type() );
	}

	///
	/// \brief Constructs the container with n default-inserted instances of T. No copies are made.
	/// \param n the size of the container
	///
	HOST explicit vector( size_type n ) : base_container_type(), n(0) {
		init( n, value_type(), __true_type() );
	}

	///
	/// \brief Constructs the container with the contents of the range [first,last).
	/// \param first,last the range to copy the elements from
	/// \param allocator allocator to use for all memory allocations of this container
	///
	template<class Iterator>
	HOST vector( Iterator first, Iterator last, const allocator_type& allocator = allocator_type() ) : base_container_type(), allocator(allocator) {
		typedef typename __is_integer<Iterator>::__type _Integral;
		init( first, last, _Integral() );
	}

	///
	/// \brief Constructs a vector with a shallow copy of each of the elements in src.
	///
	/// Be careful to note that a shallow copy means that only the pointer to the device memory
	/// that holds the elements is copied in the newly constructed container.  This allows
	/// containers to be passed-by-value to kernel functions with minimal overhead.  If a copy
	/// of the container is required in host code, use the assignment operator. For example:
	///
	/// \code{.cpp}
	/// ecuda::vector<int> vec( 10, 3 ); // create a vector of size 10 filled with 3s
	/// ecuda::vector<int> newVec( vec ); // shallow copy
	/// ecuda::vector<int> newVec; newVec = vec; // deep copy
	/// \endcode
	///
	/// \param src Another vector object of the same type, whose contents are copied.
	///
	HOST vector( const vector<value_type,allocator_type>& src ) : base_container_type(src), n(src.n), allocator(src.allocator) {}

	///
	/// \brief Copy constructor. Constructs the container with the copy of the contents of the other.
	/// \param src another container to be used as source to initialize the elements of the container with
	/// \param allocator allocator to use for all memory allocations of this container
	///
	template<class Alloc2>
	HOST vector( const vector<value_type,Alloc2>& src, const allocator_type& allocator ) : base_container_type(src), n(src.size()), allocator(allocator) {}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	///
	/// This constructor is only available if the compiler is configured to allow C++11.
	///
	/// \param src another container to be used as source to initialize the elements of the container with
	///
	HOST DEVICE vector( vector&& src ) : base_type(std::move(src)), n(std::move(src.n)) {}

	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	/// \param src another container to be used as source to initialize the elements of the container with
	/// \param allocator allocator to use for all memory allocations of this container
	///
	HOST DEVICE vector( vector&& src, const allocator_type& allocator ) : base_type(std::move(src)), n(std::move(src.n)), allocator(allocator) {}

	///
	/// \brief Constructs the container with the contents of the initializer list il.
	/// \param il initializer list to initialize the elements of the container with
	/// \param allocator allocator to use for all memory allocations of this container
	///
	HOST vector( std::initializer_list<value_type> il, const allocator_type& allocator = allocator_type() ) : n(0), allocator(allocator) {
		init( il.begin(), il.end(), __false_type() );
	}
	#endif

	//HOST DEVICE ~vector() {}

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return base_container_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return base_container_type::begin()+size(); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return base_container_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return base_container_type::begin()+size(); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(end()); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(begin()); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(end()); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(begin()); }

	///
	/// \brief Returns the number of elements in the container.
	///
	/// \returns The number of elements in the container.
	///
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return n; }

	///
	/// \brief Returns the maximum number of elements the container is able to hold due to system
	///        or library implementation limitations.
	///
	/// \returns Maximum number of elements.
	///
	HOST DEVICE __CONSTEXPR__ inline size_type max_size() const __NOEXCEPT__ { return std::numeric_limits<size_type>::max(); }

	///
	/// \brief Resizes the container to contain newSize elements.
	///
	/// If the current size is greater than newSize, the container is reduced to its first newSize
	/// elements as if by repeatedly calling pop_back().
	///
	/// \param newSize new size of the container
	/// \param value the value to initialize the new elements with
	///
	HOST void resize( size_type newSize, const value_type& value = value_type() ) {
		if( size() == newSize ) return;
		if( size() > newSize ) { n = newSize; return; }
		base_container_type oldContent( *this );
		init( newSize, value, __true_type() );
		base_container_type::copy_range_from( oldContent.begin(), oldContent.end(), base_container_type::begin() );
		n = newSize;
	}

	///
	/// \brief Returns the number of elements that the container has currently allocated space for.
	/// \return Capacity of the currently allocated storage.
	///
	HOST DEVICE inline size_type capacity() const __NOEXCEPT__ { return base_container_type::size(); }

	///
	/// \brief Checks if the container has no elements.
	///
	/// \returns true if the container is empty, false otherwise.
	///
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return !size(); }

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
	DEVICE inline reference operator[]( const size_type index ) { return base_container_type::operator[]( index ); }

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	DEVICE inline const_reference operator[]( const size_type index ) const { return base_container_type::operator[]( index ); }

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
	DEVICE inline reference front() { return operator[](0); }

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
	DEVICE inline reference back() { return operator[](size()-1); }

	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
	DEVICE inline const_reference front() const { return operator[](0); }

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
	DEVICE inline const_reference back() const { return operator[](size()-1); }

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(),data()+size()) is always a valid
	/// range, even if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
	HOST DEVICE inline pointer data() __NOEXCEPT__ { return base_container_type::data(); }

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(),data()+size()) is always a valid
	/// range, even if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
	HOST DEVICE inline const pointer data() const __NOEXCEPT__ { return base_container_type::data(); }

	///
	/// \brief Replaces the contents of the container.
	/// \param newSize the new size of the container
	/// \param value the value to initialize elements of the container with
	///
	HOST void assign( size_type newSize, const value_type& value = value_type() ) {
		growMemory(newSize); // make sure enough device memory is allocated
		init( newSize, value, __true_type() );
	}

	///
	/// \brief Replaces the contents of the container with copies of those in the range [first,last).
	/// \param first,last the range to copy the elements from
	///
	template<class Iterator>
	HOST inline void assign( Iterator first, Iterator last ) { init( first, last, __false_type() ); }

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Replaces the contents with the elements from the initializer list il.
	///
	/// This method is only available if the compiler is configured to allow C++11.
	///
	/// \param il initializer list to copy the values from
	///
	HOST void assign( std::initializer_list<value_type> il ) { init( il.begin(), il.end(), __false_type() ); }
	#endif

	///
	/// \brief Appends the given element value to the end of the container.
	/// \param value the value of the element to append
	///
	HOST void push_back( const value_type& value ) {
		growMemory(n+1);
		CUDA_CALL( cudaMemcpy<value_type>( data()+static_cast<int>(n), &value, 1, cudaMemcpyHostToDevice ) );
		++n;
	}

	/*
	 * Not implementing this function since a move operation isn't possible because the item
	 * must be copied from host to device memory.
	 *
	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Appends the given element value to the end of the container.
	///
	/// This method is only available if the compiler is configured to allow C++11.
	///
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
	HOST iterator insert( const_iterator position, const value_type& value ) {
		const size_type index = position-begin();
		std::vector< value_type, host_allocator<value_type> > v( size(), value_type() );
		base_container_type::operator>>( v );
		v.insert( v.begin()+index, value );
		growMemory(size()+1); // make sure enough device memory is allocated
		// copy expanded elements back to device
		base_container_type::copy_range_from( v.begin(), v.end(), base_container_type::begin() );
		++n;
		return begin()+index;
	}

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
	HOST iterator insert( const_iterator position, const size_type count, const value_type& value ) {
		const size_type index = position-begin();
		std::vector< value_type, host_allocator<value_type> > v( size(), value_type() );
		base_container_type::operator>>( v );
		v.insert( v.begin()+index, count, value );
		growMemory(size()+count); // make sure enough device memory is allocated
		base_container_type::copy_range_from( v.begin(), v.end(), base_container_type::begin() );
		n += count;
		return begin()+index;
	}

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

private:
	template<class Iterator>
	HOST void insert( const_iterator position, Iterator first, Iterator last, std::random_access_iterator_tag ) {
		const typename std::iterator_traits<Iterator>::difference_type newElements = std::distance(first,last);
		if( newElements <= 0 ) return;
		const size_type index = position-begin();
		std::vector< value_type, host_allocator<value_type> > v( size() );
		base_container_type::operator>>( v );
		v.insert( v.begin()+index, first, last );
		growMemory( size()+newElements );
		base_container_type::assign( v.begin(), v.end() );
		n += newElements;
	}

	template<class Iterator>
	HOST void insert( const_iterator position, Iterator first, Iterator last, std::bidirectional_iterator_tag ) {
		vector< value_type, host_allocator<value_type> > newElements( first, last );
		if( newElements.empty() ) return;
		const size_type index = position-begin();
		std::vector< value_type, host_allocator<value_type> > v( size() );
		base_container_type::operator>>( v );
		v.insert( v.begin()+index, newElements.begin(), newElements.end() );
		growMemory( size()+newElements.size() );
		base_container_type::assign( v.begin(), v.end() );
		n += newElements.size();
	}

	template<class Iterator> HOST inline void insert( const_iterator position, Iterator first, Iterator last, std::forward_iterator_tag ) { insert( first, last, std::bidirectional_iterator_tag() ); }
	template<class Iterator> HOST inline void insert( const_iterator position, Iterator first, Iterator last, std::input_iterator_tag ) { insert( first, last, std::bidirectional_iterator_tag() ); }

	template<class Iterator>
	HOST void insert( const_iterator position, Iterator first, Iterator last, contiguous_device_iterator_tag ) {
		const typename std::iterator_traits<Iterator>::difference_type newElements = last-first;
		if( newElements <= 0 ) return;
		const size_type index = position-begin();
		std::vector< value_type, host_allocator<value_type> > v1( size() );
		base_container_type::operator>>( v1 );
		vector dv( first, last, allocator );
		std::vector< value_type, host_allocator<value_type> > v2( dv.size() );
		dv.operator>>( v2 );
		v1.insert( v1.begin()+index, v2.begin(), v2.end() );
		growMemory( size()+newElements );
		base_container_type::assign( v1.begin(), v1.end() );
		n += newElements;
	}

public:
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
	template<class Iterator>
	HOST void insert( const_iterator position, Iterator first, Iterator last ) {
		if( iterator_category_traits< typename std::iterator_traits<Iterator>::iterator_category >::is_device ) {
			if( iterator_category_traits< typename std::iterator_traits<Iterator>::iterator_category >::is_contiguous ) {
				std::vector< value_type, host_allocator<value_type> > hostExistingElements( size() );
				base_container_type::copy_range_to( begin(), end(), hostExistingElements.begin() );
				typename std::iterator_traits<Iterator>::difference_type len = last-first;
				vector v( first, last );
				std::vector< value_type, host_allocator<value_type> > hostNewElements( v.size() );
				v >> hostNewElements;
				const size_type index = position-begin();
				hostExistingElements.insert( hostExistingElements.begin()+index, hostNewElements.begin(), hostNewElements.end() );
				growMemory( hostExistingElements.size() );
				base_container_type::copy_range_from( hostExistingElements.begin(), hostExistingElements.end(), base_container_type::begin() );
				n = hostExistingElements.size();
				return;
			} else {
				throw cuda_error( cudaErrorInvalidDevicePointer, "ecuda::vector::insert() cannot insert non-contiguous device elements" );
			}
		}
		std::vector< value_type, host_allocator<value_type> > hostExistingElements( size() );
		base_container_type::copy_range_to( begin(), end(), hostExistingElements.begin() );
		const size_type index = position-begin();
		hostExistingElements.insert( hostExistingElements.begin()+index, first, last );
		growMemory( hostExistingElements.size() );
		base_container_type::copy_range_to( hostExistingElements.begin(), hostExistingElements.end(), base_container_type::begin() );
		n = hostExistingElements.size();
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Inserts elements from initializer_list before position.
	///
	/// Causes reallocation if the new size() is greater than the old capacity(). If the new
	/// size() is greater than capacity(), all iterators and references are invalidated.
	/// Otherwise, only the iterators and references before the insertion point remain valid.
	/// The past-the-end iterator is also invalidated.
	///
	/// This operator is only available if the compiler is configured to allow C++11.
	///
	/// \param position iterator before which the content will be inserted. position may be the end() iterator
	/// \param il initializer list to insert values from
	///
	HOST inline void insert( const_iterator position, std::initializer_list<value_type> il ) { insert( position, il.begin(), il.end() ); }
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
	HOST iterator erase( const_iterator position ) {
		const size_type index = position-begin();
		vector v( position, end() );
		base_container_type::copy_range_from( v.begin()+1, v.end(), position );
		--n;
		return begin()+index;
	}

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
	HOST iterator erase( const_iterator first, const_iterator last ) {
		const size_type index1 = first-begin();
		const size_type index2 = last-begin();
		vector v( begin()+index2, end() );
		base_container_type::copy_range_from( v.begin(), v.end(), first );
		n -= (index2-index1);
		return (first+1);
	}

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
	HOST DEVICE void swap( vector& other ) {
		// just swap all members
		base_container_type::swap( other );
		#ifdef __CUDA_ARCH__
		ecuda::swap( n, other.n );
		#else
		std::swap( n, other.n );
		#endif
	}

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
	HOST inline allocator_type get_allocator() const { return allocator; }

	///
	/// \brief Requests the removal of unused capacity.
	///
	/// The STL specification describes this as a non-binding request to reduce capacity()
	/// to size() and it depends on the implementation if the request is fulfilled. In this
	/// implementation, the request is always fulfilled. All iterators, including the past
	/// the end iterator, are potentially invalidated.
	///
	HOST void shrink_to_fit() {
		if( size() == capacity() ) return;
		device_ptr<value_type> newMemory( allocator.allocate(size()) );
		base_container_type bt( newMemory, size() );
		bt.copy_range_from( begin(), end(), bt.begin() );
		base_container_type::operator=( bt );
	}

	///
	/// \brief Checks if the contents of two vectors are equal.
	///
	/// That is, whether size() == other.size() and each element in the this vector compares equal
	/// with the element in the other vector at the same position.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents are equal, false otherwise
	///
	HOST DEVICE inline bool operator==( const vector& other ) const {
		const derived_container_type derivedOther( other.data(), other.size() );
		return derived_container_type( data(), size() ).operator==( derivedOther );
	}

	///
	/// \brief Checks if the contents of two arrays are not equal.
	///
	/// That is, whether size() != other.size() or whether any element in the this vector does not
	/// compare equal to the element in the other vector at the same position.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents are not equal, false otherwise
	///
	HOST DEVICE inline bool operator!=( const vector& other ) const { return !operator==(other); }

	///
	/// \brief Compares the contents of two vectors lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this vector are lexicographically less than the other vector, false otherwise
	///
	HOST DEVICE inline bool operator<( const vector& other ) const {
		const derived_container_type derivedOther( other.data(), other.size() );
		return derived_container_type( data(), size() ).operator<( derivedOther );
	}

	///
	/// \brief Compares the contents of two vectors lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this vector are lexicographically greater than the other vector, false otherwise
	///
	HOST DEVICE inline bool operator>( const vector& other ) const {
		const derived_container_type derivedOther( other.data(), other.size() );
		return derived_container_type( data(), size() ).operator>( derivedOther );
	}

	///
	/// \brief Compares the contents of two vectors lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this vector are lexicographically less than or equal to the other vector, false otherwise
	///
	HOST DEVICE inline bool operator<=( const vector& other ) const { return !operator>(other); }

	///
	/// \brief Compares the contents of two vectors lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this vector are lexicographically greater than or equal to the other vector, false otherwise
	///
	HOST DEVICE inline bool operator>=( const vector& other ) const { return !operator<(other); }

	template<class Container>
	HOST const vector<value_type,allocator_type>& operator>>( Container& container ) const {
		base_container_type::copy_range_to( begin(), end(), container.begin() );
		return *this;
	}

	///
	/// \brief Copies the contents of this device vector to a host STL vector.
	///
//	template<class OtherAlloc>
//	HOST const vector<value_type,Alloc>& operator>>( std::vector<value_type,OtherAlloc>& vector ) const {
//		vector.resize( n );
//		CUDA_CALL( cudaMemcpy<value_type>( &vector.front(), deviceMemory.get(), n, cudaMemcpyDeviceToHost ) );
//		return *this;
//	}

	template<class Container>
	HOST vector<value_type,allocator_type>& operator<<( const Container& container ) {
		init( container.begin(), container.end(), __false_type() );
		return *this;
	}

	///
	/// \brief Copies the contents of a host STL vector to this device vector.
	///
//	template<class OtherAlloc>
//	HOST vector<value_type,allocator_type>& operator<<( const std::vector<value_type,OtherAlloc>& vector ) {
//		growMemory( vector.size() );
//		CUDA_CALL( cudaMemcpy<value_type>( deviceMemory.get(), &vector.front(), vector.size(), cudaMemcpyHostToDevice ) );
//		n = vector.size();
//		return *this;
//	}

	///
	/// \brief Assignment operator.
	///
	/// Copies the contents of other into this container.
	///
	/// Note that the behaviour differs depending on whether the assignment occurs on the
	/// host or the device. If called from the host, a deep copy is performed: additional
	/// memory is allocated in this container and the contents of other are copied there.
	/// If called from the device, a shallow copy is performed: the pointer to the device
	/// memory is copied only.  Therefore any changes made to this container are reflected
	/// in other as well, and vice versa.
	///
	/// \param other Container whose contents are to be assigned to this container.
	/// \return A reference to this container.
	///
	template<class Alloc2>
	HOST DEVICE vector<value_type,allocator_type>& operator=( const vector<value_type,Alloc2>& other ) {
		base_container_type::operator=( other );
		n = other.n;
//		#ifdef __CUDA_ARCH__
//		// shallow copy if called from device
//		n = other.n;
//		#else
//		// deep copy if called from host
//		deviceMemory = device_ptr<value_type>( this->allocator.allocate(m) );
//		CUDA_CALL( cudaMemcpy<value_type>( deviceMemory.get(), other.deviceMemory.get(), m, cudaMemcpyDeviceToDevice ) );
//		#endif
		return *this;
	}


};

} // namespace ecuda

#endif
