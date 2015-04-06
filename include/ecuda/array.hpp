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
// array.hpp
//
// An STL-like structure that resides in video memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ARRAY_HPP
#define ECUDA_ARRAY_HPP

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef __CPP11_SUPPORTED__
#include <array>
#include <initializer_list>
#endif

#include "global.hpp"
#include "allocators.hpp"
#include "apiwrappers.hpp"
#include "iterators.hpp"
#include "device_ptr.hpp"
#include "views.hpp"

namespace ecuda {

///
/// \brief A fixed-size array stored in device memory.
///
/// Creates a fixed size array in GPU memory.  Redeclares most of the
/// STL methods on the equivalent C++11 std::array (although this implementation
/// works with C98 compilers).  Methods are prefaced with
/// appropriate keywords to declare them as host and/or device capable.
/// In general: operations requiring memory allocation/deallocation/copying
/// are host only, operations to access the values of specific elements
/// are device only, and general information can be accessed by both.
///
template<typename T,std::size_t N>
class array : private __device_sequence<T,device_ptr<T>,__dimension_contiguous_tag,__container_type_base_tag> {

private:
	typedef __device_sequence<T,device_ptr<T>,__dimension_contiguous_tag,__container_type_base_tag> base_container_type;
	typedef __device_sequence<T,T*,           __dimension_contiguous_tag,__container_type_derived_tag> derived_container_type;

public:
	typedef typename base_container_type::value_type value_type; //!< element data type
	typedef typename base_container_type::size_type size_type; //!< unsigned integral type
	typedef typename base_container_type::difference_type difference_type; //!< signed integral type
	typedef typename base_container_type::reference reference; //!< element reference type
	typedef typename base_container_type::const_reference const_reference; //!< element const reference type
	typedef typename base_container_type::pointer pointer; //!< element pointer type
	typedef const typename base_container_type::pointer const_pointer; //!< element const pointer type

	typedef typename base_container_type::iterator iterator; //!< iterator type
	typedef typename base_container_type::const_iterator const_iterator; //!< const iterator type
	typedef typename base_container_type::reverse_iterator reverse_iterator; //!< reverse iterator type
	typedef typename base_container_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

public:
	///
	/// \brief Constructs a fixed-size array with N elements. Each element is a copy of value.
	/// \param value Value to fill the container with.
	///
	HOST array( const T& value = T() ) : base_container_type( device_ptr<T,T*>(device_allocator<value_type>().allocate(N)), N ) {
		fill( value );
	}

	///
	/// \brief Constructs a fixed-sized array with N elements taken from the sequence [begin,end).
	///
	/// If the length of the sequence is greater than N that only the first N elements are taken. If
	/// the length of the sequence is less than N, the sequence is repeated from the start until all
	/// N elements are assigned a value.
	///
	/// \param first,last Iterators to the initial and final positions in a range.  The range
	///                   used is [first,last).
	///
	template<class Iterator>
	HOST array( Iterator first, Iterator last ) : base_container_type( device_ptr<T,T*>(device_allocator<value_type>().allocate(N)), N ) {
		fill( value_type() );
		base_container_type::copy_range_from( first, last, begin() );
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Constructs a fixed-sized array with N elements taken an initializer list.
	///
	/// If the length of the initializer list is greater than N that only the first N elements are taken. If
	/// the length of the initializer list is less than N, the initializer list is repeated from the start until all
	/// N elements are assigned a value.
	///
	/// This constructor is only available if the compiler is configured to allow C++11.
	///
	/// \param il An initializer_list object. These objects are automatically constructed from initializer list
	////          declarators.
	///
	HOST array( std::initializer_list<T> il ) : base_container_type( device_ptr<T,T*>(device_allocator<value_type>().allocate(N)), N ) {
		fill( value_type() );
		base_container_type::copy_range_from( il.begin(), il.end(), begin() );
	}
	#endif

	///
	/// \brief Constructs an array with a shallow copy of each of the elements in src.
	///
	/// Be careful to note that a shallow copy means that only the pointer to the device memory
	/// that holds the elements is copied in the newly constructed container.  This allows
	/// containers to be passed-by-value to kernel functions with minimal overhead.  If a copy
	/// of the container is required in host code, use the assignment operator. For example:
	///
	/// \code{.cpp}
	/// ecuda::array<int,10> arr( 3 ); // fill array with 3s
	/// ecuda::array<int,10> newArr( arr ); // shallow copy
	/// ecuda::array<int,10> newArr; newArr = arr; // deep copy
	/// \endcode
	///
	/// \param src Another array object of the same type and size, whose contents are copied.
	///
	HOST DEVICE array( const array& src ) : base_container_type(src) {}
	
	///
	/// \brief Constructs an array with a copy of each of the elements in src, in the same order.
	///
	/// Note that the size template argument N2 in the source array can be different from the size template
	/// argument N in the constructed array.  If N2>N then only the first N elements are copied.  If N2<N then
	/// only the first N2 elements are copied while the remained are undefined (NB: this is in contrast to the
	/// behaviour of other constructors).
	///
	/// \param src Another array object of the same type (with the same class template argument T), whose contents are copied.
	///
	template<std::size_t N2>
	HOST array( const array<T,N2>& src ) : base_container_type( device_ptr<T,T*>(device_allocator<value_type>().allocate(N)), N ) {
		base_container_type::copy_range_from( src.begin(), src.begin()+(std::min(N,N2)), begin() );
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	///
	/// This constructor is only available if the compiler is configured to allow C++11.
	///
	/// \param src another container to be used as source to initialize the elements of the container with
	///
	HOST array( array<T,N>&& src ) : base_container_type(std::move(src)) {}
	#endif

	/*
	///
	/// \brief Destructs the array object.
	///
	//HOST DEVICE virtual ~array() {}
	*/

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	DEVICE inline reference operator[]( size_type index ) { return base_container_type::operator[]( index ); }

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	DEVICE inline const_reference operator[]( size_type index ) const { return base_container_type::operator[]( index ); }

	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
	DEVICE inline reference front() { return base_container_type::operator[](0); }

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
	DEVICE inline reference back() { return base_container_type::operator[](size()-1); }
	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
	DEVICE inline const_reference front() const { return base_container_type::operator[](0); }

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
	DEVICE inline const_reference back() const { return base_container_type::operator[](size()-1); }

	///
	/// \brief Checks if the container has no elements.
	///
	/// \returns true if the container is empty, false otherwise.
	///
	HOST DEVICE __CONSTEXPR__ inline bool empty() const { return N == 0; }

	///
	/// \brief Returns the number of elements in the container.
	///
	/// \returns The number of elements in the container.
	///
	HOST DEVICE __CONSTEXPR__ inline size_type size() const { return N; }

	///
	/// \brief Returns the maximum number of elements the container is able to hold due to system
	/// or library implementation limitations.
	///
	/// \returns Maximum number of elements.
	///
	HOST __CONSTEXPR__ inline size_type max_size() const { return std::numeric_limits<size_type>::max(); }

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(); data()+size()) is always a valid range, even
	/// if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
	HOST DEVICE inline pointer data() { return base_container_type::data(); }

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(); data()+size()) is always a valid range, even
	/// if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
	HOST DEVICE inline const_pointer data() const { return base_container_type::data(); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	HOST DEVICE inline iterator begin() { return base_container_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	HOST DEVICE inline iterator end() { return base_container_type::end(); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	HOST DEVICE inline const_iterator begin() const { return base_container_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	HOST DEVICE inline const_iterator end() const { return base_container_type::end(); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	HOST DEVICE inline reverse_iterator rbegin() { return base_container_type::rbegin(); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	HOST DEVICE inline reverse_iterator rend() { return base_container_type::rend(); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	HOST DEVICE inline const_reverse_iterator rbegin() const { return base_container_type::rbegin(); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	HOST DEVICE inline const_reverse_iterator rend() const { return base_container_type::rend(); }

	///
	/// \brief Assigns a given value to all elements in the container.
	///
	/// \param value the value to assign to the elements
	///
	HOST DEVICE inline void fill( const value_type& value ) { base_container_type::fill( value ); }

	///
	/// \brief Exchanges the contents of the container with those of the other.
	///
	/// Does not cause iterators and references to associate with the other container.
	///
	/// \param other container to exchange the contents with
	///
	HOST DEVICE inline void swap( array<T,N>& other ) { base_container_type::swap( other ); }

	///
	/// \brief Checks if the contents of two arrays are equal.
	///
	/// That is, whether each element in the this array compares equal with the element in
	/// another array at the same position.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents are equal, false otherwise
	///
	HOST DEVICE bool operator==( const array<T,N>& other ) const { return base_container_type::operator==( other ); }

	///
	/// \brief Checks if the contents of two arrays are not equal.
	///
	/// That is, whether any element in the this array does not compare equal with the element in
	/// another array at the same position.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents are not equal, false otherwise
	///
	HOST DEVICE inline bool operator!=( const array<T,N>& other ) const { return base_container_type::operator!=( other ); }

	///
	/// \brief Compares the contents of two arrays lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this array are lexicographically less than the other array, false otherwise
	///
	HOST DEVICE inline bool operator<( const array<T,N>& other ) const { return base_container_type::operator<( other ); }

	///
	/// \brief Compares the contents of two arrays lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this array are lexicographically greater than the other array, false otherwise
	///
	HOST DEVICE inline bool operator>( const array<T,N>& other ) const { return base_container_type::operator>( other ); }

	///
	/// \brief Compares the contents of two arrays lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this array are lexicographically less than or equal to the other array, false otherwise
	///
	HOST DEVICE inline bool operator<=( const array<T,N>& other ) const { return base_container_type::operator<=( other ); }

	///
	/// \brief Compares the contents of two arrays lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this array are lexicographically greater than or equal to the other array, false otherwise
	///
	HOST DEVICE inline bool operator>=( const array<T,N>& other ) const { return base_container_type::operator>=( other ); }

	///
	/// \brief Copies the contents of this device array to another container.
	///
	template<class Container>
	HOST const array<T,N>& operator>>( Container& container ) const {
		base_container_type::operator>>( container );
		return *this;
	}

	///
	/// \brief Copies the contents of a host STL vector to this device array.
	///
	/// \param vector std::vector to copy the contents from
	/// \exception std::length_error thrown if this array is not large enough to hold the given vector's contents
	///
	template<class Container>
	HOST array<T,N>& operator<<( const Container& container ) {
		base_container_type::copy_range_from( container.begin(), container.end(), begin() );
		return *this;
	}

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
	HOST DEVICE inline array<T,N>& operator=( const array<T,N>& other ) {
		base_container_type::operator=( other );
		return *this;
	}

};

} // namespace ecuda

#endif
