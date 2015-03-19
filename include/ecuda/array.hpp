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

#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

#include "global.hpp"
#include "algorithm.hpp"
#include "allocators.hpp"
#include "apiwrappers.hpp"
#include "iterators.hpp"
#include "memory.hpp"

#ifdef __CPP11_SUPPORTED__
#include <array>
#include <initializer_list>
#endif

namespace ecuda {

///
/// A video memory-bound fixed-size array structure.
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
class array {

public:
	typedef T value_type; //!< cell data type
	typedef std::size_t size_type; //!< index data type
	typedef std::ptrdiff_t difference_type; //!<
	typedef value_type& reference; //!< cell reference type
	typedef const value_type& const_reference; //!< cell const reference type
	typedef value_type* pointer; //!< cell pointer type
	typedef const value_type* const_pointer; //!< cell const pointer type

	typedef pointer_iterator<value_type,pointer> iterator; //!< iterator type
	typedef pointer_iterator<const value_type,pointer> const_iterator; //!< const iterator type
	typedef pointer_reverse_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef pointer_reverse_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

private:
	device_ptr<T> deviceMemory; //!< smart pointer to video card memory

public:
	///
	/// \brief Constructs a fixed-size array with N elements. Each element is a copy of value.
	/// \param value Value to fill the container with.
	///
	HOST array( const value_type& value = value_type() );

	///
	/// \brief Constructs a fixed-sized array with N elements taken from the sequence [begin,end).
	///
	/// If the length of the sequence is greater than N that only the first N elements are taken. If
	/// the length of the sequence is less than N, the sequence is repeated from the start until all
	/// N elements are assigned a value.
	///
	/// \param begin, end Input iterators to the initial and final positions in a range.  The range
	///                   used is [begin,end).
	///
	template<class InputIterator>
	HOST array( InputIterator begin, InputIterator end );

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Constructs a fixed-sized array with N elements taken an initializer list.
	///
	/// If the length of the initializer list is greater than N that only the first N elements are taken. If
	/// the length of the initializer list is less than N, the initializer list is repeated from the start until all
	/// N elements are assigned a value.
	///
	/// \param il An initializer_list object. These objects are automatically constructed from initializer list
	////          declarators.
	///
	HOST array( std::initializer_list<T> il );
	#endif

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
	HOST array( const array<T,N2>& src );

	///
	/// \brief Destructs the array object.
	///
	HOST DEVICE ~array() {}

	/*
	 * Deprecating this function since the STL standard seems to specify that the at() accessor
	 * must implement range checking that throws an exception on failure.  Since exceptions are
	 * not supported within a CUDA kernel, this cannot be satisfied.
	 *
	///
	/// \brief Returns a reference to the element at specified location index, with bounds checking.
	///
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	// Below is untrue because device code doesn't allow exceptions to be thrown.
	// If index not within the range of the container, an exception of type std::out_of_range is thrown.
	DEVICE inline reference at( size_type index ) {
		//if( index >= N ) throw std::out_of_range( "ecuda::array<T,N>::at() index parameter is out of range" );
		return deviceMemory[index]; 
	}
	*/

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	DEVICE inline reference operator[]( size_type index ) { return deviceMemory[index]; }

	/*
	 * Deprecating this function since the STL standard seems to specify that the at() accessor
	 * must implement range checking that throws an exception on failure.  Since exceptions are
	 * not supported within a CUDA kernel, this cannot be satisfied.
	 *
	///
	/// \brief Returns a reference to the element at specified location index, with bounds checking.
	///
	/// If index not within the range of the container, an exception of type std::out_of_range is thrown.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	DEVICE inline const_reference at( size_type index ) const { return deviceMemory[index]; }
	*/

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	DEVICE inline const_reference operator[]( size_type index ) const { return deviceMemory[index]; }

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
	HOST DEVICE __CONSTEXPR__ inline size_type max_size() const { return std::numeric_limits<size_type>::max(); }

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(); data()+size()) is always a valid range, even
	/// if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
	HOST DEVICE inline pointer data() { return deviceMemory.get(); }

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(); data()+size()) is always a valid range, even
	/// if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
	HOST DEVICE inline const_pointer data() const { return deviceMemory.get(); }

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
	/// \brief Assigns a given value to all elements in the container.
	///
	/// \param value the value to assign to the elements
	///
	HOST DEVICE void fill( const value_type& value ) {
		#ifdef __CUDA_ARCH__
		for( iterator iter = begin(); iter != end(); ++iter ) *iter = value;
		#else
		std::vector<value_type> v( size(), value );
		CUDA_CALL( cudaMemcpy<value_type>( deviceMemory.get(), &v.front(), size(), cudaMemcpyHostToDevice ) );
		#endif
	}

	///
	/// \brief Exchanges the contents of the container with those of the other.
	///
	/// Does not cause iterators and references to associate with the other container.
	///
	/// \param other container to exchange the contents with
	///
	HOST DEVICE void swap( array<T,N>& other ) {
		#ifdef __CUDA_ARCH__
		iterator iter1 = begin();
		iterator iter2 = other.begin();
		for( ; iter1 != end(); ++iter1, ++iter2 ) ecuda::swap( *iter1, *iter2 );
		#else
		std::vector<value_type> host1; operator>>( host1 );
		std::vector<value_type> host2; other.operator>>( host2 );
		operator<<( host2 );
		other.operator<<( host1 );
		#endif
	}

	///
	/// \brief Checks if the contents of two arrays are equal.
	///
	/// That is, whether each element in the this array compares equal with the element in
	/// another array at the same position.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents are equal, false otherwise
	///
	HOST DEVICE bool operator==( const array<T,N>& other ) const {
		#ifdef __CUDA_ARCH__
		const_iterator iter1 = begin();
		const_iterator iter2 = other.begin();
		for( ; iter1 != end(); ++iter1, ++iter2 ) if( !( *iter1 == *iter2 ) ) return false;
		return true;
		#else
		#ifdef __CPP11_SUPPORTED__
		std::array<T,N> arr1, arr2;
		#else
		std::vector<T> arr1, arr2;
		#endif
		operator>>( arr1 );
		other.operator>>( arr2 );
		return arr1 == arr2;
		#endif
	}

	///
	/// \brief Checks if the contents of two arrays are not equal.
	///
	/// That is, whether any element in the this array does not compare equal with the element in
	/// another array at the same position.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents are not equal, false otherwise
	///
	HOST DEVICE inline bool operator!=( const array<T,N>& other ) const { return !operator==(other); }

	///
	/// \brief Compares the contents of two arrays lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this array are lexicographically less than the other array, false otherwise
	///
	HOST DEVICE inline bool operator<( const array<T,N>& other ) const {
		#ifdef __CUDA_ARCH__
		return ecuda::lexicographical_compare( begin(), end(), other.begin(), other.end() );
		#else
		#ifdef __CPP11_SUPPORTED__
		std::array<T,N> arr1, arr2;
		#else
		std::vector<T> arr1, arr2;
		#endif
		operator>>( arr1 );
		other.operator>>( arr2 );
		return arr1 < arr2;
		#endif
	}

	///
	/// \brief Compares the contents of two arrays lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this array are lexicographically greater than the other array, false otherwise
	///
	HOST DEVICE inline bool operator>( const array<T,N>& other ) const {
		#ifdef __CUDA_ARCH__
		return ecuda::lexicographical_compare( other.begin(), other.end(), begin(), end() );
		#else
		#ifdef __CPP11_SUPPORTED__
		std::array<T,N> arr1, arr2;
		#else
		std::vector<T> arr1, arr2;
		#endif
		operator>>( arr1 );
		other.operator>>( arr2 );
		return arr1 > arr2;
		#endif
	}

	///
	/// \brief Compares the contents of two arrays lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this array are lexicographically less than or equal to the other array, false otherwise
	///
	HOST DEVICE inline bool operator<=( const array<T,N>& other ) const { return !operator>(other); }

	///
	/// \brief Compares the contents of two arrays lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this array are lexicographically greater than or equal to the other array, false otherwise
	///
	HOST DEVICE inline bool operator>=( const array<T,N>& other ) const { return !operator<(other); }

	///
	/// \brief Copies the contents of this device array to a host STL vector.
	///
	HOST const array<T,N>& operator>>( std::vector<value_type>& vector ) const {
		vector.resize( N );
		CUDA_CALL( cudaMemcpy<value_type>( &vector.front(), deviceMemory.get(), N, cudaMemcpyDeviceToHost ) );
		return *this;
	}

	///
	/// \brief Copies the contents of a host STL vector to this device array.
	///
	HOST array<T,N>& operator<<( std::vector<value_type>& vector ) {
		if( size() < vector.size() ) throw std::out_of_range( "ecuda::array is not large enough to fit contents of provided std::vector" );
		CUDA_CALL( cudaMemcpy<value_type>( deviceMemory.get(), &vector.front(), vector.size(), cudaMemcpyHostToDevice ) );
		return *this;
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Copies the contents of this device array to a host STL array.
	///
	HOST const array<T,N>& operator>>( std::array<value_type,N>& arr ) const {
		CUDA_CALL( cudaMemcpy<value_type>( &arr.front(), deviceMemory.get(), N, cudaMemcpyDeviceToHost ) );
		return *this;
	}
	#endif

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Copies the contents of a host STL array to this device array.
	///
	HOST array<T,N>& operator<<( std::array<value_type,N>& arr ) {
		CUDA_CALL( cudaMemcpy<value_type>( deviceMemory.get(), &arr.front(), N, cudaMemcpyHostToDevice ) );
		return *this;
	}
	#endif

	///
	/// \brief Critical method to bridge host-device code.
	///
	/// This method is called during the entry to a kernel functions to generate
	/// device-bound copies of host-bound objects passed as arguments.
	/// In this case, we simply need the pointer to the start of the array.
	///
	///
	DEVICE array<T,N>& operator=( const array<T,N>& other ) {
		deviceMemory = other.deviceMemory;
		return *this;
	}

};

template<typename T,std::size_t N>
HOST array<T,N>::array( const value_type& value ) {
	deviceMemory = device_ptr<T>( DeviceAllocator<T>().allocate(N) );
	fill( value );
}

template<typename T,std::size_t N>
template<class InputIterator>
HOST array<T,N>::array( InputIterator begin, InputIterator end ) {
	deviceMemory = device_ptr<T>( DeviceAllocator<T>().allocate(N) );
	std::vector<T> v( N );
	typename std::vector<T>::size_type index = 0;
	while( index < N ) {
		for( InputIterator current = begin; current != end and index < N; ++current, ++index ) v[index] = *current;
	}
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), N, cudaMemcpyHostToDevice ) );
}

#ifdef __CPP11_SUPPORTED__
template<typename T,std::size_t N>
HOST array<T,N>::array( std::initializer_list<T> il ) {
	deviceMemory = device_ptr<T>( DeviceAllocator<T>().allocate(N) );
	std::vector<T> v( il );
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), &v.front(), N, cudaMemcpyHostToDevice ) );
}
#endif

template<typename T,std::size_t N>
template<std::size_t N2>
HOST array<T,N>::array( const array<T,N2>& src ) {
	deviceMemory = device_ptr<T>( DeviceAllocator<T>().allocate(N) );
	CUDA_CALL( cudaMemcpy<T>( deviceMemory.get(), src.data(), std::min(N,N2), cudaMemcpyDeviceToDevice ) );
}

} // namespace ecuda

#endif
