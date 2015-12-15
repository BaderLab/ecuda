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
<<<<<<< HEAD
#include <vector>
//#ifdef __CPP11_SUPPORTED__
//#include <array>
//#include <initializer_list>
//#endif

#include "global.hpp"
#include "algorithm.hpp"
#include "allocators.hpp"
#include "apiwrappers.hpp"
#include "iterators.hpp"
#include "device_ptr.hpp"

namespace ecuda {

=======
#ifdef __CPP11_SUPPORTED__
#include <utility>
#endif

#include "global.hpp"
#include "algorithm.hpp"   // for copy
#include "allocators.hpp"  // for device_allocator
#include "memory.hpp"      // for shared_ptr
#include "impl/models.hpp" // for impl::device_fixed_sequence

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<typename T,std::size_t N> class array_kernel_argument; // forward declaration

} // namespace impl
/// \endcond

>>>>>>> ecuda2/master
///
/// \brief A fixed-size array stored in device memory.
///
/// Creates a fixed size array in GPU memory.  Redeclares most of the STL methods on the equivalent
/// C++11 std::array (although this implementation works with C98 compilers).
///
/// Methods are prefaced with appropriate keywords to declare them as host and/or device capable.
/// In general: operations requiring memory allocation/deallocation are host only, operations
/// to access the values of specific elements are device only, and copy operations on ranges of data and
/// accessors of general information can be performed on both the host and device.
///
<<<<<<< HEAD
template<typename T,std::size_t N>
class array {

public:
	typedef T value_type; //!< cell data type
	typedef std::size_t size_type; //!< unsigned integral type
	typedef std::ptrdiff_t difference_type; //!< signed integral type
	typedef value_type& reference; //!< cell reference type
	typedef const value_type& const_reference; //!< cell const reference type
	typedef value_type* pointer; //!< cell pointer type
	typedef const value_type* const_pointer; //!< cell const pointer type

	typedef contiguous_device_iterator<value_type> iterator; //!< iterator type
	typedef contiguous_device_iterator<const value_type> const_iterator; //!< const iterator type
	typedef reverse_device_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

private:
	device_ptr<value_type> deviceMemory; //!< smart pointer to video card memory
=======
template< typename T, std::size_t N, class P=shared_ptr<T> >
class array : private impl::device_fixed_sequence< T, N, P > {

private:
	typedef impl::device_fixed_sequence< T, N, P > base_type;

public:
	typedef typename base_type::value_type      value_type;      //!< cell data type
	typedef typename base_type::size_type       size_type;       //!< unsigned integral type
	typedef typename base_type::difference_type difference_type; //!< signed integral type
	typedef typename base_type::reference       reference;       //!< cell reference type
	typedef typename base_type::const_reference const_reference; //!< cell const reference type
	typedef typename base_type::pointer         pointer;         //!< cell pointer type
	typedef typename make_const<pointer>::type  const_pointer;   //!< cell const pointer type

	typedef typename base_type::iterator               iterator;               //!< iterator type
	typedef typename base_type::const_iterator         const_iterator;         //!< const iterator type
	typedef typename base_type::reverse_iterator       reverse_iterator;       //!< reverse iterator type
	typedef typename base_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

	typedef impl::array_kernel_argument<T,N> kernel_argument; //!< kernel argument type

	template<typename U,std::size_t M,class Q> friend class array;

protected:
	///
	/// \brief Used by the kernel_argument subclass to create a shallow copy using an unmanaged pointer.
	///
	template<class Q> __HOST__ __DEVICE__ array( const array<T,N,Q>& src, ecuda::true_type ) : base_type( unmanaged_cast(src.get_pointer()) ) {}

	///
	/// \brief Used by the kernel_argument subclass to create a shallow copy using an unmanaged pointer.
	///
	template<class Q>
	__HOST__ __DEVICE__ array& shallow_assign( const array<T,N,Q>& other )
	{
		base_type::get_pointer() = unmanaged_cast(other.get_pointer());
		return *this;
	}
>>>>>>> ecuda2/master

public:
	///
	/// \brief Constructs a fixed-size array with N elements.
	///
	/// Each element is a default-initialized value of T.
	///
<<<<<<< HEAD
	HOST array() {
		deviceMemory = device_ptr<value_type>( device_allocator<value_type>().allocate(N) );
=======
	__HOST__ array() : base_type( shared_ptr<T>( device_allocator<T>().allocate(N) ) )
	{
>>>>>>> ecuda2/master
		fill( value_type() );
	}

	///
<<<<<<< HEAD
	/// \brief Constructs an array with a shallow copy of each of the elements in src.
	///
	/// Be careful to note that a shallow copy means that only the pointer to the device memory
	/// that holds the elements is copied in the newly constructed container.  This allows
	/// containers to be passed-by-value to kernel functions with minimal overhead.  If a copy
	/// of the container is required in host code, use the << or >> operators. For example:
	///
	/// \code{.cpp}
	/// ecuda::array<int,10> arr;
	/// arr.fill( 3 ); // fill array with 3s
	/// ecuda::array<int,10> newArr( arr ); // shallow copy
	/// ecuda::array<int,10> newArr;
	/// newArr << arr; // deep copy
	/// arr >> newArr; // deep copy
	/// \endcode
	///
	/// \param src Another array object of the same type and size, whose contents are copied.
	///
	HOST DEVICE array( const array& src ) : deviceMemory(src.deviceMemory) {}

	/*
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
	HOST array( const array<T,N2>& src ) {
		deviceMemory = device_ptr<value_type>( device_allocator<value_type>().allocate(N) );
		CUDA_CALL( cudaMemcpy<value_type>( deviceMemory.get(), src.data(), std::min(N,N2), cudaMemcpyDeviceToDevice ) );
	}
	*/

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	///
=======
	/// \brief Copy constructor.
	///
	/// Constructs an array with a copy of the contents of src.
	///
	/// \param src another array object of the same type and size, whose contents are copied.
	///
	__HOST__ array( const array& src ) : base_type( shared_ptr<T>( device_allocator<T>().allocate(src.size()) ) )
	{
		ecuda::copy( src.begin(), src.end(), begin() );
	}

	///
	/// \brief Assignment operator.
	///
	/// Assigns new contents to the array, replacing its current contents.
	///
	/// \param other another array object of the same type and size, whose contents are assigned.
	///
	__HOST__ array& operator=( const array& other )
	{
		ecuda::copy( other.begin(), other.end(), begin() );
		return *this;
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Move constructor.
	///
	/// Constructs the container with the contents of the other using move semantics.
>>>>>>> ecuda2/master
	/// This constructor is only available if the compiler is configured to allow C++11.
	///
	/// \param src another container to be used as source to initialize the elements of the container with
	///
<<<<<<< HEAD
	HOST array( array<T,N>&& src ) : deviceMemory(std::move(src.deviceMemory)) {}
	#endif

	/*
	///
	/// \brief Destructs the array object.
	///
	//HOST DEVICE virtual ~array() {}
	*/
=======
	__HOST__ array( array&& src ) : base_type(std::move(src)) {}

	///
	/// \brief Move assignment operator.
	///
	/// Replaces the contents with those of src using move semantics (i.e. the data in src is moved from
	/// src into this container.
	///
	__HOST__ array& operator=( array&& src )
	{
		base_type::operator=(std::move(src));
		return *this;
	}
	#endif

	///
	/// \brief Returns a reference to the element at specified location index, with bounds checking.
	///
	/// If index is not within the range of the container, the current kernel will exit and
	/// cudaGetLastError will return cudaErrorUnknown.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline reference at( size_type index )
	{
		if( !(index < size()) ) {
			#ifndef __CUDACC__
			throw std::out_of_range( EXCEPTION_MSG("ecuda::array::at() index parameter is out of range") );
			#else
			// this strategy is taken from:
			// http://stackoverflow.com/questions/12521721/crashing-a-kernel-gracefully
			__threadfence();
			asm("trap;");
			#endif
		}
		return base_type::operator[](index);
	}

	///
	/// \brief Returns a reference to the element at specified location index, with bounds checking.
	///
	/// If index is not within the range of the container, the current kernel will exit and
	/// cudaGetLastError will return cudaErrorUnknown.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline const_reference at( size_type index ) const
	{
		if( !(index < size()) ) {
			#ifndef __CUDACC__
			throw std::out_of_range( EXCEPTION_MSG("ecuda::array::at() index parameter is out of range") );
			#else
			// this strategy is taken from:
			// http://stackoverflow.com/questions/12521721/crashing-a-kernel-gracefully
			__threadfence();
			asm("trap;");
			#endif
		}
		return base_type::operator[](index);
	}
>>>>>>> ecuda2/master

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
<<<<<<< HEAD
	DEVICE inline reference operator[]( size_type index ) { return *(deviceMemory.get()+index); }
=======
	__DEVICE__ inline reference operator[]( size_type index ) { return base_type::operator[](index); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
<<<<<<< HEAD
	DEVICE inline const_reference operator[]( size_type index ) const { return *(deviceMemory.get()+index); }

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
=======
	__DEVICE__ inline const_reference operator[]( size_type index ) const { return base_type::operator[](index); }

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// This is identical to operator[] but is present for consistency with higher-dimensional containers.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline reference operator()( const size_type index ) { return base_type::operator[](index); }

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// This is identical to operator[] but is present for consistency with higher-dimensional containers.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline const_reference operator()( const size_type index ) const { return base_type::operator[](index); }

>>>>>>> ecuda2/master
	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
<<<<<<< HEAD
	DEVICE inline const_reference front() const { return *deviceMemory; }
=======
	__DEVICE__ inline reference front() { return base_type::operator[](0); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
<<<<<<< HEAD
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
	HOST __CONSTEXPR__ inline size_type max_size() const { return std::numeric_limits<size_type>::max(); }
=======
	__DEVICE__ inline reference back() { return base_type::operator[](size()-1); }
	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
	__DEVICE__ inline const_reference front() const { return base_type::operator[](0); }

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
	__DEVICE__ inline const_reference back() const { return base_type::operator[](size()-1); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(); data()+size()) is always a valid range, even
	/// if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
<<<<<<< HEAD
	HOST DEVICE inline pointer data() { return deviceMemory.get(); }
=======
	__HOST__ __DEVICE__ inline pointer data() __NOEXCEPT__ { return base_type::get_pointer(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(); data()+size()) is always a valid range, even
	/// if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
<<<<<<< HEAD
	HOST DEVICE inline const_pointer data() const { return deviceMemory.get(); }
=======
	__HOST__ __DEVICE__ inline const_pointer data() const __NOEXCEPT__ { return base_type::get_pointer(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
<<<<<<< HEAD
	HOST DEVICE inline iterator begin() { return iterator(deviceMemory.get()); }
=======
	__HOST__ __DEVICE__ inline iterator begin() __NOEXCEPT__ { return base_type::begin(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
<<<<<<< HEAD
	HOST DEVICE inline iterator end() { return iterator(deviceMemory.get()+size()); }

	///
	/// \brief Returns an iterator to the first element of the container.
=======
	__HOST__ __DEVICE__ inline iterator end() __NOEXCEPT__ { return base_type::end(); }

	///
	/// \brief Returns a const_iterator to the first element of the container.
>>>>>>> ecuda2/master
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
<<<<<<< HEAD
	HOST DEVICE inline const_iterator begin() const { return const_iterator(deviceMemory.get()); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
=======
	///
	__HOST__ __DEVICE__ inline const_iterator begin() const __NOEXCEPT__ { return base_type::begin(); }

	///
	/// \brief Returns a const_iterator to the element following the last element of the container.
>>>>>>> ecuda2/master
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
<<<<<<< HEAD
	HOST DEVICE inline const_iterator end() const { return const_iterator(deviceMemory.get()+size()); }
=======
	__HOST__ __DEVICE__ inline const_iterator end() const __NOEXCEPT__ { return base_type::end(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
<<<<<<< HEAD
	HOST DEVICE inline reverse_iterator rbegin() { return reverse_iterator(end()); }
=======
	__HOST__ __DEVICE__ inline reverse_iterator rbegin() __NOEXCEPT__ { return base_type::rbegin(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
<<<<<<< HEAD
	HOST DEVICE inline reverse_iterator rend() { return reverse_iterator(begin()); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
=======
	__HOST__ __DEVICE__ inline reverse_iterator rend() __NOEXCEPT__ { return base_type::rend(); }

	///
	/// \brief Returns a const_reverse_iterator to the first element of the reversed container.
>>>>>>> ecuda2/master
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
<<<<<<< HEAD
	HOST DEVICE inline const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
=======
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return base_type::rbegin(); }

	///
	/// \brief Returns a const_reverse_iterator to the element following the last element of the reversed container.
>>>>>>> ecuda2/master
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
<<<<<<< HEAD
	HOST DEVICE inline const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

	#ifdef __CPP11_SUPPORTED__
	HOST DEVICE inline const_iterator cbegin() const __NOEXCEPT__ { return const_iterator(deviceMemory.get()); }
	HOST DEVICE inline const_iterator cend() const __NOEXCEPT__ { return const_iterator(deviceMemory.get()+size()); }
	HOST DEVICE inline const_reverse_iterator crbegin() __NOEXCEPT__ { return const_reverse_iterator(cend()); }
	HOST DEVICE inline const_reverse_iterator crend() __NOEXCEPT__ { return const_reverse_iterator(cbegin()); }
	#endif

	///
=======
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const __NOEXCEPT__ { return base_type::rend(); }

	#ifdef __CPP11_SUPPORTED__

	///
	/// \brief Returns a const_iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	__HOST__ __DEVICE__ inline const_iterator         cbegin()  const __NOEXCEPT__ { return base_type::cbegin();  }

	///
	/// \brief Returns a const_iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline const_iterator         cend()    const __NOEXCEPT__ { return base_type::cend();    }

	///
	/// \brief Returns a const_reverse_iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin() const __NOEXCEPT__ { return base_type::crbegin(); }

	///
	/// \brief Returns a const_reverse_iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline const_reverse_iterator crend() const   __NOEXCEPT__ { return base_type::crend();   }

	#endif

	///
	/// \brief Checks if the container has no elements.
	///
	/// \returns true if the container is empty, false otherwise.
	///
	__HOST__ __DEVICE__ __CONSTEXPR__ inline bool empty() const __NOEXCEPT__ { return size() == 0; }

	///
	/// \brief Returns the number of elements in the container.
	///
	/// \returns The number of elements in the container.
	///
	__HOST__ __DEVICE__ __CONSTEXPR__ inline size_type size() const __NOEXCEPT__ { return base_type::size(); }

	///
	/// \brief Returns the maximum number of elements the container is able to hold.
	///
	/// The value can be defined according to system or library implementation limitations.
	///
	/// \returns Maximum number of elements.
	///
	__HOST__ __CONSTEXPR__ inline size_type max_size() const __NOEXCEPT__ { return std::numeric_limits<size_type>::max(); }


	///
>>>>>>> ecuda2/master
	/// \brief Assigns a given value to all elements in the container.
	///
	/// \param value the value to assign to the elements
	///
<<<<<<< HEAD
	HOST DEVICE void fill( const value_type& value ) {
		#ifdef __CUDA_ARCH__
		for( iterator iter = begin(); iter != end(); ++iter ) *iter = value;
		#else
		std::vector< value_type, host_allocator<value_type> > v( size(), value );
		CUDA_CALL( cudaMemcpy<value_type>( deviceMemory.get(), &v.front(), size(), cudaMemcpyHostToDevice ) );
		#endif
	}
=======
	__HOST__ __DEVICE__ inline void fill( const value_type& value ) { ecuda::fill( begin(), end(), value ); }
>>>>>>> ecuda2/master

	///
	/// \brief Exchanges the contents of the container with those of the other.
	///
	/// Does not cause iterators and references to associate with the other container.
	///
	/// \param other container to exchange the contents with
	///
<<<<<<< HEAD
	HOST DEVICE void swap( array& other ) {
		#ifdef __CUDA_ARCH__
		iterator iter1 = begin();
		iterator iter2 = other.begin();
		for( ; iter1 != end(); ++iter1, ++iter2 ) ecuda::swap( *iter1, *iter2 );
		#else
		std::swap( deviceMemory, other.deviceMemory );
		//std::vector< value_type, host_allocator<value_type> > host1; operator>>( host1 );
		//std::vector< value_type, host_allocator<value_type> > host2; other.operator>>( host2 );
		//operator<<( host2 );
		//other.operator<<( host1 );
		#endif
=======
	__HOST__ __DEVICE__ inline void swap( array& other )
	#ifdef __CPP11_SUPPORTED__
	noexcept(noexcept(swap(std::declval<T&>(),std::declval<T&>())))
	#endif
	{
		base_type::swap( other );
>>>>>>> ecuda2/master
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
<<<<<<< HEAD
	HOST DEVICE bool operator==( const array<T,N>& other ) const {
		#ifdef __CUDA_ARCH__
		const_iterator iter1 = begin();
		const_iterator iter2 = other.begin();
		for( ; iter1 != end(); ++iter1, ++iter2 ) if( !( *iter1 == *iter2 ) ) return false;
		return true;
		#else
		//#ifdef __CPP11_SUPPORTED__
		//std::array<T,N> arr1, arr2;
		//#else
		std::vector< value_type, host_allocator<value_type> > arr1( size() ), arr2( size() );
		//#endif
		operator>>( arr1 );
		other.operator>>( arr2 );
		return arr1 == arr2;
		#endif
	}
=======
	__HOST__ __DEVICE__ inline bool operator==( const array& other ) const { return ecuda::equal( begin(), end(), other.begin() ); }
>>>>>>> ecuda2/master

	///
	/// \brief Checks if the contents of two arrays are not equal.
	///
	/// That is, whether any element in the this array does not compare equal with the element in
	/// another array at the same position.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents are not equal, false otherwise
	///
<<<<<<< HEAD
	HOST DEVICE inline bool operator!=( const array<T,N>& other ) const { return !operator==(other); }
=======
	__HOST__ __DEVICE__ inline bool operator!=( const array& other ) const { return !operator==(other); }
>>>>>>> ecuda2/master

	///
	/// \brief Compares the contents of two arrays lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this array are lexicographically less than the other array, false otherwise
	///
<<<<<<< HEAD
	HOST DEVICE inline bool operator<( const array<T,N>& other ) const {
		#ifdef __CUDA_ARCH__
		return ecuda::lexicographical_compare( begin(), end(), other.begin(), other.end() );
		#else
		//#ifdef __CPP11_SUPPORTED__
		//std::array<T,N> arr1, arr2;
		//#else
		std::vector< value_type, host_allocator<value_type> > arr1( size() ), arr2( size() );
		//#endif
		operator>>( arr1 );
		other.operator>>( arr2 );
		return arr1 < arr2;
		#endif
	}
=======
	__HOST__ __DEVICE__ inline bool operator<( const array& other ) const { return ecuda::lexicographical_compare( begin(), end(), other.begin(), other.end() ); }
>>>>>>> ecuda2/master

	///
	/// \brief Compares the contents of two arrays lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this array are lexicographically greater than the other array, false otherwise
	///
<<<<<<< HEAD
	HOST DEVICE inline bool operator>( const array<T,N>& other ) const {
		#ifdef __CUDA_ARCH__
		return ecuda::lexicographical_compare( other.begin(), other.end(), begin(), end() );
		#else
		//#ifdef __CPP11_SUPPORTED__
		//std::array<T,N> arr1, arr2;
		//#else
		std::vector< value_type, host_allocator<value_type> > arr1( size() ), arr2( size() );
		//#endif
		operator>>( arr1 );
		other.operator>>( arr2 );
		return arr1 > arr2;
		#endif
	}
=======
	__HOST__ __DEVICE__ inline bool operator>( const array& other ) const { return ecuda::lexicographical_compare( other.begin(), other.end(), begin(), end() ); }
>>>>>>> ecuda2/master

	///
	/// \brief Compares the contents of two arrays lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this array are lexicographically less than or equal to the other array, false otherwise
	///
<<<<<<< HEAD
	HOST DEVICE inline bool operator<=( const array<T,N>& other ) const { return !operator>(other); }
=======
	__HOST__ __DEVICE__ inline bool operator<=( const array& other ) const { return !operator>(other); }
>>>>>>> ecuda2/master

	///
	/// \brief Compares the contents of two arrays lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this array are lexicographically greater than or equal to the other array, false otherwise
	///
<<<<<<< HEAD
	HOST DEVICE inline bool operator>=( const array<T,N>& other ) const { return !operator<(other); }

	///
	/// \brief Copies the contents of this device array to another container.
	///
	/// \param dest container to copy contents to
	///
	template<class Container>
	HOST Container& operator>>( Container& dest ) const {
		::ecuda::copy( begin(), end(), dest.begin() );
		return dest;
	}

	///
	/// \brief Copies the contents of a container to this device array.
	///
	/// \param src container to copy the contents from
	/// \exception std::length_error thrown if this array is not large enough to hold the given vector's contents
	///
	template<class Container>
	HOST array& operator<<( const Container& src ) {
		if( ecuda::distance(src.begin(),src.end()) > static_cast<typename Container::difference_type>(size()) )
			throw std::length_error( EXCEPTION_MSG("ecuda::array is not large enough to fit contents of provided container") );
		::ecuda::copy( src.begin(), src.end(), begin() );
		return *this;
	}

	/*
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
	HOST DEVICE array& operator=( const array& other ) {
		#ifdef __CUDA_ARCH__
		// shallow copy if called from device
		deviceMemory = other.deviceMemory;
		#else
		// deep copy if called from host
		deviceMemory = device_ptr<value_type>( device_allocator<value_type>().allocate(N) );
		CUDA_CALL( cudaMemcpy<value_type>( deviceMemory.get(), other.deviceMemory.get(), N, cudaMemcpyDeviceToDevice ) );
		#endif
		return *this;
	}
	*/

};

=======
	__HOST__ __DEVICE__ inline bool operator>=( const array& other ) const { return !operator<(other); }

};

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

///
/// An array subclass that should be used as the representation of an array within kernel code.
///
/// This achieves two objectives: 1) create a new array object that is instantiated by creating
/// a shallow copy of the contents (so that older versions of the CUDA API that don't support
/// kernel pass-by-reference can specify containers in the function arguments), and 2) strip any
/// unnecessary data that will be useless to the kernel thus reducing register usage (in this
/// case by removing the unneeded reference-counting introduced by the internal shared_ptr).
///
template<typename T,std::size_t N>
class array_kernel_argument : public array<T,N,typename ecuda::add_pointer<T>::type>
{

private:
	typedef array<T,N,typename ecuda::add_pointer<T>::type> base_type;

public:
	template<class P>
	__HOST__ array_kernel_argument( const array<T,N,P>& src ) : base_type( src, ecuda::true_type() ) {}

	__HOST__ __DEVICE__ array_kernel_argument( const array_kernel_argument& src ) : base_type( src, ecuda::true_type() ) {}

	template<class P>
	__HOST__ array_kernel_argument& operator=( const array<T,N,P>& src )
	{
		base_type::shallow_assign(src);
		return *this;
	}

	#ifdef __CPP11_SUPPORTED__
	array_kernel_argument( array_kernel_argument&& src ) : base_type(std::move(src)) {}

	array_kernel_argument& operator=( array_kernel_argument&& src )
	{
		base_type::operator=(std::move(src));
		return *this;
	}
	#endif

};

} // namespace impl
/// \endcond

>>>>>>> ecuda2/master
} // namespace ecuda

#endif
