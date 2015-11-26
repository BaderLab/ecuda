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
//
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

#include "global.hpp"
#include "algorithm.hpp"
#include "allocators.hpp"
#include "memory.hpp"
#include "impl/models.hpp"
#include "type_traits.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<typename T,class Alloc> class vector_kernel_argument; // forward declaration

} // namespace impl
/// \endcond

///
/// \brief A resizable vector stored in device memory.
///
/// Methods are prefaced with appropriate keywords to declare them as host and/or device capable.
/// In general: operations requiring memory allocation/deallocation are host only, operations
/// to access the values of specific elements are device only, and copy operations on ranges of data and
/// accessors of general information can be performed on both the host and device.
///
/// Any growth of the vector follows a doubling pattern.  The existing memory allocation size
/// is doubled until the requested amount of memory is met or exceeded.
///
template< typename T, class Alloc=device_allocator<T> >
class vector : private impl::device_contiguous_sequence< T, shared_ptr<T> > {

private:
	typedef impl::device_contiguous_sequence< T, shared_ptr<T> > base_type;

public:
	typedef typename base_type::value_type      value_type;      //!< cell data type
	typedef Alloc                               allocator_type;  //!< allocator type
	typedef typename base_type::size_type       size_type;       //!< unsigned integral type
	typedef typename base_type::difference_type difference_type; //!< signed integral type
	#ifdef __CPP11_SUPPORTED__
	typedef typename base_type::reference                        reference;       //!< cell reference type
	typedef typename base_type::const_reference                  const_reference; //!< cell const reference type
	typedef typename std::allocator_traits<Alloc>::pointer       pointer;         //!< cell pointer type
	typedef typename std::allocator_traits<Alloc>::const_pointer const_pointer;   //!< cell const pointer type
	#else
	typedef typename Alloc::reference       reference;       //!< cell reference type
	typedef typename Alloc::const_reference const_reference; //!< cell const reference type
	typedef typename Alloc::pointer         pointer;         //!< cell pointer type
	typedef typename Alloc::const_pointer   const_pointer;   //!< cell const pointer type
	#endif

	typedef typename base_type::iterator               iterator;               //!< iterator type
	typedef typename base_type::const_iterator         const_iterator;         //!< const iterator type
	typedef typename base_type::reverse_iterator       reverse_iterator;       //!< reverse iterator type
	typedef typename base_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

	typedef impl::vector_kernel_argument<T,Alloc> kernel_argument; //!< kernel argument type

private:
	size_type n; //!< number of elements currently stored
	allocator_type allocator;

protected:
	__HOST__ __DEVICE__ vector( const vector& src, ecuda::true_type ) : base_type(src), n(src.n), allocator(src.allocator) {}

	__HOST__ __DEVICE__ vector& shallow_assign( const vector& other )
	{
		base_type::get_pointer() = other.get_pointer();
		n = other.n;
		allocator = other.allocator;
		return *this;
	}


private:
	__HOST__ void growMemory( size_type minimum );

	__HOST__ void init( size_type len, const value_type& value, ecuda::true_type )
	{
		growMemory( len );
		n = len;
		if( len ) ecuda::fill( begin(), end(), value );
	}

	template<class Iterator>
	__HOST__ inline void init( Iterator first, Iterator last, ecuda::false_type )
	{
		const size_type len = ::ecuda::distance(first,last);
		growMemory( len );
		ecuda::copy( first, last, begin() );
		n = len;
	}

public:
	///
	/// \brief Default constructor. Constructs empty container.
	/// \param allocator allocator to use for all memory allocations of this container
	///
	__HOST__ explicit vector( const allocator_type& allocator = allocator_type() ) : base_type(), n(0), allocator(allocator) {}

	///
	/// \brief Constructs the container with n copies of elements with value value.
	/// \param n the size of the container
	/// \param value the value to initialize elements of the container with
	/// \param allocator allocator to use for all memory allocations of this container
	///
	__HOST__ explicit vector( size_type n, const value_type& value, const allocator_type& allocator = allocator_type() ) : base_type( shared_ptr<T>( allocator.allocate(n) ), n ), n(n), allocator(allocator)
	{
		init( n, value, ecuda::true_type() );
	}

	///
	/// \brief Constructs the container with n default-inserted instances of T. No copies are made.
	/// \param n the size of the container
	///
	__HOST__ explicit vector( size_type n ) : base_type( shared_ptr<T>( Alloc().allocate(n) ) ), n(n)
	{
		init( n, value_type(), ecuda::true_type() );
	}

	///
	/// \brief Constructs the container with the contents of the range [begin,end).
	/// \param first,last the range to copy the elements from
	/// \param allocator allocator to use for all memory allocations of this container
	///
	template<class Iterator>
	__HOST__ vector( Iterator first, Iterator last, const allocator_type& allocator = allocator_type() ) : base_type(), n(0), allocator(allocator)
	{
		typedef typename ecuda::is_integral<Iterator>::type _Integral;
		init( first, last, _Integral() );
	}

	///
	/// \brief Copy constructor.
	///
	/// Constructs a vector with a copy of the contents of src.
	///
	/// \param src Another vector object of the same type and size, whose contents are copied.
	///
	__HOST__ vector( const vector& src ) :
		base_type(),
		n(src.n),
		std::allocator_traits<allocator_type>::select_on_container_copy_construction(src.get_allocator())
	{
		if( size() != src.size() ) resize( src.size() );
		ecuda::copy( src.begin(), src.end(), begin() );
	}

	__HOST__ vector& operator=( const vector& src )
	{
		if( size() != src.size() ) resize( src.size() );
		allocator = src.allocator;
		ecuda::copy( src.begin(), src.end(), begin() );
		return *this;
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	///
	/// This constructor is only available if the compiler is configured to allow C++11.
	///
	/// \param src another container to be used as source to initialize the elements of the container with
	///
	__HOST__ __DEVICE__ vector( vector&& src ) : base_type(src), n(std::move(src.n)), allocator(std::move(src.allocator)) {}

	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	/// \param src another container to be used as source to initialize the elements of the container with
	/// \param allocator allocator to use for all memory allocations of this container
	///
	__HOST__ __DEVICE__ vector( vector&& src, const allocator_type& allocator ) : base_type(src), n(std::move(src.n)), allocator(allocator) {}

	///
	/// \brief Constructs the container with the contents of the initializer list il.
	/// \param il initializer list to initialize the elements of the container with
	/// \param allocator allocator to use for all memory allocations of this container
	///
	__HOST__ vector( std::initializer_list<value_type> il, const allocator_type& allocator = allocator_type() ) : base_type(shared_ptr<T>(allocator.allocate(il.size()))), n(il.size()), allocator(allocator)
	{
		ecuda::copy( il.begin(), il.end(), begin() );
	}

	__HOST__ vector& operator=( vector&& src )
	{
		base_type::operator=(std::move(src));
		n = std::move(src.n);
		return *this;
	}
	#endif

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	__HOST__ __DEVICE__ inline iterator begin() __NOEXCEPT__ { return base_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline iterator end() __NOEXCEPT__ { return base_type::begin()+size(); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	__HOST__ __DEVICE__ inline const_iterator begin() const __NOEXCEPT__ { return base_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline const_iterator end() const __NOEXCEPT__ { return base_type::begin()+size(); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	__HOST__ __DEVICE__ inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(end()); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(begin()); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(end()); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(begin()); }

	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ inline const_iterator         cbegin() const __NOEXCEPT__ { return base_type::cbegin();         }
	__HOST__ __DEVICE__ inline const_iterator         cend() const   __NOEXCEPT__ { return base_type::cbegin()+size();  }
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin()      __NOEXCEPT__ { return base_type::crbegin();        }
	__HOST__ __DEVICE__ inline const_reverse_iterator crend()        __NOEXCEPT__ { return base_type::crbegin()+size(); }
	#endif

	///
	/// \brief Returns the number of elements in the container.
	///
	/// \returns The number of elements in the container.
	///
	__HOST__ __DEVICE__ inline size_type size() const __NOEXCEPT__ { return n; }

	///
	/// \brief Returns the maximum number of elements the container is able to hold due to system
	///        or library implementation limitations.
	///
	/// \returns Maximum number of elements.
	///
	__HOST__ __DEVICE__ __CONSTEXPR__ inline size_type max_size() const __NOEXCEPT__ { return std::numeric_limits<size_type>::max(); }

	///
	/// \brief Resizes the container to contain newSize elements.
	///
	/// If the current size is greater than newSize, the container is reduced to its first newSize
	/// elements as if by repeatedly calling pop_back().
	///
	/// \param newSize new size of the container
	/// \param value the value to initialize the new elements with
	///
	__HOST__ void resize( size_type newSize, const value_type& value = value_type() )
	{
		if( size() == newSize ) return;
		if( size() > newSize ) { n = newSize; return; }
		growMemory( newSize ); // make sure enough device memory is allocated
		std::vector< value_type, host_allocator<value_type> > v( newSize-n, value );
		ecuda::copy( v.begin(), v.end(), begin() );
		n = newSize;
	}

	///
	/// \brief Returns the number of elements that the container has currently allocated space for.
	/// \return Capacity of the currently allocated storage.
	///
	__HOST__ __DEVICE__ inline size_type capacity() const __NOEXCEPT__ { return base_type::size(); }

	///
	/// \brief Checks if the container has no elements.
	///
	/// \returns true if the container is empty, false otherwise.
	///
	__HOST__ __DEVICE__ inline bool empty() const __NOEXCEPT__ { return !n; }

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
	__HOST__ inline void reserve( size_type newCapacity ) { growMemory(newCapacity); }

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
			#ifdef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
			throw std::out_of_range( EXCEPTION_MSG("ecuda::vector::at() index parameter is out of range") );
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
			#ifdef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
			throw std::out_of_range( EXCEPTION_MSG("ecuda::vector::at() index parameter is out of range") );
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
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline reference operator[]( const size_type index ) { return base_type::operator[](index); }

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// \param index position of the element to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline const_reference operator[]( const size_type index ) const { return base_type::operator[](index); }

	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
	__DEVICE__ inline reference front() { return operator[](0); }

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
	__DEVICE__ inline reference back() { return operator[]( size()-1 ); }

	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
	__DEVICE__ inline const_reference front() const { return operator[](0); }

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
	__DEVICE__ inline const_reference back() const { return operator[]( size()-1 ); }

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(),data()+size()) is always a valid
	/// range, even if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
	__HOST__ __DEVICE__ inline pointer data() __NOEXCEPT__ { return base_type::get_pointer(); }

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(),data()+size()) is always a valid
	/// range, even if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
	__HOST__ __DEVICE__ inline const_pointer data() const __NOEXCEPT__ { return base_type::get_pointer(); }

	///
	/// \brief Replaces the contents of the container.
	/// \param newSize the new size of the container
	/// \param value the value to initialize elements of the container with
	///
	__HOST__ void assign( size_type newSize, const value_type& value = value_type() )
	{
		growMemory(newSize); // make sure enough device memory is allocated
		ecuda::fill( begin(), end(), value );
		n = newSize;
	}

	///
	/// \brief Replaces the contents of the container with copies of those in the range [first,last).
	/// \param first,last the range to copy the elements from
	///
	template<class Iterator>
	__HOST__ void assign( Iterator first, Iterator last )
	{
		typename std::iterator_traits<Iterator>::difference_type len = ::ecuda::distance(first,last);
		growMemory( len ); // make sure enough device memory is allocated
		ecuda::copy( first, last, begin() );
		n = len;
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Replaces the contents with the elements from the initializer list il.
	///
	/// This method is only available if the compiler is configured to allow C++11.
	///
	/// \param il initializer list to copy the values from
	///
	__HOST__ inline void assign( std::initializer_list<value_type> il )
	{
		assign( il.begin(), il.end() );
		//host_array_proxy<const value_type> proxy( il.begin(), il.size() );
		//assign( proxy.begin(), proxy.end() );
	}
	#endif

	///
	/// \brief Appends the given element value to the end of the container.
	/// \param value the value of the element to append
	///
	__HOST__ void push_back( const value_type& value )
	{
		growMemory(n+1);
		ecuda::copy( &value, (&value)+1, begin()+n );
		++n;
	}

	///
	/// \brief Removes the last element of the container.
	///
	/// Calling pop_back on an empty container is undefined. No iterators or references
	/// expect for back() and end() are invalidated.
	///
	/// Although this can be called from both the host and device, a call from the
	/// device removes the last element of the container in the calling thread only.
	///
	__HOST__ __DEVICE__ inline void pop_back() { /*if( n )*/ --n; } // NOTE: if called from device the host instance doesn't change

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
	__HOST__ iterator insert( const_iterator position, const value_type& value )
	{
		std::vector< value_type, host_allocator<value_type> > v( ::ecuda::distance(position,end())+1 ); // allocate staging memory
		v.front() = value; // put new element at front of staging
		::ecuda::copy( position, end(), v.begin()+1 ); // copy trailing elements to staging
		const size_type index = ::ecuda::distance(begin(),position); // get index of insert position
		growMemory(size()+1); // make sure enough device memory is allocated
		// reacquire iterator just in case new memory was allocated
		const_iterator newPosition = begin()+index;
		::ecuda::copy( v.begin(), v.end(), newPosition );
		++n;
		return newPosition;
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
	__HOST__ iterator insert( const_iterator position, const size_type count, const value_type& value )
	{
		std::vector< value_type, host_allocator<value_type> > v( ::ecuda::distance(position,end())+count, value ); // allocate staging memory
		::ecuda::copy( position, end(), v.begin()+count ); // copy trailing elements to staging
		const size_type index = ::ecuda::distance(begin(),position); // get index of insert position
		growMemory(size()+count); // make sure enough device memory is allocated
		// require iterator just in case new memory was allocated
		const_iterator newPosition = begin()+index;
		::ecuda::copy( v.begin(), v.end(), newPosition );
		n += count;
		return newPosition;
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
	__HOST__ iterator insert( const_iterator position, value_type&& value );
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
	__HOST__ void insert( const_iterator position, InputIterator first, InputIterator last )
	{
		const std::vector< value_type, host_allocator<value_type> > v( first, last ); // allocate staging memory and put new content there
		const size_type len = v.size(); // number of new elements
		v.resize( v.size()+::ecuda::distance(position,end()) ); // make room for trailing elements
		::ecuda::copy( position, end(), v.begin()+len ); // copy trailing elements to staging
		const size_type index = ::ecuda::distance(begin(),position); // get index of insert position
		growMemory(size()+len); // make sure enough device memory is allocated
		// require iterator just in case new memory was allocated
		const_iterator newPosition = begin()+index;
		::ecuda::copy( v.begin(), v.end(), newPosition );
		n += len;
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
	__HOST__ inline void insert( const_iterator position, std::initializer_list<value_type> il )
	{
		return insert( position, il.begin(), il.end() );
	}
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
	__HOST__ iterator erase( const_iterator position )
	{
		vector<value_type> v( position+1, end() ); // copy trailing elements to another device vector
		ecuda::copy( v.begin(), v.end(), position ); // overwrite erase position
		--n;
		return begin()+ecuda::distance(begin(),position);
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
	__HOST__ iterator erase( const_iterator first, const_iterator last )
	{
		vector<value_type> v( last, end() ); // copy trailing elements to another device vector
		ecuda::copy( v.begin(), v.end(), first ); // overwrite erased elements
		n -= ecuda::distance(first,last);
		return begin()+ecuda::distance(begin(),first);
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
	__HOST__ __DEVICE__ void swap( vector& other )
	{
		base_type::swap( other );
		ecuda::swap( n, other.n );
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
	__HOST__ __DEVICE__ inline void clear() { n = 0; }

	///
	/// \brief Returns the allocator associated with the container.
	/// \returns The associated allocator.
	///
	__HOST__ inline allocator_type get_allocator() const { return allocator; }

	///
	/// \brief Requests the removal of unused capacity.
	///
	/// The STL specification describes this as a non-binding request to reduce capacity()
	/// to size() and it depends on the implementation if the request is fulfilled. In this
	/// implementation, the request is always fulfilled. All iterators, including the past
	/// the end iterator, are potentially invalidated.
	///
	__HOST__ void shrink_to_fit()
	{
		if( size() == capacity() ) return;
		vector v( n );
		ecuda::copy( begin(), end(), v.begin() );
		swap( v );
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
	__HOST__ __DEVICE__ bool operator==( const vector& other ) const
	{
		if( size() != other.size() ) return false;
		#ifdef __CUDA_ARCH__
		const_iterator iter1 = begin();
		const_iterator iter2 = other.begin();
		for( ; iter1 != end(); ++iter1, ++iter2 ) if( !( *iter1 == *iter2 ) ) return false;
		return true;
		#else
		return ecuda::equal( begin(), end(), other.begin(), other.end() );
		#endif
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
	__HOST__ __DEVICE__ inline bool operator!=( const vector& other ) const { return !operator==(other); }

	///
	/// \brief Compares the contents of two vectors lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this vector are lexicographically less than the other vector, false otherwise
	///
	__HOST__ __DEVICE__ inline bool operator<( const vector& other ) const
	{
		#ifdef __CUDA_ARCH__
		return ecuda::lexicographical_compare( begin(), end(), other.begin(), other.end() );
		#else
		std::vector< value_type, host_allocator<value_type> > v1( size() ), v2( size() );
		ecuda::copy( begin(), end(), v1.begin() );
		ecuda::copy( other.begin(), other.end(), v2.begin() );
		return std::lexicographical_compare( v1.begin(), v1.end(), v2.begin(), v2.end() );
		#endif
	}

	///
	/// \brief Compares the contents of two vectors lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this vector are lexicographically greater than the other vector, false otherwise
	///
	__HOST__ __DEVICE__ inline bool operator>( const vector& other ) const
	{
		#ifdef __CUDA_ARCH__
		return ecuda::lexicographical_compare( other.begin(), other.end(), begin(), end() );
		#else
		std::vector< value_type, host_allocator<value_type> > v1( size() ), v2( size() );
		ecuda::copy( begin(), end(), v1.begin() );
		ecuda::copy( other.begin(), other.end(), v2.begin() );
		return std::lexicographical_compare( v2.begin(), v2.end(), v1.begin(), v1.end() );
		#endif
	}

	///
	/// \brief Compares the contents of two vectors lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this vector are lexicographically less than or equal to the other vector, false otherwise
	///
	__HOST__ __DEVICE__ inline bool operator<=( const vector& other ) const { return !operator>(other); }

	///
	/// \brief Compares the contents of two vectors lexicographically.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this vector are lexicographically greater than or equal to the other vector, false otherwise
	///
	__HOST__ __DEVICE__ inline bool operator>=( const vector& other ) const { return !operator<(other); }

};

template<typename T,class Alloc>
__HOST__ void vector<T,Alloc>::growMemory( size_type minimum ) {
	if( base_type::size() >= minimum ) return; // no growth neccessary
	size_type m2 = base_type::size();
	if( !m2 ) m2 = 1; // in case no memory is currently allocated
	while( m2 < minimum ) m2 <<= 1;
	// allocate larger chunk
	shared_ptr<value_type> newMemory = get_allocator().allocate( m2 );
	impl::device_contiguous_sequence< value_type, shared_ptr<value_type> > newSequence( newMemory, m2 );
	ecuda::copy( begin(), end(), newSequence.begin() );
	base_type::swap( newSequence );
}

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<typename T,class Alloc>
class vector_kernel_argument : public vector<T,Alloc> {

public:
	vector_kernel_argument( const vector<T,Alloc>& src ) : vector<T,Alloc>( src, ecuda::true_type() ) {}
	vector_kernel_argument& operator=( const vector<T,Alloc>& src ) {
		vector<T,Alloc>::shallow_assign( src );
		return *this;
	}

};

} // namespace impl
/// \endcond

} // namespace ecuda

#endif
