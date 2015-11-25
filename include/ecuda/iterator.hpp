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
// iterator.hpp
//
// STL-style iterators customized to work with device memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ITERATOR_HPP
#define ECUDA_ITERATOR_HPP

#include <iterator>

#include "global.hpp"
//#include "memory.hpp"
#include "type_traits.hpp"

namespace ecuda {

// NOTE: libc++ requires inheritance from one of the 5 STL iterator categories (libstdc++ does not)

template<typename T,typename P> class padded_ptr; // forward declaration

///
/// \brief Iterator category denoting device memory.
///
struct device_iterator_tag :                  ::std::bidirectional_iterator_tag {};

///
/// \brief Iterator category denoting contiguous device memory.
///
struct device_contiguous_iterator_tag :       ::std::random_access_iterator_tag {}; // libc++ requires inheritance from one of the 5 STL iterator categories

///
/// \brief Iterator category denoting device memory that is made of contiguous blocks (but the blocks themselves are non-contiguous).
///
/// This inherits from device_iterator_tag so any other ecuda foundation classes
/// that are not specialized for this memory model will consider it non-contiguous
/// memory.
///
struct device_contiguous_block_iterator_tag : device_iterator_tag {};

template<typename T,typename PointerType,typename Category=device_iterator_tag>
class device_iterator //: std::iterator<Category,T,std::ptrdiff_t,PointerType>
{
private:
	typedef std::iterator<Category,T,std::ptrdiff_t,PointerType> base_type;

public:
	typedef Category                                    iterator_category;
	typedef T                                           value_type;
	typedef std::ptrdiff_t                              difference_type;
	typedef PointerType                                 pointer;
	typedef typename std::add_lvalue_reference<T>::type reference;

	template<typename T2,typename PointerType2,typename Category2> friend class device_iterator;
	template<typename T2> friend class device_contiguous_iterator;

protected:
	pointer ptr;

public:
	__HOST__ __DEVICE__ device_iterator( const pointer& ptr = pointer() ) : ptr(ptr) {}
	__HOST__ __DEVICE__ device_iterator( const device_iterator& src ) : ptr(src.ptr) {}
	template<typename T2,typename PointerType2>	__HOST__ __DEVICE__ device_iterator( const device_iterator<T2,PointerType2,Category>& src ) : ptr(src.ptr) {}

	__HOST__ __DEVICE__ inline device_iterator& operator++() { ++ptr; return *this; }
	__HOST__ __DEVICE__ inline device_iterator operator++( int )
	{
		device_iterator tmp(*this);
		++(*this);
		return tmp;
	}

	__HOST__ __DEVICE__ inline device_iterator& operator--() { --ptr; return *this; }
	__HOST__ __DEVICE__ inline device_iterator operator--( int )
	{
		device_iterator tmp(*this);
		--(*this);
		return tmp;
	}

	__HOST__ __DEVICE__ inline bool operator==( const device_iterator& other ) const __NOEXCEPT__ { return ptr == other.ptr; }
	__HOST__ __DEVICE__ inline bool operator!=( const device_iterator& other ) const __NOEXCEPT__ { return !operator==(other); }

	__DEVICE__ inline reference operator*() { return *ptr; }
	__HOST__ __DEVICE__ inline pointer operator->() const { return ptr; }

	__HOST__ __DEVICE__ inline device_iterator& operator=( const device_iterator& other )
	{
		ptr = other.ptr;
		return *this;
	}

	template<typename U,typename PointerType2>
	__HOST__ __DEVICE__ inline device_iterator& operator=( const device_iterator<U,PointerType2,Category>& other )
	{
		ptr = other.ptr;
		return *this;
	}

};

template<typename T>
class device_contiguous_iterator : public device_iterator<T,T*,device_contiguous_iterator_tag>
{

private:
	typedef device_iterator<T,T*,device_contiguous_iterator_tag> base_type;

public:
	typedef typename base_type::iterator_category iterator_category;
	typedef typename base_type::value_type        value_type;
	typedef typename base_type::difference_type   difference_type;
	typedef typename base_type::pointer           pointer;
	typedef typename base_type::reference         reference;

public:
	__HOST__ __DEVICE__ device_contiguous_iterator( const pointer& ptr = pointer() ) : base_type(ptr) {}
	__HOST__ __DEVICE__ device_contiguous_iterator( const device_contiguous_iterator& src ) : base_type(src) {}
	template<typename U> __HOST__ __DEVICE__ device_contiguous_iterator( const device_contiguous_iterator<U>& src ) : base_type(src) {}

	__HOST__ __DEVICE__ inline device_contiguous_iterator operator+( int x ) const { return device_contiguous_iterator( base_type::ptr + x ); }
	__HOST__ __DEVICE__ inline device_contiguous_iterator operator-( int x ) const { return device_contiguous_iterator( base_type::ptr - x ); }

	__HOST__ __DEVICE__ inline device_contiguous_iterator& operator+=( int x ) { base_type::ptr += x; return *this; }
	__HOST__ __DEVICE__ inline device_contiguous_iterator& operator-=( int x ) { base_type::ptr -= x; return *this; }

	__DEVICE__ inline reference operator[]( int x ) const { return *(base_type::ptr+x); }

	__HOST__ __DEVICE__ inline difference_type operator-( const device_contiguous_iterator& other ) { return base_type::ptr - other.ptr; }

	__HOST__ __DEVICE__ inline bool operator<( const device_contiguous_iterator& other ) const __NOEXCEPT__ { return base_type::ptr < other.ptr; }
	__HOST__ __DEVICE__ inline bool operator>( const device_contiguous_iterator& other ) const __NOEXCEPT__ { return base_type::ptr > other.ptr; }
	__HOST__ __DEVICE__ inline bool operator<=( const device_contiguous_iterator& other ) const __NOEXCEPT__ { return operator<(other) or operator==(other); }
	__HOST__ __DEVICE__ inline bool operator>=( const device_contiguous_iterator& other ) const __NOEXCEPT__ { return operator>(other) or operator==(other); }

};

template<typename T,typename P>
class device_contiguous_block_iterator : public device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag>
{

private:
	typedef device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag> base_type;

public:
	typedef typename base_type::iterator_category iterator_category;
	typedef typename base_type::value_type        value_type;
	typedef typename base_type::difference_type   difference_type;
	typedef typename base_type::pointer           pointer;
	typedef typename base_type::reference         reference;

	//typedef P                                 contiguous_pointer;
	typedef device_contiguous_iterator<T>       contiguous_iterator;

public:
	__HOST__ __DEVICE__ device_contiguous_block_iterator( const pointer& ptr = pointer() ) : base_type(ptr) {}
	__HOST__ __DEVICE__ device_contiguous_block_iterator( const device_contiguous_block_iterator& src ) : base_type(src) {}
	template<typename U,typename Q>
	__HOST__ __DEVICE__ device_contiguous_block_iterator( const device_contiguous_block_iterator<U,Q>& src ) : base_type(src) {}

	__HOST__ __DEVICE__ inline device_contiguous_block_iterator operator+( int x ) const { return device_contiguous_block_iterator( base_type::ptr + x ); }
	__HOST__ __DEVICE__ inline device_contiguous_block_iterator operator-( int x ) const { return device_contiguous_block_iterator( base_type::ptr - x ); }

	__HOST__ __DEVICE__ inline device_contiguous_block_iterator& operator+=( int x ) { base_type::ptr += x; return *this; }
	__HOST__ __DEVICE__ inline device_contiguous_block_iterator& operator-=( int x ) { base_type::ptr -= x; return *this; }

	__DEVICE__ inline reference operator[]( int x ) const { return *(base_type::ptr+x); }

	__HOST__ __DEVICE__ inline difference_type operator-( const device_contiguous_block_iterator& other ) { return base_type::ptr - other.ptr; }

	__HOST__ __DEVICE__ inline bool operator< ( const device_contiguous_block_iterator& other ) const __NOEXCEPT__ { return base_type::ptr < other.ptr; }
	__HOST__ __DEVICE__ inline bool operator> ( const device_contiguous_block_iterator& other ) const __NOEXCEPT__ { return base_type::ptr > other.ptr; }
	__HOST__ __DEVICE__ inline bool operator<=( const device_contiguous_block_iterator& other ) const __NOEXCEPT__ { return operator<(other) or operator==(other); }
	__HOST__ __DEVICE__ inline bool operator>=( const device_contiguous_block_iterator& other ) const __NOEXCEPT__ { return operator>(other) or operator==(other); }

	//@EXPERIMENTAL START
	__HOST__ __DEVICE__ inline contiguous_iterator contiguous_begin() const __NOEXCEPT__ { return contiguous_iterator( naked_cast<typename std::add_pointer<T>::type>( base_type::ptr.get() ) ); }
	__HOST__ __DEVICE__ inline contiguous_iterator contiguous_end()   const __NOEXCEPT__ { return contiguous_iterator( naked_cast<typename std::add_pointer<T>::type>( base_type::ptr.get() ) + base_type::ptr.get_remaining_width() ); }
	//@EXPERIMENTAL END

};


template<class Iterator>
class reverse_device_iterator //: public std::iterator<device_iterator_tag,typename Iterator::value_type,typename Iterator::difference_type,typename Iterator::pointer>
{
private:
	typedef std::iterator<device_iterator_tag,typename Iterator::value_type,typename Iterator::difference_type,typename Iterator::pointer> base_type;

public:
	typedef Iterator                              iterator_type;
	typedef typename base_type::iterator_category iterator_category;
	typedef typename base_type::value_type        value_type;
	typedef typename base_type::difference_type   difference_type;
	typedef typename base_type::pointer           pointer;
	typedef typename base_type::reference         reference;

private:
	Iterator parentIterator;

public:
	__HOST__ __DEVICE__ reverse_device_iterator( Iterator parentIterator = Iterator() ) : parentIterator(parentIterator) {}
	__HOST__ __DEVICE__ reverse_device_iterator( const reverse_device_iterator& src ) : parentIterator(src.parentIterator) {}
	template<class Iterator2>
	__HOST__ __DEVICE__ reverse_device_iterator( const reverse_device_iterator<Iterator2>& src ) : parentIterator(src.base()) {}

	__HOST__ __DEVICE__ Iterator base() const { return parentIterator; }

	__HOST__ __DEVICE__ inline reverse_device_iterator& operator++() { --parentIterator; return *this; }
	__HOST__ __DEVICE__ inline reverse_device_iterator operator++( int ) {
		reverse_device_iterator tmp(*this);
		++(*this);
		return tmp;
	}

	__HOST__ __DEVICE__ inline reverse_device_iterator& operator--() { ++parentIterator; return *this; }
	__HOST__ __DEVICE__ inline reverse_device_iterator operator--( int ) {
		reverse_device_iterator tmp(*this);
		--(*this);
		return tmp;
	}

	__HOST__ __DEVICE__ inline bool operator==( const reverse_device_iterator& other ) const { return parentIterator == other.parentIterator; }
	__HOST__ __DEVICE__ inline bool operator!=( const reverse_device_iterator& other ) const { return !operator==(other); }

	__DEVICE__ inline reference operator*() const {
		Iterator tmp(parentIterator);
		--tmp;
		return tmp.operator*();
	}

	__HOST__ __DEVICE__ inline pointer operator->() const {
		Iterator tmp(parentIterator);
		--tmp;
		return tmp.operator->();
	}

	__HOST__ __DEVICE__ inline difference_type operator-( const reverse_device_iterator& other ) { return parentIterator - other.parentIterator; }

	__HOST__ __DEVICE__ inline reverse_device_iterator operator+( int x ) const { return reverse_device_iterator( parentIterator-x ); }
	__HOST__ __DEVICE__ inline reverse_device_iterator operator-( int x ) const { return reverse_device_iterator( parentIterator+x ); }

	__HOST__ __DEVICE__ inline bool operator<( const reverse_device_iterator& other ) const { return parentIterator < other.parentIterator; }
	__HOST__ __DEVICE__ inline bool operator>( const reverse_device_iterator& other ) const { return parentIterator > other.parentIterator; }
	__HOST__ __DEVICE__ inline bool operator<=( const reverse_device_iterator& other ) const { return operator<(other) or operator==(other); }
	__HOST__ __DEVICE__ inline bool operator>=( const reverse_device_iterator& other ) const { return operator>(other) or operator==(other); }

	__HOST__ __DEVICE__ inline reverse_device_iterator& operator+=( int x ) { parentIterator -= x; return *this; }
	__HOST__ __DEVICE__ inline reverse_device_iterator& operator-=( int x ) { parentIterator += x; return *this; }

	__DEVICE__ reference operator[]( int x ) const { return parentIterator.operator[]( -x-1 ); }

	__HOST__ __DEVICE__ reverse_device_iterator& operator=( const reverse_device_iterator& other ) {
		parentIterator = other.parentIterator;
		return *this;
	}

	template<class Iterator2>
	__HOST__ __DEVICE__ reverse_device_iterator& operator=( const reverse_device_iterator<Iterator2>& other ) {
		parentIterator = other.parentIterator;
		return *this;
	}

};

//template<class IteratorCategory> struct __is_contiguous { typedef std::false_type type; };
//template<> struct __is_contiguous<std::random_access_iterator_tag> { typedef std::true_type type; };
//template<typename T> struct __is_contiguous<T*> { typedef std::true_type type; };
//template<typename T> struct __is_contiguous<const T*> { typedef std::true_type type; };

template<class Iterator>
class iterator_traits : private std::iterator_traits<Iterator> {
private:
	typedef std::iterator_traits<Iterator> base_type;
public:
	typedef typename base_type::difference_type    difference_type;
	typedef typename base_type::iterator_category  iterator_category;
	typedef typename base_type::pointer            pointer;
	typedef typename base_type::reference          reference;
	typedef typename base_type::value_type         value_type;
	typedef          std::false_type               is_device_iterator;
	//                                             is_contiguous (deliberately not present for host memory iterators)
};

template<typename T,typename PointerType,typename Category>
class iterator_traits< device_iterator<T,PointerType,Category> > : private std::iterator_traits< device_iterator<T,PointerType,Category> > {
private:
	typedef std::iterator_traits< device_iterator<T,PointerType,Category> > base_type;
public:
	typedef typename base_type::difference_type    difference_type;
	typedef typename base_type::iterator_category  iterator_category;
	typedef typename base_type::pointer            pointer;
	typedef typename base_type::reference          reference;
	typedef typename base_type::value_type         value_type;
	typedef          std::true_type                is_device_iterator;
	typedef          std::false_type               is_contiguous;
};

template<typename T>
class iterator_traits< device_contiguous_iterator<T> > : private std::iterator_traits< device_contiguous_iterator<T> > {
private:
	typedef std::iterator_traits< device_contiguous_iterator<T> > base_type;
public:
	typedef typename base_type::difference_type    difference_type;
	typedef typename base_type::iterator_category  iterator_category;
	typedef typename base_type::pointer            pointer;
	typedef typename base_type::reference          reference;
	typedef typename base_type::value_type         value_type;
	typedef          std::true_type                is_device_iterator;
	typedef          std::true_type                is_contiguous;
};

template<typename T,typename P>
class iterator_traits< device_contiguous_block_iterator<T,P> > : private iterator_traits< device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag> > {
private:
	typedef iterator_traits< device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag> > base_type;
public:
	typedef typename base_type::difference_type    difference_type;
	typedef typename base_type::iterator_category  iterator_category;
	typedef typename base_type::pointer            pointer;
	typedef typename base_type::reference          reference;
	typedef typename base_type::value_type         value_type;
	typedef typename base_type::is_device_iterator is_device_iterator;
	typedef          std::true_type                is_contiguous;
};

template<typename Iterator>
class iterator_traits< reverse_device_iterator<Iterator> > : private std::iterator_traits< reverse_device_iterator<Iterator> > {
private:
	typedef std::iterator_traits< reverse_device_iterator<Iterator> > base_type;
public:
	typedef typename base_type::difference_type    difference_type;
	typedef typename base_type::iterator_category  iterator_category;
	typedef typename base_type::pointer            pointer;
	typedef typename base_type::reference          reference;
	typedef typename base_type::value_type         value_type;
	typedef          std::true_type                is_device_iterator;
	typedef          std::false_type               is_contiguous;
};

namespace impl {

template<class InputIterator, typename Distance>
__HOST__ __DEVICE__ inline
void advance(
	InputIterator& iterator,
	Distance n,
	std::true_type // device memory
)
{
	typedef typename ecuda::iterator_traits<InputIterator>::iterator_category iterator_category;
	typedef typename ecuda::iterator_traits<InputIterator>::is_contiguous     iterator_contiguity;
	const bool isIteratorSomeKindOfContiguous =
		std::is_same<iterator_contiguity,std::true_type>::value ||
		std::is_same< iterator_category, device_contiguous_block_iterator_tag >::value;
	if( isIteratorSomeKindOfContiguous ) {
		iterator += n;
	} else {
		for( Distance i = 0; i < n; ++i ) --iterator;
	}
}

template<class InputIterator, typename Distance>
__HOST__ __DEVICE__ inline
void advance(
	InputIterator& iterator,
	Distance n,
	std::false_type // host memory
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT( false, CANNOT_ACCESS_HOST_MEMORY_FROM_DEVICE_CODE );
	#else
	std::advance( iterator, n );
	#endif
}

} // namespace impl

template<class InputIterator,typename Distance>
__HOST__ __DEVICE__ inline void advance( InputIterator& iterator, Distance n )
{
	impl::advance( iterator, n, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

namespace impl {

template<class Iterator>
__HOST__ __DEVICE__ inline // yes, it's inline since the actual run-time portion always resolves to a few statements
typename std::iterator_traits<Iterator>::difference_type distance(
	Iterator first, Iterator last,
	std::true_type // device memory
)
{
	typedef typename ecuda::iterator_traits<Iterator>::iterator_category iterator_category;
	typedef typename ecuda::iterator_traits<Iterator>::is_contiguous     iterator_contiguity;
	const bool isIteratorSomeKindOfContiguous =
		std::is_same<iterator_contiguity,std::true_type>::value ||
		std::is_same<iterator_category,device_contiguous_block_iterator_tag>::value;
	#ifdef __CUDA_ARCH__
	if( isIteratorSomeKindOfContiguous ) {
		return ( last - first );
	} else {
		typename std::iterator_traits<Iterator>::difference_type n = 0;
		while( first != last ) { ++n; ++first; }
		return n;
	}
	#else
	ECUDA_STATIC_ASSERT( isIteratorSomeKindOfContiguous, CANNOT_CALCULATE_DISTANCE_OF_NONCONTIGUOUS_DEVICE_ITERATOR_FROM_HOST_CODE );
	return ( last - first );
	#endif
}


template<class Iterator>
__HOST__ __DEVICE__ inline
typename std::iterator_traits<Iterator>::difference_type distance(
	Iterator first, Iterator last,
	std::false_type // host memory
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT( false, CANNOT_ACCESS_HOST_MEMORY_FROM_DEVICE_CODE );
	return 0;
	#else
	return std::distance( first, last );
	#endif
}

} // namespace impl

template<class Iterator>
__HOST__ __DEVICE__ inline typename std::iterator_traits<Iterator>::difference_type distance( Iterator first, Iterator last )
{
	return impl::distance( first, last, typename ecuda::iterator_traits<Iterator>::is_device_iterator() );
}


} // namespace ecuda

#endif
