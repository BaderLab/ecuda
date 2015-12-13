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
#include "algorithm.hpp" // for ecuda::swap
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

template<typename T,typename P,typename Category=device_iterator_tag>
class device_iterator //: std::iterator<Category,T,std::ptrdiff_t,PointerType>
{
//private:
//	typedef std::iterator<Category,T,std::ptrdiff_t,P> base_type;

public:
	typedef Category                                      iterator_category;
	typedef T                                             value_type;
	typedef std::ptrdiff_t                                difference_type;
	typedef P                                             pointer;
	typedef typename ecuda::add_lvalue_reference<T>::type reference;

	template<typename U,typename Q,typename Category2> friend class device_iterator;
	template<typename U> friend class device_contiguous_iterator;

protected:
	pointer ptr;

public:
	__HOST__ __DEVICE__ device_iterator( const pointer& ptr = pointer() ) : ptr(ptr) {}
	__HOST__ __DEVICE__ device_iterator( const device_iterator& src ) : ptr(src.ptr) {}
	template<typename U,typename Q> __HOST__ __DEVICE__ device_iterator( const device_iterator<U,Q,Category>& src ) : ptr(src.ptr) {}

	__HOST__ __DEVICE__ inline device_iterator& operator=( const device_iterator& other )
	{
		ptr = other.ptr;
		return *this;
	}

	#ifdef __CPP11_SUPPORTED__
	__HOST__ device_iterator( device_iterator&& src ) : ptr(std::move(src.ptr)) {}
	__HOST__ device_iterator& operator=( device_iterator&& src )
	{
		ptr = std::move( src.ptr );
		return *this;
	}
	#endif

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

	template<typename U,typename Q>
	__HOST__ __DEVICE__ inline device_iterator& operator=( const device_iterator<U,Q,Category>& other )
	{
		ptr = other.ptr;
		return *this;
	}

};

template<typename T>
class device_contiguous_iterator : public device_iterator<T,typename ecuda::add_pointer<T>::type,device_contiguous_iterator_tag>
{

private:
	typedef device_iterator<T,typename ecuda::add_pointer<T>::type,device_contiguous_iterator_tag> base_type;

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

	__HOST__ __DEVICE__ device_contiguous_iterator& operator=( const device_contiguous_iterator& other )
	{
		base_type::operator=(other);
		return *this;
	}

	#ifdef __CPP11_SUPPORTED__
	__HOST__ device_contiguous_iterator( device_contiguous_iterator&& src ) : base_type(std::move(src)) {}
	__HOST__ device_contiguous_iterator& operator=( device_contiguous_iterator&& src )
	{
		base_type::operator=(std::move(src));
		return *this;
	}
	#endif

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

private:
	difference_type width;  //!< number of elements in a contiguous block
	difference_type offset; //!< offset in the current contiguous block
	template<typename U,typename Q> friend class device_contiguous_block_iterator;

public:
	__HOST__ __DEVICE__ device_contiguous_block_iterator( const pointer& ptr, const difference_type width, const difference_type offset = difference_type() ) : base_type(ptr), width(width), offset(offset) {}
	__HOST__ __DEVICE__ device_contiguous_block_iterator( const device_contiguous_block_iterator& src ) : base_type(src), width(src.width), offset(src.offset) {}
	template<typename U,typename Q>
	__HOST__ __DEVICE__ device_contiguous_block_iterator( const device_contiguous_block_iterator<U,Q>& src ) : base_type(src), width(src.width), offset(src.offset) {}

	#ifdef __CPP11_SUPPORTED__
	__HOST__ device_contiguous_block_iterator( device_contiguous_block_iterator&& src ) :
		base_type(std::move(src)),
		width(std::move(src.width)),
		offset(std::move(src.offset))
	{
	}

	__HOST__ device_contiguous_block_iterator& operator=( device_contiguous_block_iterator&& src )
	{
		base_type::operator=(std::move(src));
		width = std::move(src.width);
		offset = std::move(src.offset);
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ device_contiguous_block_iterator& operator++()
	{
		++base_type::ptr;
		++offset;
		if( offset == width ) {
			base_type::ptr.skip_bytes( base_type::ptr.get_pitch() - width*sizeof(value_type) ); // move past padding
			offset = 0;
		}
		return *this;
	}
	__HOST__ __DEVICE__ inline device_contiguous_block_iterator operator++( int )
	{
		device_contiguous_block_iterator tmp(*this);
		++(*this);
		return tmp;
	}

	__HOST__ __DEVICE__ device_contiguous_block_iterator& operator--()
	{
		--base_type::ptr;
		--offset;
		if( offset < 0 ) {
			base_type::ptr.skip_bytes( width*sizeof(value_type) - base_type::ptr.get_pitch() ); // move past padding
			offset = width-1;
		}
		return *this;
	}
	__HOST__ __DEVICE__ inline device_contiguous_block_iterator operator--( int )
	{
		device_contiguous_block_iterator tmp(*this);
		--(*this);
		return tmp;
	}

	__HOST__ __DEVICE__ device_contiguous_block_iterator operator+( int x ) const
	{
		device_contiguous_block_iterator tmp( *this );
		tmp += x;
		return tmp;
	}

	__HOST__ __DEVICE__ inline device_contiguous_block_iterator operator-( int x ) const { return operator+(-x); }

	__HOST__ __DEVICE__ device_contiguous_block_iterator& operator+=( int x )
	{
		const int rows = x / width;
		base_type::ptr.skip_bytes( rows * base_type::ptr.get_pitch() );
		x -= rows * width;
		base_type::ptr += x;
		offset += x;
		if( offset > width ) { base_type::ptr.skip_bytes( base_type::ptr.get_pitch() - width*sizeof(value_type) ); offset -= width; }
		if( offset < 0     ) { base_type::ptr.skip_bytes( width*sizeof(value_type) - base_type::ptr.get_pitch() ); offset += width; }
		return *this;
	}
	__HOST__ __DEVICE__ inline device_contiguous_block_iterator& operator-=( int x ) { operator+=(-x); return *this; }

	__DEVICE__ inline reference operator[]( int x ) const { return *operator+(x); }

	__HOST__ __DEVICE__ inline difference_type operator-( const device_contiguous_block_iterator& other ) const
	{
		typedef const char* char_pointer_type;
		char_pointer_type p = naked_cast<char_pointer_type>(base_type::ptr);
		char_pointer_type q = naked_cast<char_pointer_type>(other.ptr);
		difference_type span = ( p - q ); // bytes difference
		difference_type diff = span/base_type::ptr.get_pitch()*width;
		span = span % base_type::ptr.get_pitch();
		if( span > (width*sizeof(value_type) ) )
			diff += width;
		else
			diff += span / sizeof(value_type);
		return diff;
	}

	__HOST__ __DEVICE__ inline bool operator< ( const device_contiguous_block_iterator& other ) const __NOEXCEPT__ { return base_type::ptr < other.ptr; }
	__HOST__ __DEVICE__ inline bool operator> ( const device_contiguous_block_iterator& other ) const __NOEXCEPT__ { return base_type::ptr > other.ptr; }
	__HOST__ __DEVICE__ inline bool operator<=( const device_contiguous_block_iterator& other ) const __NOEXCEPT__ { return operator<(other) or operator==(other); }
	__HOST__ __DEVICE__ inline bool operator>=( const device_contiguous_block_iterator& other ) const __NOEXCEPT__ { return operator>(other) or operator==(other); }

	__HOST__ __DEVICE__ inline contiguous_iterator contiguous_begin() const __NOEXCEPT__ { return contiguous_iterator( naked_cast<typename ecuda::add_pointer<T>::type>( base_type::ptr.get() ) ); }
	__HOST__ __DEVICE__ inline contiguous_iterator contiguous_end()   const __NOEXCEPT__ { return contiguous_iterator( naked_cast<typename ecuda::add_pointer<T>::type>( base_type::ptr.get() ) + (width-offset) ); }
	__HOST__ __DEVICE__ inline std::size_t get_width() const __NOEXCEPT__ { return width; }
	__HOST__ __DEVICE__ inline std::size_t get_offset() const __NOEXCEPT__ { return offset; }

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

	#ifdef __CPP11_SUPPORTED__
	__HOST__ reverse_device_iterator( reverse_device_iterator&& src ) { ecuda::swap( parentIterator, src.parentIterator ); }
	__HOST__ reverse_device_iterator& operator=( reverse_device_iterator&& src )
	{
		ecuda::swap( parentIterator, src.parentIterator );
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ Iterator base() const { return parentIterator; }

	__HOST__ __DEVICE__ inline reverse_device_iterator& operator++() { --parentIterator; return *this; }
	__HOST__ __DEVICE__ inline reverse_device_iterator operator++( int )
	{
		reverse_device_iterator tmp(*this);
		++(*this);
		return tmp;
	}

	__HOST__ __DEVICE__ inline reverse_device_iterator& operator--() { ++parentIterator; return *this; }
	__HOST__ __DEVICE__ inline reverse_device_iterator operator--( int )
	{
		reverse_device_iterator tmp(*this);
		--(*this);
		return tmp;
	}

	__HOST__ __DEVICE__ inline bool operator==( const reverse_device_iterator& other ) const { return parentIterator == other.parentIterator; }
	__HOST__ __DEVICE__ inline bool operator!=( const reverse_device_iterator& other ) const { return !operator==(other); }

	__DEVICE__ inline reference operator*() const
	{
		Iterator tmp(parentIterator);
		--tmp;
		return tmp.operator*();
	}

	__HOST__ __DEVICE__ inline pointer operator->() const
	{
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

	__HOST__ __DEVICE__ reverse_device_iterator& operator=( const reverse_device_iterator& other )
	{
		parentIterator = other.parentIterator;
		return *this;
	}

	template<class Iterator2>
	__HOST__ __DEVICE__ reverse_device_iterator& operator=( const reverse_device_iterator<Iterator2>& other )
	{
		parentIterator = other.parentIterator;
		return *this;
	}

};

template<class Iterator>
class iterator_traits : private std::iterator_traits<Iterator>
{
private:
	typedef std::iterator_traits<Iterator> base_type;
public:
	typedef typename base_type::difference_type    difference_type;
	typedef typename base_type::iterator_category  iterator_category;
	typedef typename base_type::pointer            pointer;
	typedef typename base_type::reference          reference;
	typedef typename base_type::value_type         value_type;
	typedef          ecuda::false_type             is_device_iterator;
	//                                             is_contiguous (deliberately not present for host memory iterators)
};

template<typename T,typename PointerType,typename Category>
class iterator_traits< device_iterator<T,PointerType,Category> > : private std::iterator_traits< device_iterator<T,PointerType,Category> >
{
private:
	typedef std::iterator_traits< device_iterator<T,PointerType,Category> > base_type;
public:
	typedef typename base_type::difference_type    difference_type;
	typedef typename base_type::iterator_category  iterator_category;
	typedef typename base_type::pointer            pointer;
	typedef typename base_type::reference          reference;
	typedef typename base_type::value_type         value_type;
	typedef          ecuda::true_type              is_device_iterator;
	typedef          ecuda::false_type             is_contiguous;
};

template<typename T>
class iterator_traits< device_contiguous_iterator<T> > : private std::iterator_traits< device_contiguous_iterator<T> >
{
private:
	typedef std::iterator_traits< device_contiguous_iterator<T> > base_type;
public:
	typedef typename base_type::difference_type    difference_type;
	typedef typename base_type::iterator_category  iterator_category;
	typedef typename base_type::pointer            pointer;
	typedef typename base_type::reference          reference;
	typedef typename base_type::value_type         value_type;
	typedef          ecuda::true_type              is_device_iterator;
	typedef          ecuda::true_type              is_contiguous;
};

template<typename T,typename P>
class iterator_traits< device_contiguous_block_iterator<T,P> > : private iterator_traits< device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag> >
{
private:
	typedef iterator_traits< device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag> > base_type;
public:
	typedef typename base_type::difference_type    difference_type;
	typedef typename base_type::iterator_category  iterator_category;
	typedef typename base_type::pointer            pointer;
	typedef typename base_type::reference          reference;
	typedef typename base_type::value_type         value_type;
	typedef typename base_type::is_device_iterator is_device_iterator;
	typedef          ecuda::true_type              is_contiguous;
};

template<typename Iterator>
class iterator_traits< reverse_device_iterator<Iterator> > : private std::iterator_traits< reverse_device_iterator<Iterator> >
{
private:
	typedef std::iterator_traits< reverse_device_iterator<Iterator> > base_type;
public:
	typedef typename base_type::difference_type    difference_type;
	typedef typename base_type::iterator_category  iterator_category;
	typedef typename base_type::pointer            pointer;
	typedef typename base_type::reference          reference;
	typedef typename base_type::value_type         value_type;
	typedef          ecuda::true_type              is_device_iterator;
	typedef          ecuda::false_type             is_contiguous;
};

template<typename T>
class iterator_traits<T*> : private std::iterator_traits<T*>
{
private:
	typedef std::iterator_traits<T*> base_type;
public:
	typedef typename base_type::difference_type    difference_type;
	typedef typename base_type::iterator_category  iterator_category;
	typedef typename base_type::pointer            pointer;
	typedef typename base_type::reference          reference;
	typedef typename base_type::value_type         value_type;
	typedef          ecuda::false_type             is_device_iterator;
	//                                             is_contiguous (deliberately not present for host memory iterators)
};

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class InputIterator, typename Distance>
__HOST__ __DEVICE__ inline
void advance(
	InputIterator& iterator,
	Distance n,
	ecuda::true_type // device memory
)
{
	typedef typename ecuda::iterator_traits<InputIterator>::iterator_category iterator_category;
	typedef typename ecuda::iterator_traits<InputIterator>::is_contiguous     iterator_contiguity;
	const bool isIteratorSomeKindOfContiguous =
		ecuda::is_same<iterator_contiguity,ecuda::true_type>::value ||
		ecuda::is_same< iterator_category, device_contiguous_block_iterator_tag >::value;
	if( isIteratorSomeKindOfContiguous ) {
		iterator += n;
	} else {
		for( Distance i = 0; i < n; ++i ) ++iterator;
	}
}

template<class InputIterator, typename Distance>
__HOST__ __DEVICE__ inline
void advance(
	InputIterator& iterator,
	Distance n,
	ecuda::false_type // host memory
)
{
	#ifdef __CUDA_ARCH__
	// never called from device code
	#else
	std::advance( iterator, n );
	#endif
}

} // namespace impl
/// \endcond

///
/// \brief Increments given iterator by n elements.
///
/// If n is negative, the iterator is decremented. This function will work
/// on both iterators of host and device memory. However, if the iterator
/// refers to non-contiguous device memory and this function is called from
/// host code an assertion will fail at compile-time.
///
/// \param iterator iterator to be advanced
/// \param n number of elements iterator should be advanced
///
ECUDA_SUPPRESS_HD_WARNINGS
template<class InputIterator,typename Distance>
__HOST__ __DEVICE__ inline void advance( InputIterator& iterator, Distance n )
{
	impl::advance( iterator, n, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class Iterator>
__HOST__ __DEVICE__ inline // yes, it's inline since the actual run-time portion always resolves to a few statements
typename std::iterator_traits<Iterator>::difference_type distance(
	Iterator first, Iterator last,
	ecuda::true_type // device memory
)
{
	typedef typename ecuda::iterator_traits<Iterator>::iterator_category iterator_category;
	typedef typename ecuda::iterator_traits<Iterator>::is_contiguous     iterator_contiguity;
	const bool isIteratorSomeKindOfContiguous =
		ecuda::is_same<iterator_contiguity,ecuda::true_type>::value ||
		ecuda::is_same<iterator_category,device_contiguous_block_iterator_tag>::value;
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
	const Iterator& first, const Iterator& last,
	ecuda::false_type // host memory
)
{
	#ifdef __CUDA_ARCH__
	return 0; // never called from device code
	#else
	return std::distance( first, last );
	#endif
}

} // namespace impl
/// \endcond

ECUDA_SUPPRESS_HD_WARNINGS
template<class Iterator>
__HOST__ __DEVICE__ inline typename std::iterator_traits<Iterator>::difference_type distance( const Iterator& first, const Iterator& last )
{
	return impl::distance( first, last, typename ecuda::iterator_traits<Iterator>::is_device_iterator() );
}

template<typename T,typename P>
__HOST__ __DEVICE__
inline
typename std::iterator_traits< device_contiguous_block_iterator<T,P> >::difference_type
distance( const device_contiguous_block_iterator<T,P>& first, const device_contiguous_block_iterator<T,P>& last )
{
	return last - first;
}

} // namespace ecuda

#endif
