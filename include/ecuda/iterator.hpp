#pragma once
#ifndef ECUDA_ITERATOR_HPP
#define ECUDA_ITERATOR_HPP

#include <iterator>

#include "global.hpp"
#include "type_traits.hpp"

namespace ecuda {

// NOTE: libc++ requires inheritance from one of the 5 STL iterator categories (libstdc++ does not)

///
/// \brief Iterator category denoting device memory.
///
struct device_iterator_tag : ::std::bidirectional_iterator_tag {};

///
/// \brief Iterator category denoting contiguous device memory.
///
struct device_contiguous_iterator_tag : ::std::random_access_iterator_tag {}; // libc++ requires inheritance from one of the 5 STL iterator categories

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
	typedef Category iterator_category;
	typedef T value_type;
	typedef std::ptrdiff_t difference_type;
	typedef PointerType pointer;
	typedef typename std::add_lvalue_reference<T>::type reference;
	//typedef typename base_type::iterator_category iterator_category;
	//typedef typename base_type::value_type value_type;
	//typedef typename base_type::difference_type difference_type;
	//typedef typename base_type::pointer pointer;
	//typedef typename base_type::reference reference;

	template<typename T2,typename PointerType2,typename Category2> friend class device_iterator;
	template<typename T2> friend class device_contiguous_iterator;

protected:
	pointer ptr;

public:
	__HOST__ __DEVICE__ device_iterator( const pointer& ptr = pointer() ) : ptr(ptr) {}
	__HOST__ __DEVICE__ device_iterator( const device_iterator& src ) : ptr(src.ptr) {}
	template<typename T2,typename PointerType2>
	__HOST__ __DEVICE__ device_iterator( const device_iterator<T2,PointerType2,Category>& src ) : ptr(src.ptr) {}

	__HOST__ __DEVICE__ inline device_iterator& operator++() { ++ptr; return *this; }
	__HOST__ __DEVICE__ inline device_iterator operator++( int ) {
		device_iterator tmp(*this);
		++(*this);
		return tmp;
	}

	__HOST__ __DEVICE__ inline device_iterator& operator--() { --ptr; return *this; }
	__HOST__ __DEVICE__ inline device_iterator operator--( int ) {
		device_iterator tmp(*this);
		--(*this);
		return tmp;
	}

	__HOST__ __DEVICE__ inline bool operator==( const device_iterator& other ) const __NOEXCEPT__ { return ptr == other.ptr; }
	__HOST__ __DEVICE__ inline bool operator!=( const device_iterator& other ) const __NOEXCEPT__ { return !operator==(other); }

	__DEVICE__ inline reference operator*() { return *ptr; }
	__HOST__ __DEVICE__ inline pointer operator->() const { return ptr; }

	__HOST__ __DEVICE__ inline device_iterator& operator=( const device_iterator& other ) {
		ptr = other.ptr;
		return *this;
	}

	template<typename U,typename PointerType2>
	__HOST__ __DEVICE__ inline device_iterator& operator=( const device_iterator<U,PointerType2,Category>& other ) {
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
	typedef typename base_type::value_type value_type;
	typedef typename base_type::difference_type difference_type;
	typedef typename base_type::pointer pointer;
	typedef typename base_type::reference reference;

public:
	__HOST__ __DEVICE__ device_contiguous_iterator( const pointer& ptr = pointer() ) : base_type(ptr) {}
	__HOST__ __DEVICE__ device_contiguous_iterator( const device_contiguous_iterator& src ) : base_type(src) {}
	template<typename U>
	__HOST__ __DEVICE__ device_contiguous_iterator( const device_contiguous_iterator<U>& src ) : base_type(src) {}

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
	typedef typename base_type::value_type value_type;
	typedef typename base_type::difference_type difference_type;
	typedef typename base_type::pointer pointer;
	typedef typename base_type::reference reference;

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

	//__HOST__ __DEVICE__ inline difference_type operator-( const device_contiguous_block_iterator& other ) { return base_type::ptr - other.ptr; }

	__HOST__ __DEVICE__ inline bool operator<( const device_contiguous_block_iterator& other ) const __NOEXCEPT__ { return base_type::ptr < other.ptr; }
	__HOST__ __DEVICE__ inline bool operator>( const device_contiguous_block_iterator& other ) const __NOEXCEPT__ { return base_type::ptr > other.ptr; }
	__HOST__ __DEVICE__ inline bool operator<=( const device_contiguous_block_iterator& other ) const __NOEXCEPT__ { return operator<(other) or operator==(other); }
	__HOST__ __DEVICE__ inline bool operator>=( const device_contiguous_block_iterator& other ) const __NOEXCEPT__ { return operator>(other) or operator==(other); }

};


template<class Iterator>
class reverse_device_iterator //: public std::iterator<device_iterator_tag,typename Iterator::value_type,typename Iterator::difference_type,typename Iterator::pointer>
{
private:
	typedef std::iterator<device_iterator_tag,typename Iterator::value_type,typename Iterator::difference_type,typename Iterator::pointer> base_type;

public:
	typedef Iterator iterator_type;
	typedef typename base_type::iterator_category iterator_category;
	typedef typename base_type::value_type value_type;
	typedef typename base_type::difference_type difference_type;
	typedef typename base_type::pointer pointer;
	typedef typename base_type::reference reference;

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

template<class IteratorCategory> struct __is_contiguous { typedef detail::__false_type type; };
template<> struct __is_contiguous<std::random_access_iterator_tag> { typedef detail::__true_type type; };
template<typename T> struct __is_contiguous<T*> { typedef detail::__true_type type; };
template<typename T> struct __is_contiguous<const T*> { typedef detail::__true_type type; };

template<class Iterator>
struct iterator_traits : std::iterator_traits<Iterator> {
	typedef typename std::iterator_traits<Iterator>::difference_type difference_type;
	typedef typename std::iterator_traits<Iterator>::iterator_category iterator_category;
	typedef typename std::iterator_traits<Iterator>::pointer pointer;
	typedef typename std::iterator_traits<Iterator>::reference reference;
	typedef typename std::iterator_traits<Iterator>::value_type value_type;
	typedef typename detail::__false_type is_device_iterator;
	typedef typename __is_contiguous<typename std::iterator_traits<Iterator>::iterator_category>::type is_contiguous;
	static bool confirm_contiguity( Iterator first, Iterator last, difference_type n ) {
		if( !std::is_same<detail::__true_type,is_contiguous>::value ) return false;
		const pointer pFirst = first.operator->();
		const pointer pLast = last.operator->();
		return ( pLast - pFirst ) == n;
	}
};

template<typename T,typename PointerType,typename Category>
struct iterator_traits< device_iterator<T,PointerType,Category> > : std::iterator_traits< device_iterator<T,PointerType,Category> > {
	typedef typename std::iterator_traits< device_iterator<T,PointerType,Category> >::difference_type difference_type;
	typedef typename std::iterator_traits< device_iterator<T,PointerType,Category> >::iterator_category iterator_category;
	typedef typename std::iterator_traits< device_iterator<T,PointerType,Category> >::pointer pointer;
	typedef typename std::iterator_traits< device_iterator<T,PointerType,Category> >::reference reference;
	typedef typename std::iterator_traits< device_iterator<T,PointerType,Category> >::value_type value_type;
	typedef typename detail::__true_type is_device_iterator;
	typedef typename detail::__false_type is_contiguous;
};

template<typename T>
struct iterator_traits< device_contiguous_iterator<T> > : std::iterator_traits< device_contiguous_iterator<T> > {
	typedef typename std::iterator_traits< device_contiguous_iterator<T> >::difference_type difference_type;
	typedef typename std::iterator_traits< device_contiguous_iterator<T> >::iterator_category iterator_category;
	typedef typename std::iterator_traits< device_contiguous_iterator<T> >::pointer pointer;
	typedef typename std::iterator_traits< device_contiguous_iterator<T> >::reference reference;
	typedef typename std::iterator_traits< device_contiguous_iterator<T> >::value_type value_type;
	typedef typename detail::__true_type is_device_iterator;
	typedef typename detail::__true_type is_contiguous;
};

template<typename T,typename P>
struct iterator_traits< device_contiguous_block_iterator<T,P> > : iterator_traits< device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag> > {
	typedef typename iterator_traits< device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag> >::difference_type difference_type;
	typedef typename iterator_traits< device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag> >::iterator_category iterator_category;
	typedef typename iterator_traits< device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag> >::pointer pointer;
	typedef typename iterator_traits< device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag> >::reference reference;
	typedef typename iterator_traits< device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag> >::value_type value_type;
//	typedef typename detail::__true_type is_device_iterator;
//	typedef typename detail::__true_type is_contiguous;
	typedef typename iterator_traits< device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag> >::is_device_iterator is_device_iterator;
	typedef typename iterator_traits< device_iterator<T,padded_ptr<T,P>,device_contiguous_block_iterator_tag> >::is_contiguous is_contiguous;
};

template<typename Iterator>
struct iterator_traits< reverse_device_iterator<Iterator> > : std::iterator_traits< reverse_device_iterator<Iterator> > {
	typedef typename std::iterator_traits< reverse_device_iterator<Iterator> >::difference_type difference_type;
	typedef typename std::iterator_traits< reverse_device_iterator<Iterator> >::iterator_category iterator_category;
	typedef typename std::iterator_traits< reverse_device_iterator<Iterator> >::pointer pointer;
	typedef typename std::iterator_traits< reverse_device_iterator<Iterator> >::reference reference;
	typedef typename std::iterator_traits< reverse_device_iterator<Iterator> >::value_type value_type;
	typedef typename detail::__true_type is_device_iterator;
	typedef typename detail::__false_type is_contiguous;
};

/*
template<class InputIterator,typename Distance,class IsContiguous>
__HOST__ __DEVICE__ inline void __advance( InputIterator& iterator, Distance n, detail::__false_type, IsContiguous ) {
	// just defer to STL
	#ifdef __CUDA_ARCH__
	return; // never actually gets called, just here to trick nvcc
	#else
	std::advance( iterator, n );
	#endif
}

template<class InputIterator,typename Distance>
__HOST__ __DEVICE__ inline void __advance( InputIterator& iterator, Distance n, detail::__true_type, detail::__true_type ) {
	iterator += n;
}

template<class InputIterator,typename Distance>
__HOST__ __DEVICE__ inline void __advance( InputIterator& iterator, Distance n, detail::__true_type, detail::__false_type ) {
	#ifdef __CUDA_ARCH__
	for( Distance i = 0; i < n; ++i ) ++iterator;
	#else
	throw std::invalid_argument( EXCEPTION_MSG( "ecuda::advance() cannot advance non-contiguous device iterator from host code" ) );
	#endif
}
*/

template<class InputIterator, typename Distance>
__HOST__ __DEVICE__ inline void __advance( InputIterator& iterator, Distance n, device_iterator_tag ) {
//	#ifdef __CUDA_ARCH__
	for( Distance i = 0; i < n; ++i ) ++iterator;
//	#else
//	// never gets called
//	#endif
}

template<class InputIterator, typename Distance>
__HOST__ __DEVICE__ inline void __advance( InputIterator& iterator, Distance n, device_contiguous_iterator_tag ) {
//	#ifdef __CUDA_ARCH__
	iterator += n;
//	#else
//	// never gets called
//	#endif
}

template<class InputIterator, typename Distance>
__HOST__ __DEVICE__ inline void __advance( InputIterator& iterator, Distance n, device_contiguous_block_iterator_tag ) {
//	#ifdef __CUDA_ARCH__
	iterator += n;
//	#else
//	// never gets called
//	#endif
}

template<class InputIterator, typename Distance>
__HOST__ __DEVICE__ inline void __advance( InputIterator& iterator, Distance n, detail::__device ) {
//	#ifdef __CUDA_ARCH__
//	#else
	const bool isIteratorContiguous = !std::is_same<typename ecuda::iterator_traits<InputIterator>::iterator_category, device_iterator_tag>::value;
	ECUDA_STATIC_ASSERT( isIteratorContiguous, CANNOT_ADVANCE_NONCONTIGUOUS_DEVICE_ITERATOR_FROM_HOST_CODE );
	__advance( iterator, n, typename ecuda::iterator_traits<InputIterator>::iterator_category() );
//	#endif
}

template<class InputIterator, typename Distance>
__HOST__ __DEVICE__ inline void __advance( InputIterator& iterator, Distance n, detail::__host ) {
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT( false, CANNOT_ACCESS_HOST_MEMORY_FROM_DEVICE_CODE );
	#else
	std::advance( iterator, n );
	#endif
}

template<class InputIterator,typename Distance>
__HOST__ __DEVICE__ inline void advance( InputIterator& iterator, Distance n ) {
	__advance( iterator, n, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
	//__advance( iterator, n, typename ecuda::iterator_traits<InputIterator>::is_device_iterator(), typename ecuda::iterator_traits<InputIterator>::is_contiguous() );
}

/*
template<class Iterator>
__HOST__ __DEVICE__ inline typename std::iterator_traits<Iterator>::difference_type __distance( Iterator first, Iterator last, detail::__true_type, detail::__true_type ) {
	return last-first;
}

template<class Iterator>
__HOST__ __DEVICE__ inline typename std::iterator_traits<Iterator>::difference_type __distance( Iterator first, Iterator last, detail::__true_type, detail::__false_type ) {
	#ifdef __CUDA_ARCH__
	typename std::iterator_traits<Iterator>::difference_type n = 0;
	while( first != last ) { ++n; ++first; }
	return n;
	#else
	throw std::invalid_argument( EXCEPTION_MSG( "ecuda::distance() cannot determine distance of a non-contiguous device iterator from host code" ) );
	#endif
}

template<class Iterator,class IsContiguous>
__HOST__ __DEVICE__ inline typename std::iterator_traits<Iterator>::difference_type __distance( Iterator first, Iterator last, detail::__false_type, IsContiguous ) {
	#ifdef __CUDA_ARCH__
	return 0; // never actually gets called, just here to trick nvcc
	#else
	// defer to STL
	return std::distance( first, last );
	#endif
}
*/

template<class Iterator>
__HOST__ __DEVICE__ inline typename std::iterator_traits<Iterator>::difference_type __distance( Iterator first, Iterator last, device_iterator_tag ) {
//	#ifdef __CUDA_ARCH__
	typename std::iterator_traits<Iterator>::difference_type n = 0;
	while( first != last ) { ++n; ++first; }
	return n;
//	#else
//	// never called
//	#endif
}

template<class Iterator>
__HOST__ __DEVICE__ inline typename std::iterator_traits<Iterator>::difference_type __distance( Iterator first, Iterator last, device_contiguous_iterator_tag ) {
	return last - first;
}

template<class Iterator>
__HOST__ __DEVICE__ inline typename std::iterator_traits<Iterator>::difference_type __distance( Iterator first, Iterator last, device_contiguous_block_iterator_tag ) {
	return last - first;
}

template<class Iterator>
__HOST__ __DEVICE__ inline typename std::iterator_traits<Iterator>::difference_type __distance( Iterator first, Iterator last, detail::__device ) {
	//#ifdef __CUDA_ARCH__
	//#else
	const bool isIteratorContiguous = !std::is_same< typename ecuda::iterator_traits<Iterator>::iterator_category, device_iterator_tag >::value;
	ECUDA_STATIC_ASSERT( isIteratorContiguous, CANNOT_CALCULATE_DISTANCE_OF_NONCONTIGUOUS_DEVICE_ITERATOR_FROM_HOST_CODE );
	//#endif
	return __distance( first, last, typename ecuda::iterator_traits<Iterator>::iterator_category() );
}


template<class Iterator>
__HOST__ __DEVICE__ inline typename std::iterator_traits<Iterator>::difference_type __distance( Iterator first, Iterator last, detail::__host ) {
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT( false, CANNOT_ACCESS_HOST_MEMORY_FROM_DEVICE_CODE );
	return 0;
	#else
	return std::distance( first, last );
	#endif
}


template<class Iterator>
__HOST__ __DEVICE__ inline typename std::iterator_traits<Iterator>::difference_type distance( Iterator first, Iterator last ) {
	return __distance( first, last, typename ecuda::iterator_traits<Iterator>::is_device_iterator() );
	//return __distance( first, last, typename ecuda::iterator_traits<Iterator>::is_device_iterator(), typename ecuda::iterator_traits<Iterator>::is_contiguous() );
	//return __distance( first, last, typename detail::__iterator_contiguity<typename std::iterator_traits<Iterator>::iterator_category>::type() );
}


} // namespace ecuda

#endif
