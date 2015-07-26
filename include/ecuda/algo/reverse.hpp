#pragma once
#ifndef ECUDA_ALGO_REVERSE_HPP
#define ECUDA_ALGO_REVERSE_HPP

#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"
#include "../utility.hpp"
#include "../algorithm.hpp"

namespace ecuda {

// forward declaration
template<class BidirectionalIterator> __HOST__ __DEVICE__ inline void reverse( BidirectionalIterator first, BidirectionalIterator last );

template<class BidirectionalIterator,class IsContiguous>
__HOST__ __DEVICE__ inline void __reverse( BidirectionalIterator first, BidirectionalIterator last, ecuda::pair<detail::__false_type,IsContiguous> ) {
	#ifdef __CUDA_ARCH__
	return; // never actually gets called, just here to trick nvcc
	#else
	// just defer to STL
	std::reverse( first, last );
	#endif
}

template<class BidirectionalIterator>
__HOST__ __DEVICE__ inline void __reverse( BidirectionalIterator first, BidirectionalIterator last, ecuda::pair<detail::__true_type,detail::__true_type> ) {
	#ifdef __CUDA_ARCH__
	while( (first!=last) and (first!=--last) ) {
		ecuda::swap( *first, *last );
		++first;
	}
	#else
	typename ecuda::iterator_traits<BidirectionalIterator>::difference_type n = ecuda::distance( first, last );
	std::vector<typename iterator_traits<BidirectionalIterator>::value_type> v( n );
	ecuda::copy( first, last, v.begin() );
	std::reverse( v.begin(), v.end() );
	ecuda::copy( v.begin(), v.end(), first );
	#endif
}

template<class ForwardIterator,typename T>
__HOST__ __DEVICE__ inline void __reverse( ForwardIterator first, ForwardIterator last, const T& val, ecuda::pair<detail::__true_type,detail::__false_type> ) {
	#ifdef __CUDA_ARCH__
	while( (first!=last) and (first!=--last) ) {
		ecuda::swap( *first, *last );
		++first;
	}
	#else
	throw std::invalid_argument( EXCEPTION_MSG( "ecuda::reverse() cannot reverse a non-contiguous device range" ) );
	#endif
}

template<class BidirectionalIterator>
__HOST__ __DEVICE__ inline void reverse( BidirectionalIterator first, BidirectionalIterator last ) {
	__reverse(
		first, last,
		ecuda::pair<typename ecuda::iterator_traits<BidirectionalIterator>::is_device_iterator,typename ecuda::iterator_traits<BidirectionalIterator>::is_contiguous>()
	);
}

} // namespace ecuda

#endif
