#pragma once
#ifndef ECUDA_ALGO_REVERSE_HPP
#define ECUDA_ALGO_REVERSE_HPP

#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"
//#include "../utility.hpp"
#include "../algorithm.hpp"

namespace ecuda {

// forward declaration
template<class BidirectionalIterator> __HOST__ __DEVICE__ inline void reverse( BidirectionalIterator first, BidirectionalIterator last );

namespace impl {

template<class ForwardIterator>
__HOST__ __DEVICE__ inline void
reverse( ForwardIterator first, ForwardIterator last,
		 std::false_type // host memory
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_REVERSE_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	#else
	std::reverse( first, last );
	#endif
}

template<class ForwardIterator>
__HOST__ __DEVICE__ inline void
reverse( ForwardIterator first, ForwardIterator last,
		 std::true_type // device memory
)
{
	#ifdef __CUDA_ARCH__
	while( (first!=last) and (first!=--last) ) {
		ecuda::swap( *first, *last );
		++first;
	}
	#else
	{
		typedef typename ecuda::iterator_traits<ForwardIterator>::iterator_category iterator_category;
		typedef typename ecuda::iterator_traits<ForwardIterator>::is_contiguous iterator_contiguity;
		const bool isSomeKindOfContiguous =
			std::is_same<iterator_contiguity,std::true_type>::value ||
			std::is_same<iterator_category,device_contiguous_block_iterator_tag>::value;
		ECUDA_STATIC_ASSERT(isSomeKindOfContiguous,CANNOT_CALL_REVERSE_ON_NONCONTIGUOUS_DEVICE_MEMORY);
	}
	typedef typename std::remove_const<typename ecuda::iterator_traits<ForwardIterator>::value_type>::type value_type;
	std::vector< value_type, host_allocator<value_type> > v( ::ecuda::distance( first, last ) );
	::ecuda::copy( first, last, v.begin() );
	::ecuda::reverse( v.begin(), v.end() );
	::ecuda::copy( v.begin(), v.end(), first );
	#endif
}

} // namespace impl

template<class BidirectionalIterator>
__HOST__ __DEVICE__ inline void reverse( BidirectionalIterator first, BidirectionalIterator last ) {
	impl::reverse(
		first, last,
		typename ecuda::iterator_traits<BidirectionalIterator>::is_device_iterator()
	);
}

} // namespace ecuda

#endif
