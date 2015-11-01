#pragma once
#ifndef ECUDA_ALGO_FILL_HPP
#define ECUDA_ALGO_FILL_HPP

#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"
#include "../utility.hpp"

namespace ecuda {

// forward declaration
template<class ForwardIterator,typename T> __HOST__ __DEVICE__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val );

namespace impl {

template<class ForwardIterator,typename T>
__HOST__ __DEVICE__ inline void fill_device( ForwardIterator first, ForwardIterator last, const T& val )
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { *first = val; ++first; }
	#else
	typedef typename ecuda::iterator_traits<ForwardIterator>::is_contiguous iterator_contiguity;
	{
		const bool isContiguous = std::is_same<iterator_contiguity,std::true_type>::value;
		ECUDA_STATIC_ASSERT(isContiguous,CANNOT_FILL_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_ITERATOR);
	}
	typename ecuda::iterator_traits<ForwardIterator>::difference_type n = ecuda::distance( first, last );
	typedef typename ecuda::iterator_traits<ForwardIterator>::value_type value_type;
	CUDA_CALL( cudaMemset<value_type>( first.operator->(), val, static_cast<std::size_t>(n) ) );
	#endif
}

template<typename T,typename P>
__HOST__ __DEVICE__ inline void fill_device(
	device_contiguous_block_iterator<T,P> first,
	device_contiguous_block_iterator<T,P> last,
	const T& val
)
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { *first = val; ++first; }
	#else
	typedef device_contiguous_block_iterator<T,P> input_iterator_type;
	typename ecuda::iterator_traits<input_iterator_type>::difference_type n = std::distance( first, last );
	while( n > 0 ) {
		const std::size_t width = first.operator->().get_remaining_width();
		::ecuda::fill( first.contiguous_begin(), first.contiguous_end(), val );
		::ecuda::advance( first, width );
		n -= width;
	}
	#endif
}

template<class ForwardIterator,typename T>
__HOST__ __DEVICE__ inline void fill(
	ForwardIterator first, ForwardIterator last,
	const T& val,
	std::true_type // device memory
)
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { *first = val; ++first; }
	#else
	typedef typename ecuda::iterator_traits<ForwardIterator>::is_contiguous iterator_contiguity;
	typedef typename ecuda::iterator_traits<ForwardIterator>::iterator_category iterator_category;
	{
		const bool isSomeKindOfContiguous =
			std::is_same<iterator_contiguity,std::true_type>::value ||
			std::is_same<iterator_category,device_contiguous_block_iterator_tag>::value;
		ECUDA_STATIC_ASSERT(isSomeKindOfContiguous,CANNOT_FILL_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_ITERATOR);
	}
	if( std::is_same<typename ecuda::iterator_traits<ForwardIterator>::value_type,T>::value ) {
		fill_device( first, last, val );
	} else {
		typedef typename ecuda::iterator_traits<ForwardIterator>::value_type value_type;
		const value_type val2( val );
		fill_device( first, last, val2 );
	}
	#endif
}

template<class ForwardIterator,typename T>
__HOST__ __DEVICE__ inline void fill(
	ForwardIterator first, ForwardIterator last,
	const T& val,
	std::false_type // host memory
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(false,CANNOT_CALL_FILL_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	#else
	std::fill( first, last, val );
	#endif
}

} // namespace impl

template<class ForwardIterator,typename T>
__HOST__ __DEVICE__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val )
{
	impl::fill(
		first, last,
		val,
		typename ecuda::iterator_traits<ForwardIterator>::is_device_iterator()
	);
}

} // namespace ecuda

#endif
