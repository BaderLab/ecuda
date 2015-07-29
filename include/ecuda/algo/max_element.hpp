#pragma once
#ifndef ECUDA_ALGO_MAX_ELEMENT_HPP
#define ECUDA_ALGO_MAX_ELEMENT_HPP

#include <algorithm>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"

namespace ecuda {

// forward declaration
template<class ForwardIterator> __HOST__ __DEVICE__ inline ForwardIterator max_element( ForwardIterator first, ForwardIterator last );

template<class ForwardIterator>
__HOST__ __DEVICE__ inline ForwardIterator __max_element( ForwardIterator first, ForwardIterator last, detail::__true_type ) {
	// is an iterator to device memory
	#ifdef __CUDA_ARCH__
	if( first == last ) return last;
	ForwardIterator largest = first;
	++first;
	for( ; first != last; ++first ) {
		if( *largest < *first ) largest = first;
	}
	return largest;
	#else
	const bool isIteratorContiguous = std::is_same<typename ecuda::iterator_traits<ForwardIterator>::is_contiguous,detail::__true_type>::value;
	ECUDA_STATIC_ASSERT(isIteratorContiguous,CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_WITH_MAX_ELEMENT);
	typename ecuda::iterator_traits<ForwardIterator>::difference_type n = ecuda::distance( first, last );
	std::vector<typename ecuda::iterator_traits<ForwardIterator>::value_type> hostVector( n );
	ecuda::copy( first, last, hostVector.begin() );
	const std::size_t index = std::max_element( hostVector.begin(), hostVector.end() ) - hostVector.begin();
	first += index;
	return first;
	#endif
}


template<class ForwardIterator>
__HOST__ __DEVICE__ inline ForwardIterator __max_element( ForwardIterator first, ForwardIterator last, detail::__false_type ) {
	// not an iterator to device memory, delegate to STL
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_MAX_ELEMENT_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return last; // can never be called from device code, dummy return to satisfy nvcc
	#else
	return std::max_element( first, last );
	#endif
}


template<class ForwardIterator>
__HOST__ __DEVICE__ inline ForwardIterator max_element( ForwardIterator first, ForwardIterator last ) {
	__max_element(
		first, last,
		typename ecuda::iterator_traits<ForwardIterator>::is_device_iterator()
	);
}

} // namespace ecuda

#endif
