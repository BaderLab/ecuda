//----------------------------------------------------------------------------
// algo/max_element.hpp
//
// Extension of std::max_element that recognizes device memory and can be
// called from host or device code.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------
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

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class ForwardIterator>
__HOST__ __DEVICE__ inline ForwardIterator max_element( ForwardIterator first, ForwardIterator last, ecuda::true_type ) {
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
	const bool isIteratorContiguous = ecuda::is_same<typename ecuda::iterator_traits<ForwardIterator>::is_contiguous,ecuda::true_type>::value;
	ECUDA_STATIC_ASSERT(isIteratorContiguous,CANNOT_FIND_MAX_ELEMENT_IN_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_MEMORY);
	typename ecuda::iterator_traits<ForwardIterator>::difference_type n = ecuda::distance( first, last );
	std::vector<typename ecuda::iterator_traits<ForwardIterator>::value_type> hostVector( n );
	ecuda::copy( first, last, hostVector.begin() );
	const std::size_t index = std::max_element( hostVector.begin(), hostVector.end() ) - hostVector.begin();
	first += index;
	return first;
	#endif
}


template<class ForwardIterator>
__HOST__ __DEVICE__ inline ForwardIterator max_element( ForwardIterator first, ForwardIterator last, ecuda::false_type ) {
	// not an iterator to device memory, delegate to STL
	#ifdef __CUDA_ARCH__
	return last; // can never be called from device code, dummy return to satisfy nvcc
	#else
	return std::max_element( first, last );
	#endif
}

} // namespace impl
/// \endcond

template<class ForwardIterator>
__HOST__ __DEVICE__ inline ForwardIterator max_element( ForwardIterator first, ForwardIterator last ) {
	impl::max_element(
		first, last,
		typename ecuda::iterator_traits<ForwardIterator>::is_device_iterator()
	);
}

} // namespace ecuda

#endif
