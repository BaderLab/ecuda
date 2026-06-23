//----------------------------------------------------------------------------
// algo/find_if.hpp
//
// Extension of std::find_if that recognizes device memory and can be called from
// host or device code.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ALGO_FIND_IF_HPP
#define ECUDA_ALGO_FIND_IF_HPP

#include <algorithm>
#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

ECUDA_SUPPRESS_HD_WARNINGS
template<class InputIterator,class UnaryPredicate>
__HOST__ __DEVICE__ InputIterator
find_if( InputIterator first, InputIterator last, UnaryPredicate p, ecuda::true_type ) // device memory
{
	#ifdef __CUDA_ARCH__
	while( first != last ) {
		if( p(*first) ) return first;
		++first;
	}
	return first;
	#else
	typedef typename ecuda::remove_const<typename ecuda::iterator_traits<InputIterator>::value_type>::type value_type;
	std::vector< value_type, host_allocator<value_type> > v( ecuda::distance(first,last) );
	ecuda::copy( first, last, v.begin() );
	const typename ecuda::iterator_traits<InputIterator>::difference_type index = std::distance( v.begin(), std::find_if( v.begin(), v.end(), p ) );
	ecuda::advance( first, index );
	return first;
	#endif
}

ECUDA_SUPPRESS_HD_WARNINGS
template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ InputIterator
find_if( InputIterator first, InputIterator last, UnaryPredicate p, ecuda::false_type ) // host memory
{
	#ifdef __CUDA_ARCH__
	return last; // never called on device code
	#else
	return std::find_if( first, last, p );
	#endif
}

} // namespace impl
/// \endcond

ECUDA_SUPPRESS_HD_WARNINGS
template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ InputIterator
find_if( InputIterator first, InputIterator last, UnaryPredicate p )
{
	return impl::find_if( first, last, p, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

} // namespace ecuda

#endif
