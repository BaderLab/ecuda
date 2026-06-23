//----------------------------------------------------------------------------
// algo/find_if_not.hpp
//
// Extension of std::find_if_not that recognizes device memory and can be called
// from host or device code.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ALGO_FIND_IF_NOT_HPP
#define ECUDA_ALGO_FIND_IF_NOT_HPP

#ifdef ECUDA_CPP11_AVAILABLE

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
find_if_not( InputIterator first, InputIterator last, UnaryPredicate p, ecuda::true_type ) // device memory
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
	const typename ecuda::iterator_traits<InputIterator>::difference_type index = std::distance( v.begin(), std::find_if_not( v.begin(), v.end(), p ) );
	ecuda::advance( first, index );
	return first;
	#endif
}

ECUDA_SUPPRESS_HD_WARNINGS
template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ InputIterator
find_if_not( InputIterator first, InputIterator last, UnaryPredicate p, ecuda::false_type ) // host memory
{
	#ifdef __CUDA_ARCH__
	return last; // never called from device code
	#else
	return std::find_if_not( first, last, p );
	#endif
}

} // namespace impl
/// \endcond

ECUDA_SUPPRESS_HD_WARNINGS
template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ InputIterator
find_if_not( InputIterator first, InputIterator last, UnaryPredicate p )
{
	return impl::find_if_not( first, last, p, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

ECUDA_SUPPRESS_HD_WARNINGS
template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ bool
all_of( InputIterator first, InputIterator last, UnaryPredicate p )
{
	return ecuda::find_if_not( first, last, p ) == last;
}

} // namespace ecuda

#endif // ECUDA_CPP11_AVAILABLE

#endif
