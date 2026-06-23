//----------------------------------------------------------------------------
// algo/for_each.hpp
//
// Extension of std::for_each that recognizes device memory and can be called from
// host or device code.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ALGO_FOR_EACH_HPP
#define ECUDA_ALGO_FOR_EACH_HPP

#include <algorithm>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class InputIterator,class UnaryFunction>
inline __HOST__ __DEVICE__ UnaryFunction
for_each( InputIterator first, InputIterator last, UnaryFunction f, ecuda::false_type ) // host memory
{
	#ifdef __CUDA_ARCH__
	return f; // never called from device code
	#else
	// just defer to STL
	return std::for_each( first, last, f );
	#endif
}

template<class InputIterator,class UnaryFunction>
__HOST__ __DEVICE__ UnaryFunction
for_each( InputIterator first, InputIterator last, UnaryFunction f, ecuda::true_type ) // device memory
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { f(*first); ++first; }
	//return f; // never called from device code
	#else
	typedef typename ecuda::remove_const<typename ecuda::iterator_traits<InputIterator>::value_type>::type value_type;
	std::vector< value_type, host_allocator<value_type> > v( ecuda::distance(first,last) );
	ecuda::copy( first, last, v.begin() );
	std::for_each( v.begin(), v.end(), f );
	ecuda::copy( v.begin(), v.end(), first );
	#endif
	return f;
}

} // namespace impl
/// \endcond

ECUDA_SUPPRESS_HD_WARNINGS
template<class InputIterator,class UnaryFunction>
inline __HOST__ __DEVICE__ UnaryFunction
for_each( InputIterator first, InputIterator last, UnaryFunction f )
{
	return impl::for_each( first, last, f, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

} // namespace ecuda

#endif
