/*
Copyright (c) 2015, Scott Zuyderduyn
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
// algorithm.hpp
//
// CUDA implementations from STL header <numeric>.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#ifndef ECUDA_ALGORITHM_HPP
#define ECUDA_ALGORITHM_HPP

#include <iterator>
#include <vector>

#include "global.hpp"
#include "type_traits.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace detail {

typedef std::false_type host_type;
typedef std::true_type  device_type;

//typedef std::false_type non_contiguous_type;
//typedef std::true_type  contiguous_type;

} // namespace detail
/// \endcond

template<typename T>               __HOST__ __DEVICE__ inline const T& min( const T& a, const T& b ) { return b < a ? b : a; }
template<typename T,class Compare> __HOST__ __DEVICE__ inline const T& min( const T& a, const T& b, Compare cmp ) { return cmp(b,a) ? b : a; }

template<typename T>               __HOST__ __DEVICE__ inline const T& max( const T& a, const T& b ) { return b > a ? b : a; }
template<typename T,class Compare> __HOST__ __DEVICE__ inline const T& max( const T& a, const T& b, Compare cmp ) { return cmp(a,b) ? b : a; }

} // namespace ecuda

#include "algo/copy.hpp" // equivalent to std::copy
#include "algo/equal.hpp" // equivalent to std::equal
#include "algo/fill.hpp" // equivalent to std::fill
#include "algo/lexicographical_compare.hpp" // equivalent to std::lexicographical_compare
#include "algo/max_element.hpp" // equivalent to std::max_element
// NOTE: there are additional includes at the end of the file;
//       some algos require functions decleared in this file

#include "iterator.hpp"
#include <algorithm>
#include <utility>

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class InputIterator,typename T>
__HOST__ __DEVICE__ InputIterator
find( InputIterator first, InputIterator last, const T& value, std::true_type ) // device memory
{
	#ifdef __CUDA_ARCH__
	while( first != last ) {
		if( *first == value ) return first;
		++first;
	}
	return first;
	#else
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator>::value_type>::type value_type;
	std::vector< value_type, host_allocator<value_type> > v( ecuda::distance(first,last) );
	ecuda::copy( first, last, v.begin() );
	const typename ecuda::iterator_traits<InputIterator>::difference_type index = std::distance( v.begin(), std::find( v.begin(), v.end(), value ) );
	ecuda::advance( first, index );
	return first;
	#endif
}

template<class InputIterator,typename T>
inline __HOST__ __DEVICE__ InputIterator
find( InputIterator first, InputIterator last, const T& value, std::false_type ) // host memory
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_FIND_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return last;
	#else
	return std::find( first, last, value );
	#endif
}

} // namespace impl

template<class InputIterator,typename T>
inline __HOST__ __DEVICE__ InputIterator
find( InputIterator first, InputIterator last, const T& value )
{
	return impl::find( first, last, value, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

namespace impl {

template<class InputIterator,class UnaryPredicate>
__HOST__ __DEVICE__ InputIterator
find_if( InputIterator first, InputIterator last, UnaryPredicate p, std::true_type ) // device memory
{
	#ifdef __CUDA_ARCH__
	while( first != last ) {
		if( p(*first) ) return first;
		++first;
	}
	return first;
	#else
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator>::value_type>::type value_type;
	std::vector< value_type, host_allocator<value_type> > v( ecuda::distance(first,last) );
	ecuda::copy( first, last, v.begin() );
	const typename ecuda::iterator_traits<InputIterator>::difference_type index = std::distance( v.begin(), std::find_if( v.begin(), v.end(), p ) );
	ecuda::advance( first, index );
	return first;
	#endif
}

template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ InputIterator
find_if( InputIterator first, InputIterator last, UnaryPredicate p, std::false_type ) // host memory
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_FIND_IF_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return last;
	#else
	return std::find_if( first, last, p );
	#endif
}

} // namespace impl
/// \endcond

template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ InputIterator
find_if( InputIterator first, InputIterator last, UnaryPredicate p )
{
	return impl::find_if( first, last, p, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

#ifdef __CPP11_SUPPORTED__
/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class InputIterator,class UnaryPredicate>
__HOST__ __DEVICE__ InputIterator
find_if_not( InputIterator first, InputIterator last, UnaryPredicate p, std::true_type ) // device memory
{
	#ifdef __CUDA_ARCH__
	while( first != last ) {
		if( p(*first) ) return first;
		++first;
	}
	return first;
	#else
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator>::value_type>::type value_type;
	std::vector< value_type, host_allocator<value_type> > v( ecuda::distance(first,last) );
	ecuda::copy( first, last, v.begin() );
	const typename ecuda::iterator_traits<InputIterator>::difference_type index = std::distance( v.begin(), std::find_if_not( v.begin(), v.end(), p ) );
	ecuda::advance( first, index );
	return first;
	#endif
}

template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ InputIterator
find_if_not( InputIterator first, InputIterator last, UnaryPredicate p, std::false_type ) // host memory
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_FIND_IF_NOT_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return last;
	#else
	return std::find_if_not( first, last, p );
	#endif
}

} // namespace impl
/// \endcond

template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ InputIterator
find_if_not( InputIterator first, InputIterator last, UnaryPredicate p )
{
	return impl::find_if_not( first, last, p, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ bool
all_of( InputIterator first, InputIterator last, UnaryPredicate p )
{
	return ecuda::find_if_not( first, last, p ) == last;
}

#endif // __CPP11_SUPPORTED__

template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ bool
any_of( InputIterator first, InputIterator last, UnaryPredicate p )
{
	return ecuda::find_if( first, last, p ) != last;
}

template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ bool
none_of( InputIterator first, InputIterator last, UnaryPredicate p )
{
	return ecuda::find_if( first, last, p ) == last;
}


template<typename T> __HOST__ __DEVICE__ inline void swap( T& a, T& b ) __NOEXCEPT__ { T tmp = a; a = b; b = tmp; } // equivalent to std::swap

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class InputIterator,class UnaryFunction>
inline __HOST__ __DEVICE__ UnaryFunction
for_each( InputIterator first, InputIterator last, UnaryFunction f, std::false_type ) // host memory
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_FOR_EACH_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return f; // never called from device code
	#else
	// just defer to STL
	return std::for_each( first, last, f );
	#endif
}

template<class InputIterator,class UnaryFunction>
__HOST__ __DEVICE__ UnaryFunction
for_each( InputIterator first, InputIterator last, UnaryFunction f, std::true_type ) // device memory
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { f(*first); ++first; }
	return f; // never called from device code
	#else
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator>::value_type>::type value_type;
	std::vector< value_type, host_allocator<value_type> > v( ecuda::distance(first,last) );
	ecuda::copy( first, last, v.begin() );
	std::for_each( v.begin(), v.end(), f );
	ecuda::copy( v.begin(), v.end(), first );
	#endif
	return f;
}

} // namespace impl
/// \endcond

template<class InputIterator,class UnaryFunction>
inline __HOST__ __DEVICE__ UnaryFunction
for_each( InputIterator first, InputIterator last, UnaryFunction f )
{
	return impl::for_each( first, last, f, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class InputIterator,typename T>
inline __HOST__ __DEVICE__ typename ecuda::iterator_traits<InputIterator>::difference_type
count( InputIterator first, InputIterator last, const T& value, std::false_type ) // host memory
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COUNT_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return 0; // never called from device code
	#else
	// just defer to STL
	return std::count( first, last, value );
	#endif
}

template<class InputIterator,typename T>
__HOST__ __DEVICE__ typename ecuda::iterator_traits<InputIterator>::difference_type
count( InputIterator first, InputIterator last, const T& value, std::true_type ) // device memory
{
	#ifdef __CUDA_ARCH__
	typename ecuda::iterator_traits<InputIterator>::difference_type n = 0;
	while( first != last ) {
		if( *first == value ) ++n;
		++first;
	}
	return n;
	#else
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator>::value_type>::type value_type;
	std::vector< value_type, host_allocator<value_type> > v( ecuda::distance(first,last) );
	ecuda::copy( first, last, v.begin() );
	return std::count( v.begin(), v.end(), value );
	#endif
}

} // namespace impl
/// \endcond

template<class InputIterator,typename T>
__HOST__ __DEVICE__ inline typename ecuda::iterator_traits<InputIterator>::difference_type
count( InputIterator first, InputIterator last, const T& value )
{
	return impl::count( first, last, value, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ typename ecuda::iterator_traits<InputIterator>::difference_type
count_if( InputIterator first, InputIterator last, UnaryPredicate p, std::false_type ) // host memory
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COUNT_IF_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return 0; // never called from device code
	#else
	// just defer to STL
	return std::count_if( first, last, p );
	#endif
}

template<class InputIterator,class UnaryPredicate>
__HOST__ __DEVICE__ typename ecuda::iterator_traits<InputIterator>::difference_type
count_if( InputIterator first, InputIterator last, UnaryPredicate p, std::true_type ) // device memory
{
	#ifdef __CUDA_ARCH__
	typename ecuda::iterator_traits<InputIterator>::difference_type n = 0;
	while( first != last ) {
		if( p(*first) ) ++n;
		++first;
	}
	return n;
	#else
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator>::value_type>::type value_type;
	std::vector< value_type, host_allocator<value_type> > v( ecuda::distance(first,last) );
	ecuda::copy( first, last, v.begin() );
	return std::count_if( v.begin(), v.end(), p );
	#endif
}

} // namespace impl
/// \endcond

template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ typename ecuda::iterator_traits<InputIterator>::difference_type
count_if( InputIterator first, InputIterator last, UnaryPredicate p )
{
	return impl::count_if( first, last, p, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class InputIterator1,class InputIterator2>
inline __HOST__ __DEVICE__ ecuda::pair<InputIterator1,InputIterator2>
mismatch( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<std::false_type,std::false_type> ) // host/host memory
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_MISMATCH_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return ecuda::pair<InputIterator1,InputIterator2>(); // never called from device code
	#else
	// just defer to STL
	return std::mismatch( first1, last1, first2 );
	#endif
}

template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ ecuda::pair<InputIterator1,InputIterator2>
mismatch( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<std::false_type,std::true_type> ) // host/device memory
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_MISMATCH_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return ecuda::pair<InputIterator1,InputIterator2>(); // never called from device code
	#else
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator2>::value_type>::type value_type2;
	typedef std::vector< value_type2, host_allocator<value_type2> > vector_type2;
	vector_type2 v2( ecuda::distance(first1,last1) );
	ecuda::copy( first2, first2+v2.size(), v2.size() );
	std::pair<InputIterator1,typename vector_type2::iterator> p = std::mismatch( first1, last1, v2.begin() );
	ecuda::advance( first2, ecuda::distance(v2.begin(),p.second) );
	return ecuda::pair<InputIterator1,InputIterator2>( p.first, first2 );
	#endif
}

template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ ecuda::pair<InputIterator1,InputIterator2>
mismatch( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<std::true_type,std::false_type> ) // device/host memory
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_MISMATCH_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return ecuda::pair<InputIterator1,InputIterator2>(); // never called from device code
	#else
	InputIterator2 last2 = first2;
	ecuda::advance( last2, ecuda::distance(first1,last1) );
	ecuda::pair<InputIterator2,InputIterator1> p = mismatch( first2, last2, first1 );
	return ecuda::pair<InputIterator1,InputIterator2>( p.second, p.first );
	#endif
}

template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ ecuda::pair<InputIterator1,InputIterator2>
mismatch( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<std::true_type,std::true_type> ) // device/device memory
{
	#ifdef __CUDA_ARCH__
	while( (first1 != last1) && (*first1 == *first2) ) { ++first1; ++first2; }
	return ecuda::pair<InputIterator1,InputIterator2>(first1,first2);
	#else
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator1>::value_type>::type value_type1;
	typedef std::vector< value_type1, host_allocator<value_type1> > vector_type1;
	vector_type1 v1( ecuda::distance(first1,last1) );
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator2>::value_type>::type value_type2;
	typedef std::vector< value_type2, host_allocator<value_type2> > vector_type2;
	vector_type2 v2( v1.size() );
	ecuda::copy( first1, last1, v1.begin() );
	InputIterator2 last2 = first2;
	ecuda::advance( last2, ecuda::distance(first1,last1) );
	ecuda::copy( first2, last2, v2.begin() );
	std::pair<typename vector_type1::iterator,typename vector_type2::iterator> p = std::mismatch( v1.begin(), v1.end(), v2.begin() );
	ecuda::advance( first1, ecuda::distance(v1.begin(),p.first) );
	ecuda::advance( first2, ecuda::distance(v2.begin(),p.second) );
	return ecuda::pair<InputIterator1,InputIterator2>( first1, first2 );
	#endif
}

} // namespace impl
/// \endcond

template<class InputIterator1,class InputIterator2>
inline __HOST__ __DEVICE__ ecuda::pair<InputIterator1,InputIterator2>
mismatch( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2 )
{
	return impl::mismatch(
					first1, last1,
					first2,
					ecuda::pair<typename ecuda::iterator_traits<InputIterator1>::is_device_iterator(),typename ecuda::iterator_traits<InputIterator2>::is_device_iterator()>()
				);
}

} // namespace ecuda

#include "algo/reverse.hpp" // equivalent to std::reverse


#endif

