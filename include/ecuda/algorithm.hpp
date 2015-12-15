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
<<<<<<< HEAD
// Generic functions found in STL <algorithm> reimplemented so they can be
// called from device code or handle iterators to device memory depending
// on the purpose of the function.
//
// TODO: make ecuda::copy handle case where input range and output have
//       different value types
=======
// CUDA implementations from STL header <algorithm>.
>>>>>>> ecuda2/master
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

<<<<<<< HEAD
#pragma once
#ifndef ECUDA_ALGORITHM_HPP
#define ECUDA_ALGORITHM_HPP

#include <vector>

#include "global.hpp"
#include "allocators.hpp"
#include "apiwrappers.hpp"
#include "iterators.hpp"

namespace ecuda {

///
/// \brief Exchanges the given values.
///
/// \param t1,t2 the values to be swapped
///
template<typename T>
DEVICE inline void swap( T& t1, T& t2 ) {
	T tmp = t1;
	t1 = t2;
	t2 = tmp;
}

///
/// \brief Checks if the first range [begin1,end1) is lexicographically less than the second range [begin2,end2).
///
/// \returns true if the first range is lexicographically less than the second.
///
template<class InputIterator1,class InputIterator2>
DEVICE bool lexicographical_compare( InputIterator1 begin1, InputIterator1 end1, InputIterator2 begin2, InputIterator2 end2 ) {
	for( ; (begin1 != end1) and (begin2 != end2); ++begin1, ++begin2 ) {
		if( *begin1 < *begin2 ) return true;
		if( *begin2 < *begin1 ) return false;
	}
	return begin1 == end1 and begin2 == end2;
}

template<typename T> DEVICE inline const T& min( const T& a, const T& b ) { return a < b ? a : b; }

template<typename T> DEVICE inline const T& max( const T& a, const T& b ) { return a < b ? b : a; }

template<class ForwardIterator>
DEVICE ForwardIterator max_element( ForwardIterator first, ForwardIterator last ) {
	ForwardIterator best = first;
	while( ++first != last ) if( first > best ) best = first;
	return best;
}

template<class ForwardIterator>
DEVICE ForwardIterator min_element( ForwardIterator first, ForwardIterator last ) {
	ForwardIterator best = first;
	while( ++first != last ) if( first < best ) best = first;
	return best;
}

template<class InputIterator,typename T>
DEVICE T accumulate( InputIterator first, InputIterator last, T init=0 ) {
	while( first != last ) { init += *first; ++first; }
	return init;
}


/// \cond DEVELOPER_DOCUMENTATION

template<class InputIterator,class OutputIterator>
HOST OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, std::random_access_iterator_tag, ::ecuda::contiguous_device_iterator_tag ) {
	typedef typename OutputIterator::value_type value_type;
	typename std::iterator_traits<InputIterator>::difference_type len = ::ecuda::distance(first,last);
	CUDA_CALL( cudaMemcpy<value_type>( result.operator->(), first.operator->(), len, cudaMemcpyHostToDevice ) );
	return result+len;
}

template<class InputIterator,class OutputIterator>
HOST OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, std::bidirectional_iterator_tag, ::ecuda::contiguous_device_iterator_tag ) {
	typedef typename OutputIterator::value_type value_type;
	std::vector< value_type, ::ecuda::host_allocator<value_type> > v( first, last );
	return __copy( v.begin(), v.end(), result, std::random_access_iterator_tag(), ::ecuda::contiguous_device_iterator_tag() );
}

template<class InputIterator,class OutputIterator>
HOST inline OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, std::forward_iterator_tag, ::ecuda::contiguous_device_iterator_tag ) {
	return __copy( first, last, std::bidirectional_iterator_tag(), ::ecuda::contiguous_device_iterator_tag() );
}

template<class InputIterator,class OutputIterator>
HOST inline OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, std::input_iterator_tag(), ::ecuda::contiguous_device_iterator_tag ) {
	return __copy( first, last, std::bidirectional_iterator_tag(), ::ecuda::contiguous_device_iterator_tag() );
}

template<class InputIterator,class OutputIterator>
HOST OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, ::ecuda::contiguous_device_iterator_tag, ::ecuda::contiguous_device_iterator_tag ) {
	typedef typename OutputIterator::value_type value_type;
	typename std::iterator_traits<InputIterator>::difference_type len = ::ecuda::distance(first,last);
	CUDA_CALL( cudaMemcpy<value_type>( result.operator->(), first.operator->(), len, cudaMemcpyDeviceToDevice ) );
	return result+len;
}

template<class InputIterator,class OutputIterator>
HOST OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, ::ecuda::contiguous_device_iterator_tag, std::random_access_iterator_tag ) {
	typedef typename OutputIterator::value_type value_type;
	typename std::iterator_traits<InputIterator>::difference_type len = ::ecuda::distance(first,last);
	CUDA_CALL( cudaMemcpy<value_type>( result.operator->(), first.operator->(), len, cudaMemcpyDeviceToHost ) );
	return result+len;
}

template<class InputIterator,class OutputIterator>
HOST OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, ::ecuda::contiguous_device_iterator_tag, std::bidirectional_iterator_tag ) {
	typedef typename OutputIterator::value_type value_type;
	typename std::iterator_traits<InputIterator>::difference_type len = ::ecuda::distance(first,last);
	std::vector< value_type, ::ecuda::host_allocator<value_type> > v( len );
	__copy( first, last, v.begin(), ::ecuda::contiguous_device_iterator_tag(), std::random_access_iterator_tag() );
	for( typename std::vector< value_type, ::ecuda::host_allocator<value_type> >::iterator iter = v.begin(); iter != v.end(); ++iter, ++result ) *result = *iter;
	return result;
}

template<class InputIterator,class OutputIterator>
HOST inline OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, ::ecuda::contiguous_device_iterator_tag, std::forward_iterator_tag ) {
	return __copy( first, last, result, ::ecuda::contiguous_device_iterator_tag(), std::bidirectional_iterator_tag() );
}

template<class InputIterator,class OutputIterator>
HOST inline OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, ::ecuda::contiguous_device_iterator_tag, std::output_iterator_tag ) {
	return __copy( first, last, result, ::ecuda::contiguous_device_iterator_tag(), std::bidirectional_iterator_tag() );
}

/// \endcond

///
/// \brief Copy range of elements.
///
/// This is analagous to the std::copy method in the STL \<algorithm\> header.  In this case, the behaviour
/// of the copy is determined at compile-time depending on whether the source and destination iterators
/// are contiguous and whether they refer to device or host memory.
///
/// \param first,last input iterators to the initial and final positions in a sequence to be copied. The
///        range used is [first,last), which contains all the elements between first and last, including
///        the element pointed by first but the not the element pointed by last.
/// \param result Output iterator to the initial position in the destination sequence. This shall not
///               pointer to any element in the range [first,last).
/// \return An iterator to the end of the destination range where elements have been copied.
///
template<class InputIterator,class OutputIterator>
HOST inline OutputIterator copy( InputIterator first, InputIterator last, OutputIterator result ) {
	return __copy( first, last, result, typename std::iterator_traits<InputIterator>::iterator_category(), typename std::iterator_traits<OutputIterator>::iterator_category() );
=======
#ifndef ECUDA_ALGORITHM_HPP
#define ECUDA_ALGORITHM_HPP

#include <iterator>
#include <vector>

#include "global.hpp"

namespace ecuda {

template<typename T>               __HOST__ __DEVICE__ inline const T& min( const T& a, const T& b )              { return b < a ? b : a; }
template<typename T,class Compare> __HOST__ __DEVICE__ inline const T& min( const T& a, const T& b, Compare cmp ) { return cmp(b,a) ? b : a; }

template<typename T>               __HOST__ __DEVICE__ inline const T& max( const T& a, const T& b )              { return b > a ? b : a; }
template<typename T,class Compare> __HOST__ __DEVICE__ inline const T& max( const T& a, const T& b, Compare cmp ) { return cmp(a,b) ? b : a; }

template<typename T> __HOST__ __DEVICE__ inline void swap( T& a, T& b ) __NOEXCEPT__ { T tmp = a; a = b; b = tmp; } // equivalent to std::swap

} // namespace ecuda

#include "iterator.hpp"
#include "type_traits.hpp"

#include "algo/copy.hpp"                    // equivalent to std::copy
#include "algo/equal.hpp"                   // equivalent to std::equal
#include "algo/fill.hpp"                    // equivalent to std::fill
#include "algo/lexicographical_compare.hpp" // equivalent to std::lexicographical_compare
#include "algo/max_element.hpp"             // equivalent to std::max_element
#include "algo/find.hpp"                    // equivalent to std::find
#include "algo/find_if.hpp"                 // equivalent to std::find_if
#include "algo/find_if_not.hpp"             // equivalent to std::find_if_not (NOTE: C++11 only)
#include "algo/for_each.hpp"                // equivalent to std::for_each
#include "algo/count.hpp"                   // equivalent to std::count
#include "algo/count_if.hpp"                // equivalent to std::count_if
#include "algo/mismatch.hpp"                // equivalent to std::mismatch
#include "algo/reverse.hpp"                 // equivalent to std::reverse

namespace ecuda {

ECUDA_SUPPRESS_HD_WARNINGS
template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__
bool any_of( InputIterator first, InputIterator last, UnaryPredicate p )
{
	return ecuda::find_if( first, last, p ) != last;
}

ECUDA_SUPPRESS_HD_WARNINGS
template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__
bool none_of( InputIterator first, InputIterator last, UnaryPredicate p )
{
	return ecuda::find_if( first, last, p ) == last;
>>>>>>> ecuda2/master
}

} // namespace ecuda

#endif
<<<<<<< HEAD
=======

>>>>>>> ecuda2/master
