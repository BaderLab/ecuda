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
// algo/equal.hpp
//
// Extension of std::equal that recognizes device memory and can be called from
// host or device code.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------
#pragma once
#ifndef ECUDA_ALGO_EQUAL_HPP
#define ECUDA_ALGO_EQUAL_HPP

#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"
#include "../utility.hpp"

namespace ecuda {

#ifdef __CUDA_ARCH__

// forward declaration
template<class InputIterator1,class InputIterator2> __DEVICE__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2 );

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class InputIterator1,typename T,typename P>
__DEVICE__ inline bool equal(
	InputIterator1 first1, InputIterator1 last1,
	device_contiguous_block_iterator<T,P> first2,
	ecuda::pair<ecuda::true_type,ecuda::true_type>
)
{
	while( first1 != last1 ) {
		typename device_contiguous_block_iterator<T,P>::contiguous_iterator blockBegin = first2.contiguous_begin();
		typename device_contiguous_block_iterator<T,P>::contiguous_iterator blockEnd = first2.contiguous_end();
		if( !equal( blockBegin, blockEnd, first1 ) ) return false;
		first1 += distance( blockBegin, blockEnd );
	}
	return true;
}

template<class InputIterator1,class InputIterator2>
__DEVICE__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<ecuda::true_type,ecuda::true_type> )
{
	for( ; first1 != last1; ++first1, ++first2 ) if( !(*first1 == *first2) ) return false;
	return true;
}

template<class InputIterator1,class InputIterator2,class Type1,class Type2>
__DEVICE__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<Type1,Type2> )
{
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_EQUAL_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return false; // never actually gets called, just here to trick nvcc
}

} // namespace impl
/// \endcond

///
/// \brief Replacement for std::equal.
///
/// ecuda::equal is identical to std::equal, but can be a) called from device code, and b) supports
/// device memory when called from host code.
///
/// Compile-time checks are performed to determine which action should be taken. If called from
/// device code, then it must be true that both ranges refer to device memory (otherwise nvcc will
/// fail before evaluating the ecuda::equal call) and the comparison between ranges is done on-device.
/// If the called from host code and both ranges refer to host memory, the evaluation is delegated
/// to std::equal. If called from host code, and one or both ranges refer to device memory, the
/// range(s) are copied to temporary host memory before delegating to std::equal.
///
/// \returns true if the range [first1,last1) is equal to the range [first2,first2+(last1-first1)),
/// and false otherwise.
///
template<class InputIterator1,class InputIterator2>
__DEVICE__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2 )
{
	return impl::equal( first1, last1, first2, ecuda::pair<typename ecuda::iterator_traits<InputIterator1>::is_device_iterator,typename ecuda::iterator_traits<InputIterator2>::is_device_iterator>() );
}

#else // __CUDA_ARCH__ not defined

// forward declaration
template<class InputIterator1,class InputIterator2> __HOST__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2 );

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class InputIterator1,class InputIterator2>
__HOST__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<ecuda::false_type,ecuda::false_type> ) {
	return std::equal( first1, last1, first2 );
}

template<class InputIterator1,class InputIterator2>
__HOST__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<ecuda::true_type,ecuda::false_type> ) {
	typedef typename ecuda::remove_const<typename ecuda::iterator_traits<InputIterator1>::value_type>::type valtype1;
	std::vector< valtype1, host_allocator<valtype1> > v1( static_cast<std::size_t>(ecuda::distance(first1,last1)) );
	ecuda::copy( first1, last1, v1.begin() );
	return std::equal( v1.begin(), v1.end(), first2 );
}

template<class InputIterator1,class InputIterator2>
__HOST__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<ecuda::false_type,ecuda::true_type> ) {
	InputIterator2 last2 = first2;
	ecuda::advance( last2, ecuda::distance(first1,last1) );
	return ecuda::equal( first2, last2, first1 );
}

template<class InputIterator1,typename T,typename P>
__HOST__ inline bool equal(
	InputIterator1 first1, InputIterator1 last1,
	device_contiguous_block_iterator<T,P> first2,
	ecuda::pair<ecuda::true_type,ecuda::true_type>
)
{
	while( first1 != last1 ) {
		typename device_contiguous_block_iterator<T,P>::contiguous_iterator blockBegin = first2.contiguous_begin();
		typename device_contiguous_block_iterator<T,P>::contiguous_iterator blockEnd = first2.contiguous_end();
		if( !equal( blockBegin, blockEnd, first1 ) ) return false;
		first1 += distance( blockBegin, blockEnd );
	}
	return true;
}

template<class InputIterator1,class InputIterator2>
__HOST__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<ecuda::true_type,ecuda::true_type> ) {
	// strip const qualifiers otherwise cannot create std::vector<const T>
	typedef typename ecuda::remove_const<typename ecuda::iterator_traits<InputIterator1>::value_type>::type valtype1;
	typedef typename ecuda::remove_const<typename ecuda::iterator_traits<InputIterator2>::value_type>::type valtype2;
	InputIterator2 last2 = first2;
	ecuda::advance( last2, ecuda::distance(first1,last1) );
	// allocate temporary memory using host_allocator (i.e. cudaMallocHost) for potential performance improvement
	std::vector< valtype1, host_allocator<valtype1> > v1( ecuda::distance( first1, last1 ) );
	std::vector< valtype2, host_allocator<valtype2> > v2( ecuda::distance( first2, last2 ) );
	ecuda::copy( first1, last1, v1.begin() );
	ecuda::copy( first2, last2, v2.begin() );
	return std::equal( v1.begin(), v1.end(), v2.begin() );
}

} // namespace impl
/// \endcond

///
/// \brief Replacement for std::equal.
///
/// ecuda::equal is identical to std::equal, but can be a) called from device code, and b) supports
/// device memory when called from host code.
///
/// Compile-time checks are performed to determine which action should be taken. If called from
/// device code, then it must be true that both ranges refer to device memory (otherwise nvcc will
/// fail before evaluating the ecuda::equal call) and the comparison between ranges is done on-device.
/// If the called from host code and both ranges refer to host memory, the evaluation is delegated
/// to std::equal. If called from host code, and one or both ranges refer to device memory, the
/// range(s) are copied to temporary host memory before delegating to std::equal.
///
/// \returns true if the range [first1,last1) is equal to the range [first2,first2+(last1-first1)),
/// and false otherwise.
///
template<class InputIterator1,class InputIterator2>
__HOST__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2 )
{
	return impl::equal( first1, last1, first2, ecuda::pair<typename ecuda::iterator_traits<InputIterator1>::is_device_iterator,typename ecuda::iterator_traits<InputIterator2>::is_device_iterator>() );
}

#endif // __CUDA_ARCH__

} // namespace ecuda

#endif
