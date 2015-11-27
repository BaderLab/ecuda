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
// algo/fill.hpp
//
// Extension of std::fill that recognizes device memory and can be called from
// host or device code.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------
#pragma once
#ifndef ECUDA_ALGO_FILL_HPP
#define ECUDA_ALGO_FILL_HPP

#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"
//#include "../utility.hpp"

namespace ecuda {

// forward declaration
template<class ForwardIterator,typename T> __HOST__ __DEVICE__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val );

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

namespace fill_device {

template<class ForwardIterator,typename T>
__HOST__ __DEVICE__ inline void fill(
	ForwardIterator first, ForwardIterator last,
	const T& val
)
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { *first = val; ++first; }
	#else
	typedef typename ecuda::iterator_traits<ForwardIterator>::is_contiguous iterator_contiguity;
	{
		const bool isContiguous = ecuda::is_same<iterator_contiguity,ecuda::true_type>::value;
		ECUDA_STATIC_ASSERT(isContiguous,CANNOT_FILL_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_ITERATOR);
	}
	typename ecuda::iterator_traits<ForwardIterator>::difference_type n = ecuda::distance( first, last );
	typedef typename ecuda::iterator_traits<ForwardIterator>::value_type value_type;
	CUDA_CALL( cudaMemset<value_type>( first.operator->(), val, static_cast<std::size_t>(n) ) );
	#endif
}

template<typename T,typename P>
__HOST__ __DEVICE__ inline void fill(
	device_contiguous_block_iterator<T,P> first, device_contiguous_block_iterator<T,P> last,
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

} // namespace fill_device

template<class ForwardIterator,typename T>
__HOST__ __DEVICE__ inline void fill(
	ForwardIterator first, ForwardIterator last,
	const T& val,
	ecuda::true_type // device memory
)
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { *first = val; ++first; }
	#else
	typedef typename ecuda::iterator_traits<ForwardIterator>::is_contiguous iterator_contiguity;
	typedef typename ecuda::iterator_traits<ForwardIterator>::iterator_category iterator_category;
	{
		const bool isSomeKindOfContiguous =
			ecuda::is_same<iterator_contiguity,ecuda::true_type>::value ||
			ecuda::is_same<iterator_category,device_contiguous_block_iterator_tag>::value;
		ECUDA_STATIC_ASSERT(isSomeKindOfContiguous,CANNOT_FILL_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_ITERATOR);
	}
	if( ecuda::is_same<typename ecuda::iterator_traits<ForwardIterator>::value_type,T>::value ) {
		fill_device::fill( first, last, val );
	} else {
		typedef typename ecuda::iterator_traits<ForwardIterator>::value_type value_type;
		const value_type val2( val );
		fill_device::fill( first, last, val2 );
	}
	#endif
}

template<class ForwardIterator,typename T>
__HOST__ __DEVICE__ inline void fill(
	ForwardIterator first, ForwardIterator last,
	const T& val,
	ecuda::false_type // host memory
)
{
	#ifdef __CUDA_ARCH__
	//ECUDA_STATIC_ASSERT( false, CANNOT_CALL_FILL_ON_HOST_MEMORY_INSIDE_DEVICE_CODE );
	#else
	std::fill( first, last, val );
	#endif
}

} // namespace impl
/// \endcond

#pragma hd_warning_disable
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
