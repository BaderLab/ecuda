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
// algo/reverse.hpp
//
// Extension of std::reverse that recognizes device memory and can be called
// from host or device code.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------
#pragma once
#ifndef ECUDA_ALGO_REVERSE_HPP
#define ECUDA_ALGO_REVERSE_HPP

#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"
//#include "../utility.hpp"
#include "../algorithm.hpp"

namespace ecuda {

// forward declaration
template<class BidirectionalIterator> __HOST__ __DEVICE__ inline void reverse( BidirectionalIterator first, BidirectionalIterator last );

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class ForwardIterator>
__HOST__ __DEVICE__ inline void
reverse( ForwardIterator first, ForwardIterator last,
		 ecuda::false_type // host memory
)
{
	#ifdef __CUDA_ARCH__
	//ECUDA_STATIC_ASSERT(false,CANNOT_CALL_REVERSE_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	#else
	std::reverse( first, last );
	#endif
}

template<class ForwardIterator>
__HOST__ __DEVICE__ inline void
reverse( ForwardIterator first, ForwardIterator last,
		 ecuda::true_type // device memory
)
{
	#ifdef __CUDA_ARCH__
	while( (first!=last) and (first!=--last) ) {
		ecuda::swap( *first, *last );
		++first;
	}
	#else
	{
		typedef typename ecuda::iterator_traits<ForwardIterator>::iterator_category iterator_category;
		typedef typename ecuda::iterator_traits<ForwardIterator>::is_contiguous iterator_contiguity;
		const bool isSomeKindOfContiguous =
			ecuda::is_same<iterator_contiguity,ecuda::true_type>::value ||
			ecuda::is_same<iterator_category,device_contiguous_block_iterator_tag>::value;
		ECUDA_STATIC_ASSERT(isSomeKindOfContiguous,CANNOT_CALL_REVERSE_ON_NONCONTIGUOUS_DEVICE_MEMORY);
	}
	typedef typename ecuda::remove_const<typename ecuda::iterator_traits<ForwardIterator>::value_type>::type value_type;
	std::vector< value_type, host_allocator<value_type> > v( ::ecuda::distance( first, last ) );
	::ecuda::copy( first, last, v.begin() );
	::ecuda::reverse( v.begin(), v.end() );
	::ecuda::copy( v.begin(), v.end(), first );
	#endif
}

} // namespace impl
/// \endcond

#pragma hd_warning_disable
template<class BidirectionalIterator>
__HOST__ __DEVICE__ inline void reverse( BidirectionalIterator first, BidirectionalIterator last ) {
	impl::reverse(
		first, last,
		typename ecuda::iterator_traits<BidirectionalIterator>::is_device_iterator()
	);
}

} // namespace ecuda

#endif
