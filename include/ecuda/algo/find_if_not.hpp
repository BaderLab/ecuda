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

#ifdef __CPP11_SUPPORTED__

#include <algorithm>
#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

#pragma hd_warning_disable
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

#pragma hd_warning_disable
template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ InputIterator
find_if_not( InputIterator first, InputIterator last, UnaryPredicate p, ecuda::false_type ) // host memory
{
	#ifdef __CUDA_ARCH__
	//ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_FIND_IF_NOT_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return last;
	#else
	return std::find_if_not( first, last, p );
	#endif
}

} // namespace impl
/// \endcond

#pragma hd_warning_disable
template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ InputIterator
find_if_not( InputIterator first, InputIterator last, UnaryPredicate p )
{
	return impl::find_if_not( first, last, p, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

#pragma hd_warning_disable
template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ bool
all_of( InputIterator first, InputIterator last, UnaryPredicate p )
{
	return ecuda::find_if_not( first, last, p ) == last;
}

} // namespace ecuda

#endif // __CPP11_SUPPORTED__

#endif
