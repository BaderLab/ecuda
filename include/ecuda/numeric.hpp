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
// numeric.hpp
// CUDA implementations from STL header <numeric>.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_NUMERIC_HPP
#define ECUDA_NUMERIC_HPP

#include <numeric>

#include "global.hpp"
#include "iterator.hpp"
#include "type_traits.hpp"

namespace ecuda {

namespace impl {

template<class InputIterator,typename T,class BinaryOperation>
__HOST__ __DEVICE__ inline T accumulate( InputIterator first, InputIterator last, T init, BinaryOperation op, std::true_type )
{
	// is an iterator to device memory
	#ifdef __CUDA_ARCH__
	while( first != last ) { init = op(init,*first); ++first; }
	return init;
	#else
	const bool isIteratorContiguous = std::is_same<typename ecuda::iterator_traits<InputIterator>::is_contiguous,std::true_type>::value;
	ECUDA_STATIC_ASSERT(isIteratorContiguous,CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_WITH_ACCUMULATE);
	typename ecuda::iterator_traits<ForwardIterator>::difference_type n = ecuda::distance( first, last );
	std::vector<typename ecuda::iterator_traits<InputIterator>::value_type> hostVector( n );
	ecuda::copy( first, last, hostVector.begin() );
	return std::accumulate( hostVector.begin(), hostVector.end(), init, op );
	#endif
}

template<class InputIterator,typename T>
__HOST__ __DEVICE__ inline T accumulate( InputIterator first, InputIterator last, T init, std::true_type )
{
	// is an iterator to device memory
	#ifdef __CUDA_ARCH__
	while( first != last ) { init += *first; ++first; }
	return init;
	#else
	const bool isIteratorContiguous = std::is_same<typename ecuda::iterator_traits<InputIterator>::is_contiguous,std::true_type>::value;
	ECUDA_STATIC_ASSERT(isIteratorContiguous,CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_WITH_ACCUMULATE);
	typename ecuda::iterator_traits<ForwardIterator>::difference_type n = ecuda::distance( first, last );
	std::vector<typename ecuda::iterator_traits<InputIterator>::value_type> hostVector( n );
	ecuda::copy( first, last, hostVector.begin() );
	return std::accumulate( hostVector.begin(), hostVector.end(), init, op );
	#endif
}

template<class InputIterator,typename T,class BinaryOperation>
__HOST__ __DEVICE__ inline T accumulate( InputIterator first, InputIterator last, T init, BinaryOperation op, std::false_type )
{
	// not an iterator to device memory, delegate to STL
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_ACCUMULATE_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return last; // can never be called from device code, dummy return to satisfy nvcc
	#else
	return std::accumulate( first, last, init, op );
	#endif
}

template<class InputIterator,typename T>
__HOST__ __DEVICE__ inline T accumulate( InputIterator first, InputIterator last, T init, std::false_type )
{
	// not an iterator to device memory, delegate to STL
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_ACCUMULATE_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return last; // can never be called from device code, dummy return to satisfy nvcc
	#else
	return std::accumulate( first, last, init );
	#endif
}


} // namespace impl

template<class InputIterator,typename T>
__HOST__ __DEVICE__ inline T accumulate( InputIterator first, InputIterator last, T init )
{
	return impl::accumulate( first, last, init, ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

template<class InputIterator,typename T,class BinaryOperation>
__HOST__ __DEVICE__ inline T accumulate( InputIterator first, InputIterator last, T init, BinaryOperation op )
{
	return impl::accumulate( first, last, init, op, ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

} // namespace ecuda

#endif
