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
//
// CUDA implementations from STL header <numeric>.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_NUMERIC_HPP
#define ECUDA_NUMERIC_HPP

#include <numeric>

#include "global.hpp"
#include "iterator.hpp"    // for ecuda::iterator_traits
#include "type_traits.hpp" // for ecuda::is_same

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class InputIterator,typename T,class BinaryOperation>
__HOST__ __DEVICE__ inline T accumulate( InputIterator first, InputIterator last, T init, BinaryOperation op, ecuda::true_type )
{
	// is an iterator to device memory
	#ifdef __CUDA_ARCH__
	while( first != last ) { init = op(init,*first); ++first; }
	return init;
	#else
	const bool isIteratorContiguous = ecuda::is_same<typename ecuda::iterator_traits<InputIterator>::is_contiguous,ecuda::true_type>::value;
	ECUDA_STATIC_ASSERT(isIteratorContiguous,CANNOT_ACCUMULATE_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_MEMORY);
	typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	std::vector<typename ecuda::iterator_traits<InputIterator>::value_type> hostVector( n );
	ecuda::copy( first, last, hostVector.begin() );
	return std::accumulate( hostVector.begin(), hostVector.end(), init, op );
	#endif
}

template<class InputIterator,typename T>
__HOST__ __DEVICE__ inline T accumulate( InputIterator first, InputIterator last, T init, ecuda::true_type )
{
	// is an iterator to device memory
	#ifdef __CUDA_ARCH__
	while( first != last ) { init += *first; ++first; }
	return init;
	#else
	const bool isIteratorContiguous = ecuda::is_same<typename ecuda::iterator_traits<InputIterator>::is_contiguous,ecuda::true_type>::value;
	ECUDA_STATIC_ASSERT(isIteratorContiguous,CANNOT_ACCUMULATE_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_MEMORY);
	typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	std::vector<typename ecuda::iterator_traits<InputIterator>::value_type> hostVector( n );
	ecuda::copy( first, last, hostVector.begin() );
	return std::accumulate( hostVector.begin(), hostVector.end(), init );
	#endif
}

template<class InputIterator,typename T,class BinaryOperation>
__HOST__ __DEVICE__ inline T accumulate( InputIterator first, InputIterator last, T init, BinaryOperation op, ecuda::false_type )
{
	// not an iterator to device memory, delegate to STL
	#ifdef __CUDA_ARCH__
	return last; // can never be called from device code, dummy return to satisfy nvcc
	#else
	return std::accumulate( first, last, init, op );
	#endif
}

template<class InputIterator,typename T>
__HOST__ __DEVICE__ inline T accumulate( InputIterator first, InputIterator last, T init, ecuda::false_type )
{
	// not an iterator to device memory, delegate to STL
	#ifdef __CUDA_ARCH__
	return last; // can never be called from device code, dummy return to satisfy nvcc
	#else
	return std::accumulate( first, last, init );
	#endif
}

} // namespace impl
/// \endcond

///
/// \brief Computes the sum of a sequence of elements.
///
/// Computes the sum of the given value init and the elements in the range [first,last).
///
/// \param first,last the range of elements to sum
/// \param init initial value of the sum
/// \returns the sum of the given value and elements in the given range
///
template<class InputIterator,typename T>
__HOST__ __DEVICE__ inline T accumulate( InputIterator first, InputIterator last, T init )
{
	return impl::accumulate( first, last, init, ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

///
/// \brief Computes the sum of a sequence of elements.
///
/// Computes the sum of the given value init and the elements in the range [first,last).
/// The sum is calculated using the binary operation function op that should have a signature
/// equivalent to:
/// \code{.cpp}
/// Ret fun( const Type1& a, const Type2& b );
/// \endcode
///
/// \param first,last the range of elements to sum
/// \param init initial value of the sum
/// \param op binary operation function object that will be applied
/// \returns the sum of the given value and elements in the given range
///
template<class InputIterator,typename T,class BinaryOperation>
__HOST__ __DEVICE__ inline T accumulate( InputIterator first, InputIterator last, T init, BinaryOperation op )
{
	return impl::accumulate( first, last, init, op, ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}

} // namespace ecuda

#endif
