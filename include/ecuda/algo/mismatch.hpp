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
// algo/mismatch.hpp
//
// Extension of std::mismatch that recognizes device memory and can be called from
// host or device code.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ALGO_MISMATCH_HPP
#define ECUDA_ALGO_MISMATCH_HPP

#include <algorithm>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"
#include "../utility.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class InputIterator1,class InputIterator2>
inline __HOST__ __DEVICE__ ecuda::pair<InputIterator1,InputIterator2>
mismatch( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<ecuda::false_type,ecuda::false_type> ) // host/host memory
{
	#ifdef __CUDA_ARCH__
	//ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_MISMATCH_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return ecuda::pair<InputIterator1,InputIterator2>(); // never called from device code
	#else
	// just defer to STL
	std::pair<InputIterator1,InputIterator2> p = std::mismatch( first1, last1, first2 );
	return ecuda::pair<InputIterator1,InputIterator2>(p.first,p.second);
	#endif
}

#pragma hd_warning_disable
template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ ecuda::pair<InputIterator1,InputIterator2>
mismatch( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<ecuda::false_type,ecuda::true_type> ) // host/device memory
{
	#ifdef __CUDA_ARCH__
	//ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_MISMATCH_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return ecuda::pair<InputIterator1,InputIterator2>(); // never called from device code
	#else
	typedef typename ecuda::remove_const<typename ecuda::iterator_traits<InputIterator2>::value_type>::type value_type2;
	typedef std::vector< value_type2, host_allocator<value_type2> > vector_type2;
	vector_type2 v2( ecuda::distance(first1,last1) );
	ecuda::copy( first2, first2+v2.size(), v2.begin() );
	std::pair<InputIterator1,typename vector_type2::iterator> p = std::mismatch( first1, last1, v2.begin() );
	ecuda::advance( first2, ecuda::distance(v2.begin(),p.second) );
	return ecuda::pair<InputIterator1,InputIterator2>( p.first, first2 );
	#endif
}

template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ ecuda::pair<InputIterator1,InputIterator2>
mismatch( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<ecuda::true_type,ecuda::false_type> ) // device/host memory
{
	#ifdef __CUDA_ARCH__
	//ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_MISMATCH_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return ecuda::pair<InputIterator1,InputIterator2>(); // never called from device code
	#else
	InputIterator2 last2 = first2;
	ecuda::advance( last2, ecuda::distance(first1,last1) );
	ecuda::pair<InputIterator2,InputIterator1> p = mismatch( first2, last2, first1, ecuda::pair<ecuda::false_type,ecuda::true_type>() );
	return ecuda::pair<InputIterator1,InputIterator2>( p.second, p.first );
	#endif
}

template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ ecuda::pair<InputIterator1,InputIterator2>
mismatch( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<ecuda::true_type,ecuda::true_type> ) // device/device memory
{
	#ifdef __CUDA_ARCH__
	while( (first1 != last1) && (*first1 == *first2) ) { ++first1; ++first2; }
	return ecuda::pair<InputIterator1,InputIterator2>(first1,first2);
	#else
	typedef typename ecuda::remove_const<typename ecuda::iterator_traits<InputIterator1>::value_type>::type value_type1;
	typedef std::vector< value_type1, host_allocator<value_type1> > vector_type1;
	vector_type1 v1( ecuda::distance(first1,last1) );
	typedef typename ecuda::remove_const<typename ecuda::iterator_traits<InputIterator2>::value_type>::type value_type2;
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

#pragma hd_warning_disable
template<class InputIterator1,class InputIterator2>
inline __HOST__ __DEVICE__ ecuda::pair<InputIterator1,InputIterator2>
mismatch( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2 )
{
	return impl::mismatch(
					first1, last1,
					first2,
					ecuda::pair<typename ecuda::iterator_traits<InputIterator1>::is_device_iterator,typename ecuda::iterator_traits<InputIterator2>::is_device_iterator>()
				);
}

} // namespace ecuda

#endif
