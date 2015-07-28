#pragma once
#ifndef ECUDA_ALGO_EQUAL_HPP
#define ECUDA_ALGO_EQUAL_HPP

#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"
#include "../utility.hpp"

namespace ecuda {

// forward declaration
template<class InputIterator1,class InputIterator2> __HOST__ __DEVICE__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2 );


template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ inline bool __equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<detail::__false_type,detail::__false_type> ) {
	#ifdef __CUDA_ARCH__
	return false; // never actually gets called, just here to trick nvcc
	#else
	return std::equal( first1, last1, first2 );
	#endif
}

template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ inline bool __equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<detail::__true_type,detail::__false_type> ) {
	#ifdef __CUDA_ARCH__
	return false; // never actually gets called, just here to trick nvcc
	#else
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator1>::value_type>::type valtype1;
	std::vector< valtype1, host_allocator<valtype1> > v1( static_cast<std::size_t>(ecuda::distance(first1,last1)) );
	ecuda::copy( first1, last1, v1.begin() );
	return std::equal( v1.begin(), v1.end(), first2 );
	#endif
}

template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ inline bool __equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<detail::__false_type,detail::__true_type> ) {
	InputIterator2 last2 = first2;
	ecuda::advance( last2, ecuda::distance(first1,last1) );
	return ecuda::equal( first2, last2, first1 );
}

template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ inline bool __equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, ecuda::pair<detail::__true_type,detail::__true_type> ) {
	#ifdef __CUDA_ARCH__
	for( ; first1 != last1; ++first1, ++first2 ) if( !(*first1 == *first2) ) return false;
	return true;
	#else
	// strip const qualifiers otherwise cannot create std::vector<const T>
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator1>::value_type>::type valtype1;
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator2>::value_type>::type valtype2;
	InputIterator2 last2 = first2;
	ecuda::advance( last2, ecuda::distance(first1,last1) );
	std::vector< valtype1, host_allocator<valtype1> > v1( ecuda::distance( first1, last1 ) );
	std::vector< valtype2, host_allocator<valtype2> > v2( ecuda::distance( first2, last2 ) );
	ecuda::copy( first1, last1, v1.begin() );
	ecuda::copy( first2, last2, v2.begin() );
	return std::equal( v1.begin(), v1.end(), v2.begin() );
	#endif
}


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
__HOST__ __DEVICE__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2 ) {
	return __equal( first1, last1, first2, ecuda::pair<typename ecuda::iterator_traits<InputIterator1>::is_device_iterator,typename ecuda::iterator_traits<InputIterator2>::is_device_iterator>() );
}

} // namespace ecuda

#endif
