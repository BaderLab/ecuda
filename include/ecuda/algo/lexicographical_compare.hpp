#pragma once
#ifndef ECUDA_ALGO_LEXICOGRAPHICAL_COMPARE_HPP
#define ECUDA_ALGO_LEXICOGRAPHICAL_COMPARE_HPP

#include <iterator>
#include <utility>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"

namespace ecuda {

// forward declaration
template<class InputIterator1,class InputIterator2> __host__ __device__ inline bool lexicographical_compare( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2 );



template<class InputIterator1,class InputIterator2>
__host__ __device__ inline bool __lexicographical_compare( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, std::pair<detail::__false_type,detail::__false_type> ) {
	return std::lexicographical_compare( first1, last1, first2, last2 );
}

template<class InputIterator1,class InputIterator2>
__host__ __device__ inline bool __lexicographical_compare( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, std::pair<detail::__true_type,detail::__false_type> ) {
	return std::lexicographical_compare( first1, last1, first2, last2 );
}

template<class InputIterator1,class InputIterator2>
__host__ __device__ inline bool __lexicographical_compare( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, std::pair<detail::__false_type,detail::__true_type> ) {
	return std::lexicographical_compare( first1, last1, first2, last2 );
}

template<class InputIterator1,class InputIterator2>
__host__ __device__ inline bool __lexicographical_compare( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, std::pair<detail::__true_type,detail::__true_type> ) {
	#ifdef __CUDA_ARCH__
	for( ; (first1 != last1) and (first2 != last2); ++first1, ++first2 ) {
		if( *first1 < *first2 ) return true;
		if( *first2 < *first1 ) return false;
	}
	return (first1 == last1) and (first2 != last2);
	#else
	// strip const qualifiers if present because std::vector<const T> is useless
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator1>::value_type>::type valtype1;
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator2>::value_type>::type valtype2;
	std::vector< valtype1, host_allocator<valtype1> > v1( ecuda::distance( first1, last1 ) );
	std::vector< valtype2, host_allocator<valtype2> > v2( ecuda::distance( first2, last2 ) );
	ecuda::copy( first1, last1, v1.begin() );
	ecuda::copy( first2, last2, v2.begin() );
	return std::lexicographical_compare( v1.begin(), v1.end(), v2.begin(), v2.end() );
	#endif
}

template<class InputIterator1,class InputIterator2>
__host__ __device__ inline bool lexicographical_compare( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2 ) {
	return __lexicographical_compare( first1, last1, first2, last2, std::pair<typename ecuda::iterator_traits<InputIterator1>::is_device_iterator,typename ecuda::iterator_traits<InputIterator2>::is_device_iterator>() );
}

} // namespace ecuda

#endif
