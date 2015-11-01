#pragma once
#ifndef ECUDA_ALGO_LEXICOGRAPHICAL_COMPARE_HPP
#define ECUDA_ALGO_LEXICOGRAPHICAL_COMPARE_HPP

#include <iterator>
#include <utility>
#include <vector>

#include "../global.hpp"
#include "../allocators.hpp"
#include "../iterator.hpp"
#include "../utility.hpp"

namespace ecuda {

// forward declaration
template<class InputIterator1,class InputIterator2> __HOST__ __DEVICE__ inline bool lexicographical_compare( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2 );

namespace impl {

template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ inline bool
lexicographical_compare( InputIterator1 first1, InputIterator1 last1,
						 InputIterator2 first2, InputIterator2 last2,
						 pair<std::false_type,std::false_type>
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_LEXICOGRAPHICAL_COMPARE_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return false; // never actually gets called, just here to trick nvcc
	#else
	return std::lexicographical_compare( first1, last1, first2, last2 );
	#endif
}

template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ inline bool
lexicographical_compare( InputIterator1 first1, InputIterator1 last1,
						 InputIterator2 first2, InputIterator2 last2,
						 pair<std::true_type,std::false_type>
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_LEXICOGRAPHICAL_COMPARE_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return false; // never actually gets called, just here to trick nvcc
	#else
	return ::ecuda::lexicographical_compare( first2, last2, first1, last1 ); // switch positions to resolve to function below
	#endif
}

template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ inline bool
lexicographical_compare( InputIterator1 first1, InputIterator1 last1,
						 InputIterator2 first2, InputIterator2 last2,
						 pair<std::false_type,std::true_type>
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_LEXICOGRAPHICAL_COMPARE_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return false; // never actually gets called, just here to trick nvcc
	#else
	typedef typename std::remove_const<typename ecuda::iterator_traits<InputIterator2>::value_type>::type value_type;
	std::vector< value_type, host_allocator<value_type> > v1( ecuda::distance( first2, last2 ) );
	{
		typedef typename ecuda::iterator_traits<InputIterator2>::iterator_category iterator_category;
		typedef typename ecuda::iterator_traits<InputIterator2>::is_contiguous iterator_contiguity;
		const bool isSomeKindOfContiguous =
			std::is_same<iterator_contiguity,std::true_type>::value ||
			std::is_same<iterator_category,device_contiguous_block_iterator_tag>::value;
		ECUDA_STATIC_ASSERT(isSomeKindOfContiguous,CANNOT_CALL_LEXICOGRAPHICAL_COMPARE_NONCONTIGUOUS_DEVICE_MEMORY);
	}
	std::vector< value_type, host_allocator<value_type> > v( ::ecuda::distance(first2,last2) );
	::ecuda::copy( first2, last2, v.begin() );
	return ::ecuda::lexicographical_compare( first1, last1, v.begin(), v.end() );
	#endif
}

template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ inline
bool lexicographical_compare( InputIterator1 first1, InputIterator1 last1,
							  InputIterator2 first2, InputIterator2 last2,
							  pair<std::true_type,std::true_type> // compare device to device
)
{
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

} // namespace impl

template<class InputIterator1,class InputIterator2>
__HOST__ __DEVICE__ inline bool lexicographical_compare( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2 ) {
	return impl::lexicographical_compare( first1, last1, first2, last2, pair<typename ecuda::iterator_traits<InputIterator1>::is_device_iterator,typename ecuda::iterator_traits<InputIterator2>::is_device_iterator>() );
}

} // namespace ecuda

#endif
