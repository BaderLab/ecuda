#ifndef ECUDA_ALGORITHM_HPP
#define ECUDA_ALGORITHM_HPP

#include <iterator>
#include <vector>

#include "global.hpp"
#include "apiwrappers.hpp"
#include "iterator.hpp"

#include "algo/copy.hpp"

namespace ecuda {

 // forward declarations


template<class Iterator> __host__ __device__ inline typename std::iterator_traits<Iterator>::difference_type distance( Iterator first, Iterator last );
template<class InputIterator1,class InputIterator2> __host__ __device__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2 );
template<class ForwardIterator,typename T> __host__ __device__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val );
template<class InputIterator1,class InputIterator2> __host__ __device__ inline bool lexicographical_compare( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2 );

template<typename T> __host__ __device__ void swap( T& a, T& b ) __NOEXCEPT__ { T& tmp = a; a = b; b = tmp; }












template<class InputIterator1,class InputIterator2>
__host__ __device__ inline bool __equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, detail::__false_type, detail::__false_type ) {
	#ifdef __CUDA_ARCH__
	return false; // never actually gets called, just here to trick nvcc
	#else
	return std::equal( first1, last1, first2 );
	#endif
}

template<class InputIterator1,class InputIterator2>
__host__ __device__ inline bool __equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, detail::__true_type, detail::__false_type ) {
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
__host__ __device__ inline bool __equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, detail::__false_type, detail::__true_type ) {
	InputIterator2 last2 = first2;
	ecuda::advance( last2, ecuda::distance(first1,last1) );
	return ecuda::equal( first2, last2, first1 );
}

template<class InputIterator1,class InputIterator2>
__host__ __device__ inline bool __equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, detail::__true_type, detail::__true_type ) {
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


template<class InputIterator1,class InputIterator2>
__host__ __device__ inline bool equal( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2 ) {
	return __equal( first1, last1, first2, typename ecuda::iterator_traits<InputIterator1>::is_device_iterator(), typename ecuda::iterator_traits<InputIterator2>::is_device_iterator() );
}



template<class InputIterator1,class InputIterator2>
__host__ __device__ inline bool __lexicographical_compare( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, detail::__false_type, detail::__false_type ) {
	return std::lexicographical_compare( first1, last1, first2, last2 );
}

template<class InputIterator1,class InputIterator2>
__host__ __device__ inline bool __lexicographical_compare( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, detail::__true_type, detail::__false_type ) {
	return std::lexicographical_compare( first1, last1, first2, last2 );
}

template<class InputIterator1,class InputIterator2>
__host__ __device__ inline bool __lexicographical_compare( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, detail::__false_type, detail::__true_type ) {
	return std::lexicographical_compare( first1, last1, first2, last2 );
}

template<class InputIterator1,class InputIterator2>
__host__ __device__ inline bool __lexicographical_compare( InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, detail::__true_type, detail::__true_type ) {
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
	return __lexicographical_compare( first1, last1, first2, last2, typename ecuda::iterator_traits<InputIterator1>::is_device_iterator(), typename ecuda::iterator_traits<InputIterator2>::is_device_iterator() );
}



template<class ForwardIterator,typename T,class IsContiguous>
__host__ __device__ inline void __fill( ForwardIterator first, ForwardIterator last, const T& val, detail::__false_type, IsContiguous ) {
	#ifdef __CUDA_ARCH__
	return; // never actually gets called, just here to trick nvcc
	#else
	// just defer to STL
	std::fill( first, last, val );
	#endif
}

template<class ForwardIterator,typename T>
__host__ __device__ inline void __fill( ForwardIterator first, ForwardIterator last, const T& val, detail::__true_type, detail::__true_type ) {
	#ifdef __CUDA_ARCH__
	while( first != last ) { *first = val; ++first; }
	#else
	typename ecuda::iterator_traits<ForwardIterator>::difference_type n = ecuda::distance( first, last );
	CUDA_CALL( cudaMemset<T>( first.operator->(), val, static_cast<std::size_t>(n) ) );
	#endif
}

template<class ForwardIterator,typename T>
__host__ __device__ inline void __fill( ForwardIterator first, ForwardIterator last, const T& val, detail::__true_type, detail::__false_type ) {
	#ifdef __CUDA_ARCH__
	while( first != last ) { *first = val; ++first; }
	#else
	throw std::invalid_argument( EXCEPTION_MSG( "ecuda::fill() cannot fill a non-contiguous device range" ) );
	#endif
}


template<class ForwardIterator,typename T>
__host__ __device__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val ) {
	__fill(
		first, last,
		val,
		typename ecuda::iterator_traits<ForwardIterator>::is_device_iterator(),
		typename ecuda::iterator_traits<ForwardIterator>::is_contiguous()
	);
}



} // namespace ecuda

#endif

