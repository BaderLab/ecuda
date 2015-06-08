#pragma once
#ifndef ECUDA_ALGO_FILL_HPP
#define ECUDA_ALGO_FILL_HPP

#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"
#include "../utility.hpp"

namespace ecuda {

// forward declaration
template<class ForwardIterator,typename T> __host__ __device__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val );

template<class ForwardIterator,typename T,class IsContiguous>
__host__ __device__ inline void __fill( ForwardIterator first, ForwardIterator last, const T& val, ecuda::pair<detail::__false_type,IsContiguous> ) {
	#ifdef __CUDA_ARCH__
	return; // never actually gets called, just here to trick nvcc
	#else
	// just defer to STL
	std::fill( first, last, val );
	#endif
}

template<class ForwardIterator,typename T>
__host__ __device__ inline void __fill( ForwardIterator first, ForwardIterator last, const T& val, ecuda::pair<detail::__true_type,detail::__true_type> ) {
	#ifdef __CUDA_ARCH__
	while( first != last ) { *first = val; ++first; }
	#else
	typename ecuda::iterator_traits<ForwardIterator>::difference_type n = ecuda::distance( first, last );
	CUDA_CALL( cudaMemset<T>( first.operator->(), val, static_cast<std::size_t>(n) ) );
	#endif
}

template<class ForwardIterator,typename T>
__host__ __device__ inline void __fill( ForwardIterator first, ForwardIterator last, const T& val, ecuda::pair<detail::__true_type,detail::__false_type> ) {
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
		ecuda::pair<typename ecuda::iterator_traits<ForwardIterator>::is_device_iterator,typename ecuda::iterator_traits<ForwardIterator>::is_contiguous>()
	);
}

} // namespace ecuda

#endif
