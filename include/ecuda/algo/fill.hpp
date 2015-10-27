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
template<class ForwardIterator,typename T> __HOST__ __DEVICE__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val );

namespace impl {

template<class ForwardIterator,typename T,class IsContiguous>
__HOST__ __DEVICE__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val, ecuda::pair<std::false_type,IsContiguous> ) {
	#ifdef __CUDA_ARCH__
	return; // never actually gets called, just here to trick nvcc
	#else
	// just defer to STL
	std::fill( first, last, val );
	#endif
}

template<class ForwardIterator,typename T>
__HOST__ __DEVICE__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val, ecuda::pair<std::true_type,std::true_type> ) {
	#ifdef __CUDA_ARCH__
	while( first != last ) { *first = val; ++first; }
	#else
	typename ecuda::iterator_traits<ForwardIterator>::difference_type n = ecuda::distance( first, last );
	CUDA_CALL( cudaMemset<T>( first.operator->(), val, static_cast<std::size_t>(n) ) );
	#endif
}

template<class ForwardIterator,typename T>
__HOST__ __DEVICE__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val, ecuda::pair<std::true_type,std::false_type> ) {
	#ifdef __CUDA_ARCH__
	while( first != last ) { *first = val; ++first; }
	#else
	throw std::invalid_argument( EXCEPTION_MSG( "ecuda::fill() cannot fill a non-contiguous device range" ) );
	#endif
}
/*
template<class ForwardIterator, typename T>
__HOST__ __DEVICE__ inline void __fill( ForwardIterator first, ForwardIterator last, const T& val, device_iterator_tag ) {
	#ifdef __CUDA_ARCH__
	// never gets called
	#else
	ECUDA_STATIC_ASSERT(false,CANNOT_FILL_WITH_NONCONTIGUOUS_DEVICE_ITERATOR);
	#endif
}
*/

template<class ForwardIterator, typename T>
__HOST__ __DEVICE__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val, device_contiguous_iterator_tag ) {
	#ifdef __CUDA_ARCH__
	// never gets called
	#else
	typename ecuda::iterator_traits<ForwardIterator>::difference_type n = ecuda::distance( first, last );
	CUDA_CALL( cudaMemset<T>( first.operator->(), val, static_cast<std::size_t>(n) ) );
	#endif
}

template<class ForwardIterator, typename T>
__HOST__ __DEVICE__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val, device_contiguous_block_iterator_tag ) {
	#ifdef __CUDA_ARCH__
	// never gets called
	#else
	typename ecuda::iterator_traits<ForwardIterator>::pointer p = first.operator->();
	typename ecuda::iterator_traits<ForwardIterator>::pointer q = last.operator->();

	typename ecuda::iterator_traits<ForwardIterator>::difference_type n = ecuda::distance( first, last );
	CUDA_CALL( cudaMemset<T>( first.operator->(), val, static_cast<std::size_t>(n) ) );
	#endif
}


template<class ForwardIterator, typename T>
__HOST__ __DEVICE__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val, detail::device_type ) {
	#ifdef __CUDA_ARCH__
	while( first != last ) { *first = *val; ++first; }
	#else
	const bool isIteratorContiguous = std::is_same<typename ecuda::iterator_traits<ForwardIterator>::iterator_category,device_iterator_tag>::value;
	ECUDA_STATIC_ASSERT( isIteratorContiguous, CANNOT_FILL_WITH_NONCONTIGUOUS_DEVICE_ITERATOR );
	fill( first, last, val, typename ecuda::iterator_traits<ForwardIterator>::iterator_category() );
	std::fill( first, last, val );
	#endif
}


template<class ForwardIterator,typename T>
__HOST__ __DEVICE__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val, detail::host_type ) {
	#ifdef __CUDA_ARCH__
	#else
	std::fill( first, last, val );
	#endif
}

} // namespace impl

template<class ForwardIterator,typename T>
__HOST__ __DEVICE__ inline void fill( ForwardIterator first, ForwardIterator last, const T& val ) {
	impl::fill(
		first, last,
		val,
		ecuda::pair<typename ecuda::iterator_traits<ForwardIterator>::is_device_iterator,typename ecuda::iterator_traits<ForwardIterator>::is_contiguous>()
	);
}

} // namespace ecuda

#endif
