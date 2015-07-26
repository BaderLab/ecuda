#pragma once
#ifndef ECUDA_ALGO_COPY_HPP
#define ECUDA_ALGO_COPY_HPP

#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../apiwrappers.hpp"
#include "../iterator.hpp"

namespace ecuda {

///
/// \brief copy
///
/// Extension to the STL std::copy function that resolves device iterators
/// and decides the appropriate action at compile-time for device->host,
/// host->device, and device->device copy requests. Host->host requests are
/// delegated to std::copy.
///
/// \param first
/// \param last
/// \param result
/// \return
///
template<class InputIterator,class OutputIterator> __HOST__ __DEVICE__ inline OutputIterator copy( InputIterator first, InputIterator last, OutputIterator result );

// definitions for the 4 possible device/host iterator pairs
template<class InputIterator,class OutputIterator> __HOST__ __DEVICE__ inline OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, detail::__false_type, detail::__false_type );
template<class InputIterator,class OutputIterator> __HOST__ __DEVICE__ inline OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, detail::__false_type, detail::__true_type );
template<class InputIterator,class OutputIterator> __HOST__ __DEVICE__ inline OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, detail::__true_type,  detail::__false_type );
template<class InputIterator,class OutputIterator> __HOST__ __DEVICE__ inline OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, detail::__true_type,  detail::__true_type );

//namespace detail {
//
//template<class Iterator,typename Distance>
//__HOST__ inline bool __is_contiguous( Iterator first, Iterator last, Distance dist ) { return ( last.operator->() - first.operator->() ) == dist; }
//
//} // namespace detail


/**
 * Device to device
 */

template<class InputIterator,class OutputIterator,typename T>
__HOST__ __DEVICE__ inline OutputIterator __copy_device_to_device(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	detail::__true_type, // is_contiguous
	detail::__true_type, // is_contiguous
	T, T
)
{
	#ifdef __CUDA_ARCH__
	//while( first != last ) { *result = *first; ++first; ++result; }
	return result; // never actually gets called, just here to trick nvcc
	#else
	typedef typename std::iterator_traits<OutputIterator>::value_type value_type;
	typename std::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	CUDA_CALL( cudaMemcpy<value_type>( result.operator->(), first.operator->(), static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice ) );
	ecuda::advance( result, static_cast<std::size_t>(n) );
	return result;
	#endif
}

template<class InputIterator,class OutputIterator,typename T,typename U>
__HOST__ __DEVICE__ inline OutputIterator __copy_device_to_device(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	detail::__true_type, // is_contiguous
	detail::__true_type, // is_contiguous
	T, U
)
{
	#ifdef __CUDA_ARCH__
	//while( first != last ) { *result = *first; ++first; ++result; }
	return result; // never actually gets called, just here to trick nvcc
	#else
	if( std::is_same<T,U>::value ) throw std::runtime_error( "ecuda::__copy_host_to_device(InputIterator,InputIterator,OutputIterator,...) different type variety called but types are the same" );
	typedef typename ecuda::iterator_traits<InputIterator>::value_type T1;
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type T2;
	// different types
	std::vector<T1> v1( static_cast<std::size_t>( ecuda::distance(first,last) ) );
	ecuda::copy( first, last, v1.begin() );
	std::vector<T2> v2( v1.size() );
	std::copy( v1.begin(), v1.end(), v2.begin() );
	return ecuda::copy( v2.begin(), v2.end(), result );
	#endif
}

template<class InputIterator,class OutputIterator,typename ContiguousInput,typename ContiguousOutput,typename T,typename U>
__HOST__ __DEVICE__ inline OutputIterator __copy_device_to_device(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	ContiguousInput,  // one or both of these
	ContiguousOutput, // is not contiguous
	T, U
)
{
	#ifdef __CUDA_ARCH__
	//while( first != last ) { *result = *first; ++first; ++result; }
	return result; // never actually gets called, just here to trick nvcc
	#else
	throw std::invalid_argument( EXCEPTION_MSG( "ecuda::copy() cannot copy to or from non-contiguous device iterator" ) );
	#endif
}


template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator __copy(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	detail::__true_type, // is_device_iterator
	detail::__true_type  // is_device_iterator
)
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { *result = *first; ++first; ++result; }
	return result;
	#else
	typedef typename ecuda::iterator_traits<InputIterator>::value_type T1;
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type T2;
	return __copy_device_to_device( first, last, result, typename ecuda::iterator_traits<InputIterator>::is_contiguous(), typename ecuda::iterator_traits<OutputIterator>::is_contiguous(), T1(), T2() );
	#endif
}


/**
 * Host to device
 */

template<class InputIterator,class OutputIterator,typename T>
__HOST__ __DEVICE__ inline OutputIterator __copy_host_to_device(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	detail::__false_type, // is_contiguous
	detail::__true_type,  // is_contiguous
	T, T
)
{
	#ifdef __CUDA_ARCH__
	return result; // never actually gets called, just here to trick nvcc
	#else
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type value_type;
	const typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	std::vector<value_type> v( static_cast<std::size_t>(n) );
	ecuda::copy( first, last, v.begin() );
	ecuda::copy( v.begin(), v.end(), result );
	ecuda::advance( result, static_cast<std::size_t>(n) );
	return result;
	#endif
}

template<class InputIterator,class OutputIterator,typename ContiguousInput,typename T,typename U>
__HOST__ __DEVICE__ inline OutputIterator __copy_host_to_device(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	ContiguousInput,      // contiguity of input is irrelevant here
	detail::__false_type, // is_contiguous
	T, U
)
{
	#ifdef __CUDA_ARCH__
	return result; // never actually gets called, just here to trick nvcc
	#else
	throw std::invalid_argument( EXCEPTION_MSG( "ecuda::copy() cannot copy to non-contiguous device iterator" ) );
	#endif
}

template<class InputIterator,class OutputIterator,typename T>
__HOST__ __DEVICE__ inline OutputIterator __copy_host_to_device(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	detail::__true_type, // is_contiguous
	detail::__true_type, // is_contiguous
	T, T
)
{
	#ifdef __CUDA_ARCH__
	return result; // never actually gets called, just here to trick nvcc
	#else
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type value_type;
	const typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	// explicitly confirm that the output iterator is contiguous
	// @todo - in upcoming C++17 there is a ContiguousIterator type being proposed that will make this check unnecessary
	if( !ecuda::iterator_traits<InputIterator>::confirm_contiguity( first, last, n ) ) return __copy_host_to_device( first, last, result, detail::__false_type(), detail::__true_type(), T(), T() );
	CUDA_CALL( cudaMemcpy<value_type>( result.operator->(), first.operator->(), static_cast<std::size_t>(n), cudaMemcpyHostToDevice ) );
	ecuda::advance( result, static_cast<std::size_t>(n) );
	return result;
	#endif
}

template<class InputIterator,class OutputIterator,typename ContiguousInput,typename T,typename U>
__HOST__ __DEVICE__ inline OutputIterator __copy_host_to_device(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	ContiguousInput,     // is_contiguous
	detail::__true_type, // is_contiguous
	T, U
)
{
	#ifdef __CUDA_ARCH__
	return result; // never actually gets called, just here to trick nvcc
	#else
	if( std::is_same<T,U>::value ) throw std::runtime_error( "ecuda::__copy_host_to_device(InputIterator,InputIterator,OutputIterator,...) different type variety called but types are the same" );
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type value_type;
	const typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	std::vector<U> v( static_cast<std::size_t>(n) );
	std::copy( first, last, v.begin() );
	CUDA_CALL( cudaMemcpy<value_type>( result.operator->(), &v.front(), static_cast<std::size_t>(n), cudaMemcpyHostToDevice ) );
	ecuda::advance( result, static_cast<std::size_t>(n) );
	return result;
	#endif
}

template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator __copy(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	detail::__false_type, // is_device_iterator
	detail::__true_type   // is_device_iterator
)
{
	#ifdef __CUDA_ARCH__
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	// normalize types
	typedef typename ecuda::iterator_traits<InputIterator>::value_type T1;
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type T2;
	return __copy_host_to_device( first, last, result, typename ecuda::iterator_traits<InputIterator>::is_contiguous(), typename ecuda::iterator_traits<OutputIterator>::is_contiguous(), T1(), T2() );
	#endif
}

/**
 * Device to host
 */

template<class InputIterator,class OutputIterator,class ContiguousOutput,typename T,typename U>
__HOST__ __DEVICE__ inline OutputIterator __copy_device_to_host(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	device_iterator_tag,  // non-contiguous
	//detail::__false_type, // is_contiguous
	ContiguousOutput,     // contiguity of output is irrelevant here
	T, U
)
{
	#ifdef __CUDA_ARCH__
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	throw std::invalid_argument( EXCEPTION_MSG( "ecuda::copy() cannot copy from non-contiguous device iterator" ) );
	#endif
}

template<class InputIterator,class OutputIterator,typename T,typename U>
__HOST__ __DEVICE__ inline OutputIterator __copy_device_to_host(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	device_contiguous_iterator_tag, // contiguous
	//detail::__true_type,  // is_contiguous
	detail::__false_type, // is_contiguous
	T, U
)
{
	#ifdef __CUDA_ARCH__
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	// create contiguous staging area for output
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type value_type;
	const typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	std::vector<value_type> v( static_cast<std::size_t>(n) );
	ecuda::copy( first, last, v.begin() );
	std::copy( v.begin(), v.end(), result );
	ecuda::advance( result, static_cast<std::size_t>(n) );
	return result;
	#endif
}

template<class InputIterator,class OutputIterator,typename T>
__HOST__ __DEVICE__ inline OutputIterator __copy_device_to_host(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	device_contiguous_iterator_tag, // contiguous
	//detail::__true_type, // is_contiguous
	detail::__true_type, // is_contiguous
	T, T
)
{
	#ifdef __CUDA_ARCH__
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type value_type;
	const typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	// explicitly confirm that the output iterator is contiguous
	// @todo - in upcoming C++17 there is a ContiguousIterator type being proposed that will make this check unnecessary
	if( !ecuda::iterator_traits<OutputIterator>::confirm_contiguity( result, result+n, n ) ) return __copy_device_to_host( first, last, result, device_contiguous_iterator_tag(), detail::__false_type(), T(), T() );
	CUDA_CALL( cudaMemcpy<value_type>( result.operator->(), first.operator->(), static_cast<std::size_t>(n), cudaMemcpyDeviceToHost ) );
	ecuda::advance( result, static_cast<std::size_t>(n) );
	return result;
	#endif
}

///
/// \todo do this asap
///
template<class InputIterator,class OutputIterator,typename T>
__HOST__ __DEVICE__ inline OutputIterator __copy_device_to_host(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	device_contiguous_block_iterator_tag, // contiguous blocks
	detail::__true_type, // is_contiguous
	T, T
)
{
	#ifdef __CUDA_ARCH__
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	for( ; first != last; result += first.operator->().get_remaining_width(), first += first.operator->().get_remaining_width() )
		__copy_device_to_device( first, first+first.operator->().get_remaining_width(), result, device_contiguous_iterator_tag(), detail::__true_type(), T(), T() );
	return result;
	#endif
}


template<class InputIterator,class OutputIterator,typename T,typename U>
__HOST__ __DEVICE__ inline OutputIterator __copy_device_to_host(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	device_contiguous_iterator_tag, // contiguous
	//detail::__true_type, // is_contiguous
	detail::__true_type, // is_contiguous
	T, U
)
{
	#ifdef __CUDA_ARCH__
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	if( std::is_same<T,U>::value ) throw std::runtime_error( "ecuda::__copy_host_to_device(InputIterator,InputIterator,OutputIterator,...) different type variety called but types are the same" );
	const typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance(first,last);
	std::vector<T> v1( static_cast<std::size_t>(n) );
	CUDA_CALL( cudaMemcpy<T>( &v1.front(), first.operator->(), static_cast<std::size_t>(n), cudaMemcpyDeviceToHost ) );
	std::vector<U> v2( v1.size() );
	std::copy( v1.begin(), v1.end(), v2.begin() );
	CUDA_CALL( cudaMemcpy<U>( result.operator->(), &v2.front(), static_cast<std::size_t>(n), cudaMemcpyDeviceToHost ) );
	ecuda::advance( result, static_cast<std::size_t>(n) );
	return result;
	#endif
}

template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator __copy( // device to host
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	detail::__true_type, // is_device_iterator
	detail::__false_type // is_device_iterator
)
{
	#ifdef __CUDA_ARCH__
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	// normalize types
	typedef typename ecuda::iterator_traits<InputIterator>::value_type T1;
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type T2;
	//return __copy_device_to_host( first, last, result, typename ecuda::iterator_traits<InputIterator>::is_contiguous(), typename ecuda::iterator_traits<OutputIterator>::is_contiguous(), T1(), T2() );
	return __copy_device_to_host( first, last, result, typename ecuda::iterator_traits<InputIterator>::iterator_category(), typename ecuda::iterator_traits<OutputIterator>::is_contiguous(), T1(), T2() );
	#endif
}

/**
 * Host to host
 */

template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator __copy(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	detail::__false_type, // is_device_iterator
	detail::__false_type  // is_device_iterator
)
{
	#ifdef __CUDA_ARCH__
	return result; // never actually gets called, just here to trick nvcc
	#else
	// just defer to STL
	return std::copy( first, last, result );
	#endif
}



template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator copy( InputIterator first, InputIterator last, OutputIterator result ) {
	typedef typename ecuda::iterator_traits<InputIterator>::is_device_iterator is_input_device_iterator;
	typedef typename ecuda::iterator_traits<OutputIterator>::is_device_iterator is_output_device_iterator;
	return __copy( first, last,	result, is_input_device_iterator(), is_output_device_iterator() );
}

} // namespace ecuda

#endif
