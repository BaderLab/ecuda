#pragma once
#ifndef ECUDA_ALGO_COPY_HPP
#define ECUDA_ALGO_COPY_HPP

#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../apiwrappers.hpp"
#include "../iterator.hpp"
#include "../utility.hpp"

namespace ecuda {

///
/// \brief Replacement for std::copy.
///
/// ecuda::copy is identical to std::copy, but can be a) called from device code, and b) supports
/// device memory when called from host code.
///
/// Compile-time checks are performed to determine which action should be taken. If called from
/// device code, then it must be true that both the input and output refer to device memory (otherwise
/// nvcc will fail before evaluating the ecuda::copy call) and the copying is done on-device.
/// If the called from host code and both the input and output refer to host memory, the evaluation
/// is delegated to std::copy. If called from host code, and one or both of the input and output refers
/// to device memory, there is a compile-time assertion that fails if the device memory is non-contiguous.
/// Otherwise, a call to cudaMemcpy is performed with parameters depending on the input and output
/// memory types (e.g. if input is host and if output is device, then cudaMemcpy is called with
/// cudaMemcpyHostToDevice used as the cudaMemcpyKind parameter).  In addition, when one or both of the
/// input and output iterators refers to device memory, a call to ecuda::copy from host code results in
/// a compile-time check to determine if the value_type of the input and output iterator are the same.
/// If not, and the call is on host code, host staging memory is allocated to perform the type
/// conversion.
///
/// \param first,last the range of elements to copy
/// \param result the beginning of the destination range
/// \returns Output iterator to the element in the destination range, one past the last element copied.
///
template<class InputIterator,class OutputIterator> __HOST__ __DEVICE__ inline OutputIterator copy( InputIterator first, InputIterator last, OutputIterator result );

// definitions for the 4 possible device/host iterator pairs
//template<class InputIterator,class OutputIterator> __HOST__ __DEVICE__ inline OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, std::false_type, std::false_type );
//template<class InputIterator,class OutputIterator> __HOST__ __DEVICE__ inline OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, std::false_type, std::true_type );
//template<class InputIterator,class OutputIterator> __HOST__ __DEVICE__ inline OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, std::true_type,  std::false_type );
//template<class InputIterator,class OutputIterator> __HOST__ __DEVICE__ inline OutputIterator __copy( InputIterator first, InputIterator last, OutputIterator result, std::true_type,  std::true_type );

//namespace detail {
//
//template<class Iterator,typename Distance>
//__HOST__ inline bool __is_contiguous( Iterator first, Iterator last, Distance dist ) { return ( last.operator->() - first.operator->() ) == dist; }
//
//} // namespace detail

namespace impl {

/**
 * Device to device
 */

template<class InputIterator,class OutputIterator,typename T>
__HOST__ __DEVICE__ inline OutputIterator copy_device_to_device(
	InputIterator first, InputIterator last,
	OutputIterator result,
	ecuda::pair<detail::contiguous_type,detail::contiguous_type>,
	T, T
)
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { *result = *first; ++first; ++result; }
	return result;
	#else
	typedef typename std::iterator_traits<OutputIterator>::value_type value_type;
	typename std::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	CUDA_CALL( cudaMemcpy<value_type>( result.operator->(), first.operator->(), static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice ) );
	ecuda::advance( result, static_cast<std::size_t>(n) );
	return result;
	#endif
}

template<class InputIterator,class OutputIterator,typename T,typename U>
__HOST__ __DEVICE__ inline OutputIterator copy_device_to_device(
	InputIterator first, InputIterator last,
	OutputIterator result,
	ecuda::pair<detail::contiguous_type,detail::contiguous_type>,
	T, U
)
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { *result = *first; ++first; ++result; }
	return result;
	#else
	const bool isSameTypes = std::is_same<T,U>::value;
	ECUDA_STATIC_ASSERT(!isSameTypes,COPY_IS_UNEXPECTEDLY_BETWEEN_IDENTICAL_TYPES);
	//if( std::is_same<T,U>::value ) throw std::runtime_error( "ecuda::__copy_host_to_device(InputIterator,InputIterator,OutputIterator,...) different type variety called but types are the same" );
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

/*
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
*/

template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator copy(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	ecuda::pair<detail::device_type,detail::device_type> // device -> device
)
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { *result = *first; ++first; ++result; }
	return result;
	#else
	const bool isInputIteratorContiguous = std::is_same<typename ecuda::iterator_traits<InputIterator>::is_contiguous,std::true_type>::value;
	ECUDA_STATIC_ASSERT(isInputIteratorContiguous,CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_AS_SOURCE_FOR_COPY);
	const bool isOutputIteratorContiguous = std::is_same<typename ecuda::iterator_traits<OutputIterator>::is_contiguous,std::true_type>::value;
	ECUDA_STATIC_ASSERT(isOutputIteratorContiguous,CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_AS_DESTINATION_FOR_COPY);
	typedef typename ecuda::iterator_traits<InputIterator>::value_type T1;
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type T2;
	typedef typename ecuda::iterator_traits<InputIterator>::is_contiguous input_contiguity;
	typedef typename ecuda::iterator_traits<OutputIterator>::is_contiguous output_contiguity;
	return copy_device_to_device( first, last, result, ecuda::pair<input_contiguity,output_contiguity>(), T1(), T2() );
	#endif
}


/**
 * Host to device
 */

template<class InputIterator,class OutputIterator,typename T>
__HOST__ __DEVICE__ inline OutputIterator copy_host_to_device(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	std::false_type, // is_contiguous
	device_contiguous_iterator_tag,
	T, T
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
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

template<class InputIterator,class OutputIterator,typename T>
__HOST__ __DEVICE__ inline OutputIterator copy_host_to_device(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	std::true_type, // is_contiguous
	device_contiguous_iterator_tag,
	T, T
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return result; // never actually gets called, just here to trick nvcc
	#else
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type value_type;
	const typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	// explicitly confirm that the output iterator is contiguous
	// @todo - in upcoming C++17 there is a ContiguousIterator type being proposed that will make this check unnecessary
	if( !ecuda::iterator_traits<InputIterator>::confirm_contiguity( first, last, n ) ) return copy_host_to_device( first, last, result, std::false_type(), device_contiguous_iterator_tag(), T(), T() );
	typedef typename std::add_pointer<value_type>::type pointer;
	pointer p = naked_cast<pointer>( result.operator->() );
	pointer q = naked_cast<pointer>( first.operator->() );
	//typename ecuda::pointer_traits<typename ecuda::iterator_traits<OutputIterator>::pointer>::naked_pointer p =
	//	ecuda::pointer_traits<typename ecuda::iterator_traits<OutputIterator>::pointer>().undress( result.operator->() );
	//typename ecuda::pointer_traits<typename ecuda::iterator_traits<InputIterator>::pointer>::naked_pointer q =
	//	ecuda::pointer_traits<typename ecuda::iterator_traits<InputIterator>::pointer>().undress( first.operator->() );
	CUDA_CALL(
		cudaMemcpy<value_type>(
			p, q,
			//ecuda::pointer_traits<typename ecuda::iterator_traits<OutputIterator>::pointer>().undress( result.operator->() ),
			//ecuda::pointer_traits<typename ecuda::iterator_traits<InputIterator>::pointer>().undress( first.operator->() ),
			static_cast<std::size_t>(n),
			cudaMemcpyHostToDevice
		)
	);
	ecuda::advance( result, static_cast<std::size_t>(n) );
	return result;
	#endif
}

template<class InputIterator,class OutputIterator,typename T>
__HOST__ __DEVICE__ inline OutputIterator copy_host_to_device(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	std::true_type, // is_contiguous
	device_contiguous_block_iterator_tag,
	T, T
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return result; // never actually gets called, just here to trick nvcc
	#else
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type value_type;
	typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	const std::size_t width = result.operator->().get_width();
	while( n > 0 ) {
		result = copy_host_to_device( first, first+width, result, std::true_type(), device_contiguous_iterator_tag(), T(), T() );
		first += width;
		n -= width;
	}
	return result;
	#endif
}

template<class InputIterator,class OutputIterator,typename ContiguousInput,typename DeviceIteratorTag,typename T,typename U>
__HOST__ __DEVICE__ inline OutputIterator copy_host_to_device(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	ContiguousInput,     // is_contiguous
	DeviceIteratorTag,   // device iterator type doesn't matter yet
	T, U
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return result; // never actually gets called, just here to trick nvcc
	#else
	//if( std::is_same<T,U>::value ) throw std::runtime_error( "ecuda::__copy_host_to_device(InputIterator,InputIterator,OutputIterator,...) different type variety called but types are the same" );
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type value_type;
	const typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	std::vector<U> v( static_cast<std::size_t>(n) );
	std::copy( first, last, v.begin() );
	return copy_host_to_device( v.begin(), v.end(), result, typename ecuda::iterator_traits<InputIterator>::is_contiguous(), DeviceIteratorTag(), U(), U() );
	//CUDA_CALL( cudaMemcpy<value_type>( result.operator->(), &v.front(), static_cast<std::size_t>(n), cudaMemcpyHostToDevice ) );
	//ecuda::advance( result, static_cast<std::size_t>(n) );
	//return result;
	#endif
}

template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator copy( InputIterator first, InputIterator last, OutputIterator result, ecuda::pair<detail::host_type,detail::device_type> ) { // host -> device
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	// normalize types
	//const bool isOutputIteratorContiguous = std::is_same<typename ecuda::iterator_traits<OutputIterator>::is_contiguous,std::true_type>::value;
	//const bool isOutputIteratorBlockContiguous = std::is_same<typename ecuda::iterator_traits<OutputIterator>::iterator_category,ecuda::device_contiguous_block_iterator_tag>::value;
	const bool isSomeKindOfContiguous =
		std::is_same<typename ecuda::iterator_traits<OutputIterator>::is_contiguous,std::true_type>::value ||
		std::is_same<typename ecuda::iterator_traits<OutputIterator>::iterator_category,ecuda::device_contiguous_block_iterator_tag>::value;
	//ECUDA_STATIC_ASSERT(isOutputIteratorContiguous || isOutputIteratorBlockContiguous,CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_AS_DESTINATION_FOR_COPY);
	ECUDA_STATIC_ASSERT(isSomeKindOfContiguous,CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_AS_DESTINATION_FOR_COPY);
	typedef typename ecuda::iterator_traits<InputIterator>::value_type T1;
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type T2;
	return copy_host_to_device( first, last, result, typename ecuda::iterator_traits<InputIterator>::is_contiguous(), typename ecuda::iterator_traits<OutputIterator>::iterator_category(), T1(), T2() );
	#endif
}

/**
 * Device to host
 */

/*
template<class InputIterator,class OutputIterator,class ContiguousOutput,typename T,typename U>
__HOST__ __DEVICE__ inline OutputIterator __copy_device_to_host(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	device_iterator_tag,  // non-contiguous
	//std::false_type, // is_contiguous
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
*/

template<class InputIterator,class OutputIterator,typename T,typename U>
__HOST__ __DEVICE__ inline OutputIterator copy_device_to_host(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	device_contiguous_iterator_tag, // contiguous
	//std::true_type,  // is_contiguous
	std::false_type, // is_contiguous
	T, U
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
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
__HOST__ __DEVICE__ inline OutputIterator copy_device_to_host(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	device_contiguous_iterator_tag, // contiguous
	//std::true_type, // is_contiguous
	std::true_type, // is_contiguous
	T, T
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type value_type;
	const typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	// explicitly confirm that the output iterator is contiguous
	// @todo - in upcoming C++17 there is a ContiguousIterator type being proposed that will make this check unnecessary
	if( !ecuda::iterator_traits<OutputIterator>::confirm_contiguity( result, result+n, n ) ) return copy_device_to_host( first, last, result, device_contiguous_iterator_tag(), std::false_type(), T(), T() );
	CUDA_CALL( cudaMemcpy<value_type>( result.operator->(), first.operator->(), static_cast<std::size_t>(n), cudaMemcpyDeviceToHost ) );
	ecuda::advance( result, static_cast<std::size_t>(n) );
	return result;
	#endif
}

///
/// \todo do this asap
///
template<class InputIterator,class OutputIterator,typename T>
__HOST__ __DEVICE__ inline OutputIterator copy_device_to_host(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	device_contiguous_block_iterator_tag, // contiguous blocks
	std::true_type, // is_contiguous
	T, T
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	for( ; first != last; result += first.operator->().get_remaining_width(), first += first.operator->().get_remaining_width() )
		copy_device_to_device( first, first+first.operator->().get_remaining_width(), result, device_contiguous_iterator_tag(), std::true_type(), T(), T() );
	return result;
	#endif
}


template<class InputIterator,class OutputIterator,typename T,typename U>
__HOST__ __DEVICE__ inline OutputIterator copy_device_to_host(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	device_contiguous_iterator_tag, // contiguous
	//std::true_type, // is_contiguous
	std::true_type, // is_contiguous
	T, U
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
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
__HOST__ __DEVICE__ inline OutputIterator copy(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	ecuda::pair<detail::device_type,detail::host_type> // device -> host
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	// compile-time assertion that checks that the input device iterator is contiguous memory
	const bool isInputIteratorContiguous = std::is_same<typename ecuda::iterator_traits<InputIterator>::is_contiguous,std::true_type>::value;
	ECUDA_STATIC_ASSERT(isInputIteratorContiguous,CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_AS_SOURCE_FOR_COPY);
	// normalize types
	typedef typename ecuda::iterator_traits<InputIterator>::value_type T1;
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type T2;
	//return __copy_device_to_host( first, last, result, typename ecuda::iterator_traits<InputIterator>::is_contiguous(), typename ecuda::iterator_traits<OutputIterator>::is_contiguous(), T1(), T2() );
	return copy_device_to_host( first, last, result, typename ecuda::iterator_traits<InputIterator>::iterator_category(), typename ecuda::iterator_traits<OutputIterator>::is_contiguous(), T1(), T2() );
	#endif
}

/**
 * Host to host
 */

template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator copy(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	ecuda::pair<detail::host_type,detail::host_type> // host -> host
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	// just defer to STL
	return std::copy( first, last, result );
	#endif
}

} // namespace impl

template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator copy( InputIterator first, InputIterator last, OutputIterator result ) {
	typedef typename ecuda::iterator_traits<InputIterator>::is_device_iterator input_memory_type;   // host_type is an alias for std::false_type
	typedef typename ecuda::iterator_traits<OutputIterator>::is_device_iterator output_memory_type; // device_type is an alias for std::true_type
	return impl::copy( first, last, result, ecuda::pair<input_memory_type,output_memory_type>() );
}

} // namespace ecuda

#endif
