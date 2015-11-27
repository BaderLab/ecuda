/*
Copyright (c) 2014-2015, Scott Zuyderduyn
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
// algo/copy.hpp
//
// Extension of std::copy that recognizes device memory and can be called from
// host or device code.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------
#pragma once
#ifndef ECUDA_ALGO_COPY_HPP
#define ECUDA_ALGO_COPY_HPP

#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../algorithm.hpp"
#include "../allocators.hpp"
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

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

//
// Start of DEVICE to DEVICE implementations
//

//
// Source:      any device memory
// Destination: any device memory
// Value types: any
// On Device  : element-by-element copy
// On Host    : determine contiguity and data types of device memories, generate
//              a compile-time error if called from the host and either of the
//              memories are non-contiguous, otherwise delegate to a same-type
//              copy or different-type copy as appropriate
//
template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator copy(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	ecuda::pair<detail::device_type,detail::device_type> // device -> device
);
// implementation makes calls to functions in the ecuda::impl::device_to_device namespace

namespace device_to_device {

//
// Source:      contiguous device memory
// Destination: contiguous device memory
// Value types: same
// On Device  : element-by-element copy
// On Host    : call ecuda::cudaMemcpy to copy sequence
//
template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator copy(
	InputIterator first, InputIterator last,
	OutputIterator result,
	ecuda::pair<device_contiguous_iterator_tag,device_contiguous_iterator_tag>
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

//
// Source:      blocks of contiguous device memory
// Destination: contiguous device memory
// Value types: same
// On Device  : element-by-element copy
// On Host    : call copy on each individual contiguous block
//
template<class OutputIterator,typename T,typename P>
__HOST__ __DEVICE__ inline OutputIterator copy(
	device_contiguous_block_iterator<T,P> first, device_contiguous_block_iterator<T,P> last,
	OutputIterator result,
	ecuda::pair<device_contiguous_block_iterator_tag,device_contiguous_iterator_tag>
)
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { *result = *first; ++first; ++result; }
	return result;
	#else
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type value_type;
	typedef device_contiguous_block_iterator<T,P> input_iterator_type;

	{
		// if there is any leading data before the memory becomes regularly aligned
		// then copy it first
		const std::size_t leading = first.operator->().get_remaining_width();
		if( leading < first.operator->().get_width() ) { // leading with partial row
			typename input_iterator_type::contiguous_iterator first2 = first.contiguous_begin();
			result = ::ecuda::copy( first2, first2 + leading, result );
			::ecuda::advance( first, leading );
		}
	}

	{
		// memory is now guaranteed to be regularly aligned so we can use cudaMemcpy2D
		typedef typename ecuda::add_pointer<value_type>::type pointer;
		pointer dest = naked_cast<pointer>( result.operator->() );
		typedef typename ecuda::add_pointer<const value_type>::type const_pointer;
		const_pointer src = naked_cast<const_pointer>( first.operator->() );

		const size_t pitch = first.operator->().get_pitch();
		const std::size_t width = first.operator->().get_width();
		const std::size_t rows = ::ecuda::distance( first, last ) / width;
		CUDA_CALL( cudaMemcpy2D<value_type>( dest, width*sizeof(value_type), src, pitch, width, rows, cudaMemcpyDeviceToDevice ) );
		::ecuda::advance( first, width*rows );
		::ecuda::advance( result, width*rows );
	}

	{
		// if there is any trailing data where the memory ceases to be regularly aligned
		// then copy the rest of it
		const std::size_t trailing = ::ecuda::distance( first, last ) % first.operator->().get_width();
		if( trailing > 0 ) {
			typename input_iterator_type::contiguous_iterator first2 = first.contiguous_begin();
			result = ::ecuda::copy( first2, first2 + trailing, result );
		}
	}

	return result;
	/*
	typename ecuda::iterator_traits<input_iterator_type>::difference_type n = ecuda::distance( first, last );
	while( n > 0 ) {
		const std::size_t width = first.operator->().get_remaining_width();
		const std::size_t copy_width = width > n ? n : width;
		typename input_iterator_type::contiguous_iterator first2 = first.contiguous_begin();
		result = ::ecuda::copy( first2, first2+copy_width, result );
		n -= copy_width;
	}
	return result;
	*/
	#endif
}

//
// Source     : contiguous device memory
// Destination: blocks of contiguous device memory
// Value types: same
// On Device  : element-by-element copy
// On Host    : call copy on each individual contiguous block
//
template<class InputIterator,typename T,typename P>
__HOST__ __DEVICE__ inline device_contiguous_block_iterator<T,P> copy(
	InputIterator first, InputIterator last,
	device_contiguous_block_iterator<T,P> result,
	ecuda::pair<device_contiguous_iterator_tag,device_contiguous_block_iterator_tag>
)
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { *result = *first; ++first; ++result; }
	return result;
	#else
	typedef device_contiguous_block_iterator<T,P> output_iterator_type;
	typedef typename ecuda::iterator_traits<output_iterator_type>::value_type value_type;

	{
		// if there is any leading data before the memory becomes regularly aligned
		// then copy it first
		const std::size_t leading = result.operator->().get_remaining_width();
		if( leading < result.operator->().get_width() ) { // leading with partial row
			::ecuda::copy( first, first + leading, result.contiguous_begin() );
			::ecuda::advance( first, leading );
			::ecuda::advance( result, leading );
		}
	}

	{
		// memory is now guaranteed to be regularly aligned so we can use cudaMemcpy2D
		typedef typename ecuda::add_pointer<value_type>::type pointer;
		pointer dest = naked_cast<pointer>( result.operator->() );
		typedef typename ecuda::add_pointer<const value_type>::type const_pointer;
		const_pointer src = naked_cast<const_pointer>( first.operator->() );

		const size_t pitch = result.operator->().get_pitch();
		const std::size_t width = result.operator->().get_width();
		const std::size_t rows = ::ecuda::distance( first, last ) / width;

		CUDA_CALL( cudaMemcpy2D<value_type>( dest, pitch, src, width*sizeof(value_type), width, rows, cudaMemcpyDeviceToDevice ) );
		::ecuda::advance( first, width*rows );
		::ecuda::advance( result, width*rows );
	}

	{
		// if there is any trailing data where the memory ceases to be regularly aligned
		// then copy the rest of it
		const std::size_t trailing = ::ecuda::distance( first, last ) % result.operator->().get_width();
		if( trailing > 0 ) {
			::ecuda::copy( first, first + trailing, result.contiguous_begin() );
			::ecuda::advance( result, trailing );
		}
	}

	return result;
	/*
	typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	while( n > 0 ) {
		const std::size_t width = result.operator->().get_remaining_width();
		const std::size_t copy_width = width > n ? n : width;
		typename output_iterator_type::contiguous_iterator result2 = result.contiguous_begin();
		::ecuda::copy( first, first+copy_width, result2 );
		result += copy_width;
		n -= copy_width;
	}
	return result;
	*/
	#endif
}

//
// Source     : blocks of contiguous device memory
// Destination: blocks of contiguous device memory
// Value types: same
// On Device  : element-by-element copy
// On Host    : call copy on each individual contiguous block
//
template<typename T,typename P,typename U,typename Q>
__HOST__ __DEVICE__ inline device_contiguous_block_iterator<U,Q> copy(
	device_contiguous_block_iterator<T,P> first, device_contiguous_block_iterator<T,P> last,
	device_contiguous_block_iterator<U,Q> result,
	ecuda::pair<device_contiguous_block_iterator_tag,device_contiguous_block_iterator_tag>
)
{
	#ifdef __CUDA_ARCH__
	while( first != last ) { *result = *first; ++first; ++result; }
	return result;
	#else
	typedef device_contiguous_block_iterator<T,P> input_iterator_type;
	typedef device_contiguous_block_iterator<U,Q> output_iterator_type;
	typedef typename ecuda::iterator_traits<output_iterator_type>::value_type value_type;

	if( ( first.operator->().get_width()           == result.operator->().get_width() ) &&
		( first.operator->().get_remaining_width() == result.operator->().get_remaining_width() ) ) { // only when this is true can we do a cudaMemcpy2D call

		{
			// if there is any leading data before the memory becomes regularly aligned
			// then copy it first
			const std::size_t leading = result.operator->().get_remaining_width();
			if( leading < result.operator->().get_width() ) { // leading with partial row
				::ecuda::copy( first, first + leading, result.contiguous_begin() );
				::ecuda::advance( first, leading );
				::ecuda::advance( result, leading );
			}
		}

		{
			// memory is now guaranteed to be regularly aligned so we can use cudaMemcpy2D
			typedef typename ecuda::add_pointer<value_type>::type pointer;
			pointer dest = naked_cast<pointer>( result.operator->() );
			typedef typename ecuda::add_pointer<const value_type>::type const_pointer;
			const_pointer src = naked_cast<const_pointer>( first.operator->() );

			const size_t src_pitch = first.operator->().get_pitch();
			const size_t dest_pitch = result.operator->().get_pitch();
			const std::size_t width = result.operator->().get_width();
			const std::size_t rows = ::ecuda::distance( first, last ) / width;

			CUDA_CALL( cudaMemcpy2D<value_type>( dest, dest_pitch, src, src_pitch, width, rows, cudaMemcpyDeviceToDevice ) );
			::ecuda::advance( first, width*rows );
			::ecuda::advance( result, width*rows );
		}

		{
			// if there is any trailing data where the memory ceases to be regularly aligned
			// then copy the rest of it
			const std::size_t trailing = ::ecuda::distance( first, last ) % result.operator->().get_width();
			if( trailing > 0 ) {
				::ecuda::copy( first, first + trailing, result.contiguous_begin() );
				::ecuda::advance( result, trailing );
			}
		}

		return result;

	} // end case where both alignments are compatible

	// this will copy pieces of differently aligned memory that
	// will work, but requires (possibly many) calls to
	// cudaMemcpy, so there will be a performance hit

	typename input_iterator_type::difference_type n = ecuda::distance( first, last );
	while( n > 0 ) {
		const std::size_t width = ecuda::min( first.operator->().get_remaining_width(), result.operator->().get_remaining_width() );
		const std::size_t copy_width = width > n ? n : width;
		typename input_iterator_type::contiguous_iterator first2 = first.contiguous_begin();
		typename output_iterator_type::contiguous_iterator result2 = result.contiguous_begin();
		::ecuda::copy( first2, first2+copy_width, result2 );
		first  += copy_width; // original input iterator has to catch up
		result += copy_width; // original output iterator has to catch up
		n -= copy_width;
	}
	return result;
	#endif
}

} // namespace device_to_device

//
// Implementation only, function declaration and documentation above.
//
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

	typedef typename ecuda::iterator_traits<InputIterator>::is_contiguous     input_contiguity;
	typedef typename ecuda::iterator_traits<InputIterator>::iterator_category input_iterator_category;
	{
		// compile-time check that input iterator traverses contiguous memory
		const bool isSomeKindOfContiguous =
			ecuda::is_same<input_contiguity,ecuda::true_type>::value ||
			ecuda::is_same<input_iterator_category,device_contiguous_block_iterator_tag>::value;
		ECUDA_STATIC_ASSERT(isSomeKindOfContiguous,CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_AS_SOURCE_FOR_COPY);
	}

	typedef typename ecuda::iterator_traits<OutputIterator>::is_contiguous     output_contiguity;
	typedef typename ecuda::iterator_traits<OutputIterator>::iterator_category output_iterator_category;
	{
		// compile-time check that output iterator traverses contiguous memory
		const bool isSomeKindOfContiguous =
			ecuda::is_same<output_contiguity,ecuda::true_type>::value ||
			ecuda::is_same<output_iterator_category,device_contiguous_block_iterator_tag>::value;
		ECUDA_STATIC_ASSERT(isSomeKindOfContiguous,CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_AS_DESTINATION_FOR_COPY);
	}
	typedef typename ecuda::iterator_traits<InputIterator>::value_type  T;
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type U;
	{
		// compile-time check that types are the same
		// if not, copy to host staging memory, do type conversion, then copy
		// final result to destination device memory
		const bool isSameType = ecuda::is_same<T,U>::value;
		if( !isSameType ) {
			std::vector< typename ecuda::remove_const<T>::type, host_allocator<typename ecuda::remove_const<T>::type> > v1( std::distance( first, last ) );
			::ecuda::copy( first, last, v1.begin() );
			std::vector< U, host_allocator<U> > v2( v1.size() );
			::ecuda::copy( v1.begin(), v1.end(), v2.begin() );
			return ::ecuda::copy( v2.begin(), v2.end(), result );
		} else {
			return impl::device_to_device::copy( first, last, result, ecuda::pair<input_iterator_category,output_iterator_category>() );
		}
	}
	#endif
}

//
// Start of HOST to DEVICE implementations
//

//
// Source:      any host memory
// Destination: any device memory
// Value types: any
// On Device  : compile-time assertion
// On Host    : determine contiguity and data types of device memory, generate
//              a compile-time error if called from the host and device memory
//              is non-contiguous, perform a compile-time check for type equality
//              and insert a conversion routine if necessary, then delegate to
//              a device_contiguous or device_block_contiguous copy as
//              appropriate
//
template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator copy(
	InputIterator first, InputIterator last,
	OutputIterator result,
	ecuda::pair<detail::host_type,detail::device_type> memory_types
);
// implementation makes calls to functions in the ecuda::impl::host_to_device namespace

namespace host_to_device {

//
// Source:      contiguous host memory
// Destination: contiguous device memory
// Value types: same
// On Device  : compile-time assertion
// On Host    : copy the host memory sequence to a contiguous block, and
//              call copy again
//
template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator copy(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	device_contiguous_iterator_tag
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return result; // never actually gets compiled, just here to satisfy nvcc
	#else
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type value_type;
	const typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last ); // get length of host sequence
	typedef typename ecuda::add_pointer<value_type>::type pointer;
	pointer dest = naked_cast<pointer>( result.operator->() );
	typedef typename ecuda::add_pointer<const value_type>::type const_pointer;
	const_pointer src = naked_cast<const_pointer>( first.operator->() );
	CUDA_CALL( cudaMemcpy<value_type>( dest, src, static_cast<std::size_t>(n), cudaMemcpyHostToDevice ) );
	ecuda::advance( result, static_cast<std::size_t>(n) );
	return result;
	#endif
}

//
// Source:      contiguous host memory
// Destination: disparate blocks of contiguous device memory
// Value types: same
// On Device  : compile-time assertion
// On Host    : call copy on each contiguous block of device memory
//
template<class InputIterator,typename T,typename P>
__HOST__ __DEVICE__ inline device_contiguous_block_iterator<T,P> copy(
	InputIterator first,
	InputIterator last,
	device_contiguous_block_iterator<T,P> result,
	device_contiguous_block_iterator_tag
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return result; // never actually gets compiled, just here to satisfy nvcc
	#else
	typedef device_contiguous_block_iterator<T,P> output_iterator_type;
	typedef typename ecuda::iterator_traits<output_iterator_type>::value_type value_type;

	{
		// if there is any leading data before the memory becomes regularly aligned
		// then copy it first
		const std::size_t leading = result.operator->().get_remaining_width();
		if( leading < result.operator->().get_width() ) {
			::ecuda::copy( first, first + leading, result.contiguous_begin() );
			::ecuda::advance( first, leading );
			::ecuda::advance( result, leading );
		}
	}

	{
		// memory is now guaranteed to be regularly aligned so we can use cudaMemcpy2D
		typedef typename ecuda::add_pointer<value_type>::type pointer;
		pointer dest = naked_cast<pointer>( result.operator->() );
		typedef typename ecuda::add_pointer<const value_type>::type const_pointer;
		const_pointer src = naked_cast<const_pointer>( first.operator->() );

		const size_t pitch = result.operator->().get_pitch();
		const std::size_t width = result.operator->().get_width();
		const std::size_t rows = ::ecuda::distance( first, last ) / width;
		CUDA_CALL( cudaMemcpy2D<value_type>( dest, pitch, src, width*sizeof(value_type), width, rows, cudaMemcpyHostToDevice ) );
		::ecuda::advance( first, width*rows );
		::ecuda::advance( result, width*rows );
	}

	{
		// if there is any trailing data where the memory ceases to be regularly aligned
		// then copy the rest of it
		const std::size_t trailing = ::ecuda::distance( first, last ) % result.operator->().get_width();
		if( trailing > 0 ) {
			::ecuda::copy( first, first + trailing, result.contiguous_begin() );
			::ecuda::advance( result, trailing );
		}
	}

	return result;
	/*
	typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	const std::size_t width = result.operator->().get_width();
	while( n > 0 ) {
		result = copy( first, first+width, result, device_contiguous_iterator_tag() );
		first += width;
		n -= width;
	}
	return result;
	*/
	#endif
}

} // namespace host_to_device

//
// Implementation only, function declaration and documentation above.
//
template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator copy(
	InputIterator first, InputIterator last,
	OutputIterator result,
	ecuda::pair<detail::host_type,detail::device_type> memory_types // host -> device
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	// is the device iterator contiguous?
	{
		// compile time check that device iterator traverses contiguous memory
		// or is at least comprised of a set of contiguous blocks
		const bool isSomeKindOfContiguous =
			ecuda::is_same<typename ecuda::iterator_traits<OutputIterator>::is_contiguous,ecuda::true_type>::value ||
			ecuda::is_same<typename ecuda::iterator_traits<OutputIterator>::iterator_category,ecuda::device_contiguous_block_iterator_tag>::value;
		ECUDA_STATIC_ASSERT(isSomeKindOfContiguous,CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_AS_DESTINATION_FOR_COPY);
	}
	typedef typename ecuda::iterator_traits<InputIterator>::value_type  T;
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type U;
	{
		// run-time check that the host iterator traverses contiguous memory
		// if not, make it so and call copy again
		const typename std::iterator_traits<InputIterator>::pointer pStart = first.operator->();
		const typename std::iterator_traits<InputIterator>::pointer pEnd   = last.operator->();
		if( (pEnd-pStart) != std::distance(first,last) ) {
			std::vector< U, host_allocator<U> > v( first, last ); // get type conversion here for free
			return host_to_device::copy( v.begin(), v.end(), result, typename ecuda::iterator_traits<OutputIterator>::iterator_category() );
		}
	}
	// compile-time check that the input and output types are the same
	// if not, do the conversion and call copy again
	const bool isSameType = ecuda::is_same<T,U>::value;
	if( !isSameType ) {
		std::vector< U, host_allocator<U> > v( first, last ); // type conversion
		return host_to_device::copy( v.begin(), v.end(), result, typename ecuda::iterator_traits<OutputIterator>::iterator_category() );
	} else {
		return host_to_device::copy( first, last, result, typename ecuda::iterator_traits<OutputIterator>::iterator_category() );
	}
	#endif
}

//
// Start of DEVICE to HOST implementations
//

namespace device_to_host {

//
// Source:      contiguous device memory
// Destination: contiguous host memory
// Value types: same
// On Device  : compile-time assertion
// On Host    : call ecuda::cudaMemcpy to copy sequence
//
template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator copy(
	InputIterator first,
	InputIterator last,
	OutputIterator result,
	device_contiguous_iterator_tag // contiguous
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type value_type;
	typedef typename ecuda::add_pointer<const value_type>::type           src_pointer_type;
	typedef typename ecuda::add_pointer<value_type>::type                 dest_pointer_type;
	src_pointer_type src   = naked_cast<src_pointer_type>( first.operator->() );
	dest_pointer_type dest = naked_cast<dest_pointer_type>( result.operator->() );
	const typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
	CUDA_CALL( cudaMemcpy<value_type>( dest, src, static_cast<std::size_t>(n), cudaMemcpyDeviceToHost ) );
	ecuda::advance( result, static_cast<std::size_t>(n) );
	return result;
	#endif
}

//
// Source:      blocks of contiguous device memory
// Destination: contiguous host memory
// Value types: same
// On Device  : compile-time assertion
// On Host    : call copy on each contiguous block of device memory
//
template<typename T,typename P,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator copy(
	device_contiguous_block_iterator<T,P> first,
	device_contiguous_block_iterator<T,P> last,
	OutputIterator result,
	device_contiguous_block_iterator_tag // contiguous blocks
)
{
	#ifdef __CUDA_ARCH__
	ECUDA_STATIC_ASSERT(__CUDA_ARCH__,CANNOT_CALL_COPY_ON_HOST_MEMORY_INSIDE_DEVICE_CODE);
	return result; // can never be called from device code, dummy return to satisfy nvcc
	#else
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type value_type;
	typedef device_contiguous_block_iterator<T,P> input_iterator_type;

	{
		// if there is any leading data before the memory becomes regularly aligned
		// then copy it first
		const std::size_t leading = first.operator->().get_remaining_width();
		if( leading < first.operator->().get_width() ) { // leading with partial row
			typename input_iterator_type::contiguous_iterator first2 = first.contiguous_begin();
			result = ::ecuda::copy( first2, first2 + leading, result );
			::ecuda::advance( first, leading );
		}
	}

	{
		// memory is now guaranteed to be regularly aligned so we can use cudaMemcpy2D
		typedef typename ecuda::add_pointer<value_type>::type pointer;
		pointer dest = naked_cast<pointer>( result.operator->() );
		typedef typename ecuda::add_pointer<const value_type>::type const_pointer;
		const_pointer src = naked_cast<const_pointer>( first.operator->() );

		const size_t pitch = first.operator->().get_pitch();
		const std::size_t width = first.operator->().get_width();
		const std::size_t rows = ::ecuda::distance( first, last ) / width;

		CUDA_CALL( cudaMemcpy2D<value_type>( dest, width*sizeof(value_type), src, pitch, width, rows, cudaMemcpyDeviceToHost ) );
		::ecuda::advance( first, width*rows );
		::ecuda::advance( result, width*rows );
	}

	{
		// if there is any trailing data where the memory ceases to be regularly aligned
		// then copy the rest of it
		const std::size_t trailing = ::ecuda::distance( first, last ) % first.operator->().get_width();
		if( trailing > 0 ) {
			typename input_iterator_type::contiguous_iterator first2 = first.contiguous_begin();
			result = ::ecuda::copy( first2, first2 + trailing, result );
		}
	}

	return result;
	/*
	for( ; first != last; result += first.operator->().get_remaining_width(), first += first.operator->().get_remaining_width() ) {
		::ecuda::impl::device_to_host::copy( first, first+first.operator->().get_remaining_width(), result, device_contiguous_iterator_tag() );
	}
	return result;
	*/
	#endif
}

} // namespace device_to_host

//
// Source:      any device memory
// Destination: any host memory
// Value types: any
// On Device  : compile-time assertion
// On Host    : determine contiguity and data types of device memory, generate
//              a compile-time error if called from the host and device memory
//              is non-contiguous, perform a compile-time check for type equality
//              and insert a conversion routine if necessary, then delegate to
//              a device_contiguous or device_block_contiguous copy as
//              appropriate
//
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
	{
		// compile time check that device iterator traverses contiguous memory
		// or is at least comprised of a set of contiguous blocks
		const bool isSomeKindOfContiguous =
			ecuda::is_same<typename ecuda::iterator_traits<InputIterator>::is_contiguous,ecuda::true_type>::value ||
			ecuda::is_same<typename ecuda::iterator_traits<InputIterator>::iterator_category,ecuda::device_contiguous_block_iterator_tag>::value;
		ECUDA_STATIC_ASSERT(isSomeKindOfContiguous,CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_AS_SOURCE_FOR_COPY);
	}
	typedef typename ecuda::iterator_traits<InputIterator>::value_type  T;
	typedef typename ecuda::iterator_traits<OutputIterator>::value_type U;
	{
		// run time check that host iterator traverses contiguous memory
		// if not, create a temporary container that is and re-call copy
		typename ecuda::iterator_traits<InputIterator>::pointer pStart = first.operator->();
		typename ecuda::iterator_traits<InputIterator>::pointer pEnd   = last.operator->();
		typename ecuda::iterator_traits<InputIterator>::difference_type n = ecuda::distance( first, last );
		if( ( pEnd-pStart ) != n ) {
			typedef typename ecuda::remove_const<T>::type T2; // need to strip source const otherwise this can't act as staging
			std::vector< T2, host_allocator<T2> > v( ecuda::distance( first, last ) );
			::ecuda::copy( first, last, v.begin() );
			return ::ecuda::copy( v.begin(), v.end(), result ); // get type conversion if needed, should resolve directly to std::copy
		}
	}
	// compile-time check that the input and output types are the same
	// if not, provide a temp destination of the correct type, copy
	// there temporarily, and then do a host-to-host copy that does
	// the type conversion
	const bool isSameType = ecuda::is_same<T,U>::value;
	if( !isSameType ) {
		typedef typename ecuda::remove_const<T>::type T2; // need to strip source const otherwise this can't act as staging
		std::vector< T2, host_allocator<T2> > v( ecuda::distance( first, last ) );
		device_to_host::copy( first, last, v.begin(), typename ecuda::iterator_traits<InputIterator>::iterator_category() );
		return ::ecuda::copy( v.begin(), v.end(), result ); // type conversion occurs here, should resolve directly to std::copy
	} else {
		return device_to_host::copy( first, last, result, typename ecuda::iterator_traits<InputIterator>::iterator_category() );
	}
	#endif
}

//
// Start of HOST to HOST implementations
//

//
// Source:      any host memory
// Destination: any host memory
// Value types: any
// On Device  : compile-time assertion
// On Host    : just delegate to std::copy
//
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
	while( first != last ) {
		*result = *first;
		++first;
		++result;
	}
	return result;
//	return std::copy( first, last, result );
	#endif
}

} // namespace impl
/// \endcond

//
// Entry point of the ecuda::copy function.
//

ECUDA_SUPRESS_EXEC_WARNINGS
template<class InputIterator,class OutputIterator>
__HOST__ __DEVICE__ inline OutputIterator copy( InputIterator first, InputIterator last, OutputIterator result )
{
	typedef typename ecuda::iterator_traits<InputIterator>::is_device_iterator  input_memory_type;
	typedef typename ecuda::iterator_traits<OutputIterator>::is_device_iterator output_memory_type;
	return impl::copy( first, last, result, ecuda::pair<input_memory_type,output_memory_type>() );
}

} // namespace ecuda

#endif
