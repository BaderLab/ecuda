/*
Copyright (c) 2014-2016, Scott Zuyderduyn
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
// global.hpp
//
// Global defines and macros.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_GLOBAL_HPP
#define ECUDA_GLOBAL_HPP

#include <stdexcept>
#include <sstream>

///
/// \cond DEVELOPER_DOCUMENTATION
///
/// \endcond
///

// idea taken from the VTK-m project (https://gitlab.kitware.com/vtk/vtk-m)
// to suppress annoying warnings from the compiler about calling __host__
// code from a __host__ __device__ function
#ifdef __CUDACC__
#define ECUDA_SUPPRESS_HD_WARNINGS \
	#pragma hd_warning_disable
#else
#define ECUDA_SUPPRESS_HD_WARNINGS
#endif

#include "impl/host_emulation.hpp" // host-only replacements of CUDA C API functions
#include "cuda_error.hpp"          // specialized std::exception for ecuda/CUDA runtime errors

// Generic check for C++11 support
#if __cplusplus >= 201103L || defined(__GXX_EXPERIMENTAL_CXX0X__) // latter check works for GCC 4.6 because it has a bad __cplusplus flag
#define ECUDA_CPP11_AVAILABLE
#endif
// Windows check for C++11 support (Visual Studio 2013 and greater)
#if defined(_MSC_VER) && _MSC_VER >= 1800 // Visual Studio 2013
#define ECUDA_CPP11_AVAILABLE
#endif

///
/// Macro function that captures a CUDA error code and then does something
/// with it.  All calls to functions in the CUDA API that return an error code
/// should use this.
///
#ifdef __CUDACC__
// Macro function currently throws an ecuda::cuda_error exception containing a
// description of the problem error code.
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { std::ostringstream oss; oss << __FILE__; oss << ":"; oss << __LINE__; oss << " "; oss << cudaGetErrorString(cudaGetLastError()); throw ::ecuda::cuda_error(x,oss.str()); /*std::runtime_error(oss.str());*/ }} while(0);
#else
// cannot do CUDA calls when emulating with host only
#define CUDA_CALL(x) x
#endif

#define S(x) #x
#define S_(x) S(x)
#define S__LINE__ S_(__LINE__)
///
/// String wrapper that adds the source file and line to a given error message.
///
#define EXCEPTION_MSG(x) "" __FILE__ ":" S__LINE__ " " x

///
/// Macro that performs a check for any outstanding CUDA errors.  This macro
/// should be declared after any CUDA API calls that do not return an error code
/// (e.g. after calling kernel functions). Calling this when a CUDA API call
/// has not been made is safe.
///
#ifdef __CUDACC__
#define CUDA_CHECK_ERRORS() do { cudaError_t error = cudaGetLastError(); if( error != cudaSuccess ) throw ::ecuda::cuda_error(error,std::string(cudaGetErrorString(error))); } while(0);
#else
// cannot check CUDA errors when emulating with host only
#define CUDA_CHECK_ERRORS() do {} while(0);
#endif

///
/// Macro that calls a CUDA kernel function, waits for completion, and throws
/// an ecuda::cuda_error exception if any errors are reported by cudaGetLastError().
///
#ifdef __CUDACC__
#define CUDA_CALL_KERNEL_AND_WAIT(...) do {\
		__VA_ARGS__;\
		{ cudaError_t error = cudaGetLastError(); if( error != cudaSuccess ) throw ::ecuda::cuda_error(error,std::string(cudaGetErrorString(error))); }\
		cudaDeviceSynchronize();\
		{ cudaError_t error = cudaGetLastError(); if( error != cudaSuccess ) throw ::ecuda::cuda_error(error,std::string(cudaGetErrorString(error))); }\
	} while(0);
#else
// cannot do CUDA calls when emulating with host only
#define CUDA_CALL_KERNEL_AND_WAIT(...) do {\
		__VA_ARGS__;\
	} while( 0 );
#endif

/** Replace nullptr with NULL if nvcc still doesn't support C++11. */
#ifndef ECUDA_CPP11_AVAILABLE
#define nullptr NULL
#endif

/** Allow noexcept and constexpr if C++11 supported. */
#ifdef ECUDA_CPP11_AVAILABLE
#if defined(_MSC_VER) && _MSC_VER == 1800 // Visual Studio 2013 has only partial C++11 support and doesn't know these
#define __NOEXCEPT__
#define __CONSTEXPR__
#else
#define __NOEXCEPT__ noexcept
#define ECUDA_NOEXCEPT_KEYWORD_ENABLED
#define __CONSTEXPR__ constexpr
#define ECUDA_CONSTEXPR_KEYWORD_ENABLED
#endif
#else
#define __NOEXCEPT__
#define __CONSTEXPR__
#endif

#ifdef __CUDACC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#define __CONSTANT__ __constant__
#else
// strip all __host__ and __device__ declarations when using host only
#define __HOST__
#define __DEVICE__
#define __CONSTANT__
#endif

//
// Quick implementation of compile-time assertions. If C++11 is available, then
// just use the new static_assert keyword.
//
// This approach was borrowed from the Eigen linear algebra template library
// (http://eigen.tuxfamily.org).
//
#ifdef ECUDA_CPP11_AVAILABLE
#define ECUDA_STATIC_ASSERT(x,msg) static_assert(x,#msg)
#else

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<bool condition> struct static_assertion {};
template<> struct static_assertion<true>
{
	enum {
		CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_AS_SOURCE_FOR_COPY,
		CANNOT_USE_NONCONTIGUOUS_DEVICE_ITERATOR_AS_DESTINATION_FOR_COPY,
		CANNOT_FILL_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_ITERATOR,
		CANNOT_LEXICOGRAPHICALLY_COMPARE_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_MEMORY,
		CANNOT_FIND_MAX_ELEMENT_IN_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_MEMORY,
		CANNOT_REVERSE_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_MEMORY,
		CANNOT_ACCUMULATE_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_MEMORY,
		CANNOT_CALCULATE_DISTANCE_OF_NONCONTIGUOUS_DEVICE_ITERATOR_FROM_HOST_CODE
	};
};

} // namespace impl
/// \endcond

} // namespace ecuda

#define ECUDA_STATIC_ASSERT(x,msg) if(ecuda::impl::static_assertion<static_cast<bool>(x)>::msg) {}

#endif // ECUDA_CPP11_AVAILABLE

#endif
