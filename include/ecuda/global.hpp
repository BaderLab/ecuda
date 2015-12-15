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
// global.hpp
<<<<<<< HEAD
// General functions for use with CUDA.
=======
//
// Global defines and macros.
>>>>>>> ecuda2/master
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_GLOBAL_HPP
#define ECUDA_GLOBAL_HPP

#include <stdexcept>
#include <sstream>

<<<<<<< HEAD
=======
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

#include "impl/host_emulation.hpp"
#include "cuda_error.hpp"

>>>>>>> ecuda2/master
// Alias for detecting C++11 support because GCC 4.6 screws up the __cplusplus flag
#if __cplusplus >= 201103L || defined(__GXX_EXPERIMENTAL_CXX0X__)
#define __CPP11_SUPPORTED__
#endif

<<<<<<< HEAD
namespace ecuda {

///
/// \brief Exception for CUDA API cudaError_t errors.
///
class cuda_error : public std::runtime_error {
private:
	cudaError_t errorType;
public:
	explicit cuda_error( cudaError_t errorType, const std::string& what_arg ) : std::runtime_error( what_arg ), errorType(errorType) {}
	explicit cuda_error( cudaError_t errorType, const char* what_arg ) : std::runtime_error( what_arg ), errorType(errorType) {}
	inline cudaError_t get_cuda_error_type() const { return errorType; }
};

} // namespace ecuda

///
/// Function wrapper that capture and throw an exception on error.  All calls
/// to functions in the CUDA API that return an error code should use this.
///
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { std::ostringstream oss; oss << __FILE__; oss << ":"; oss << __LINE__; oss << " "; oss << cudaGetErrorString(cudaGetLastError()); throw ::ecuda::cuda_error(x,oss.str()); /*std::runtime_error(oss.str());*/ }} while(0);

///
/// String wrapper that adds the source file and line to a given error message.
///
#define S(x) #x
#define S_(x) S(x)
#define S__LINE__ S_(__LINE__)
=======
///
/// Macro function that captures a CUDA error code and then does something
/// with it.  All calls to functions in the CUDA API that return an error code
/// should use this.
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
>>>>>>> ecuda2/master
#define EXCEPTION_MSG(x) "" __FILE__ ":" S__LINE__ " " x

///
/// Macro that performs a check for any outstanding CUDA errors.  This macro
/// should be declared after any CUDA API calls that do not return an error code
<<<<<<< HEAD
/// (e.g. after calling kernel functions).
///
#define CUDA_CHECK_ERRORS() do { cudaError_t error = cudaGetLastError(); if( error != cudaSuccess ) throw ::ecuda::cuda_error(error,std::string(cudaGetErrorString(error))); } while(0);


#define DEVICE __device__
#define HOST __host__

/** Replace nullptr with NULL if nvcc still doesn't support C++11. */
#ifndef __CPP11_SUPPORTED__
#ifndef nullptr
#define nullptr NULL
#endif
#endif
=======
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
	} while(0);
#endif

/** Replace nullptr with NULL if nvcc still doesn't support C++11. */
#ifndef __CPP11_SUPPORTED__
#define nullptr NULL
#endif
>>>>>>> ecuda2/master

/** Allow noexcept and constexpr if C++11 supported. */
#ifdef __CPP11_SUPPORTED__
#define __NOEXCEPT__ noexcept
#define __CONSTEXPR__ constexpr
#else
#define __NOEXCEPT__
#define __CONSTEXPR__
#endif

<<<<<<< HEAD
/// \cond DEVELOPER_DOCUMENTATION

///
/// Metaprogramming trick to get the type of a dereferenced pointer. Helpful
/// for implementing the strategy required to make const/non-const iterators.
/// C++11 type_traits would allow this to be done inline, but nvcc currently
/// lacks C++11 support. Example:
///
///   typedef int* pointer;
///   ecuda::dereference<pointer>::type value; // equivalent to int& value;
///
namespace ecuda {
	template<typename T> struct dereference;
	template<typename T> struct dereference<T*> { typedef T& type; };
	template<typename T> struct dereference<T* const> { typedef const T& type; };
	template<typename T> struct reference {
		typedef T* pointer_type;
		typedef T& reference_type;
		typedef T element_type;
	};
} // namespace ecuda

/// \endcond

=======
#ifdef __CUDACC__
// strip all __host__ and __device__ declarations when using host only
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif

//
// Quick implementation of compile-time assertions. If C++11 is available, then
// just use the new static_assert keyword.
//
// This approach was borrowed from the Eigen linear algebra template library
// (http://eigen.tuxfamily.org).
//
#ifdef __CPP11_SUPPORTED__
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
#endif

>>>>>>> ecuda2/master
#endif
