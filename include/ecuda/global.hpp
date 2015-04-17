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
// General functions for use with CUDA.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_GLOBAL_HPP
#define ECUDA_GLOBAL_HPP

#include <stdexcept>
#include <sstream>

// Alias for detecting C++11 support because GCC 4.6 screws up the __cplusplus flag
#if __cplusplus >= 201103L || defined(__GXX_EXPERIMENTAL_CXX0X__)
#define __CPP11_SUPPORTED__
#endif

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
#define EXCEPTION_MSG(x) "" __FILE__ ":" S__LINE__ " " x

///
/// Macro that performs a check for any outstanding CUDA errors.  This macro
/// should be declared after any CUDA API calls that do not return an error code
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

/** Allow noexcept and constexpr if C++11 supported. */
#ifdef __CPP11_SUPPORTED__
#define __NOEXCEPT__ noexcept
#define __CONSTEXPR__ constexpr
#else
#define __NOEXCEPT__
#define __CONSTEXPR__
#endif

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

#endif
