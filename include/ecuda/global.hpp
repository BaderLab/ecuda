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

/** Alias for detecting C++11 support because GCC 4.6 screws up the __cplusplus flag */
#if __cplusplus >= 201103L || defined(__GXX_EXPERIMENTAL_CXX0X__)
#define __CPP11_SUPPORTED__
#endif

///
/// Function wrapper that capture and throw an exception on error.  All calls
/// to functions in the CUDA API that return an error code should use this.
///
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { std::ostringstream oss; oss << __FILE__; oss << ":"; oss << __LINE__; oss << " "; oss << cudaGetErrorString(cudaGetLastError()); throw std::runtime_error(oss.str()); }} while(0);

///
/// Macro that performs a check for any outstanding CUDA errors.  This macro
/// should be declared after any CUDA API calls that do not return an error code
/// (e.g. after calling kernel functions).
///
#define CUDA_CHECK_ERRORS() do { cudaError_t error = cudaGetLastError(); if( error != cudaSuccess ) throw std::runtime_error(std::string(cudaGetErrorString(error))); } while(0);

#define DEVICE __device__
#define HOST __host__

/** Replace nullptr with NULL if nvcc still doesn't support C++11. */
#ifndef __CPP11_SUPPORTED__
#define nullptr NULL
#endif

/** Allow noexcept and constexpr if C++11 supported. */
#ifdef __CPP11_SUPPORTED__
#define __NOEXCEPT__ noexcept
#define __CONSTEXPR__ constexpr
#else
#define __NOEXCEPT__
#define __CONSTEXPR__
#endif

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

#endif
