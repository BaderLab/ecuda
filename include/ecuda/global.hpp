//----------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
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
#define CUDA_CHECK_ERRORS do { cudaError_t error = cudaGetLastError(); if( error != cudaSuccess ) throw std::runtime_error(std::string(cudaGetErrorString(error))); } while(0);

#define DEVICE __device__
#define HOST __host__

/** Replace nullptr with NULL if nvcc still doesn't support C++11. */
#ifndef __CPP11_SUPPORTED__
#define nullptr NULL
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
} // namespace ecuda

#endif
