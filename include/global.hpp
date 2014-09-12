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

/*
#define CUDA_CHECK_ERRORS
do {
	std::ostringstream oss;
	while(1) {
		cudaError_t error = cudaGetLastError();
		if( error != cudaSuccess ) oss << std::string(cudaGetErrorString(error)); else break;
	}
} while(0);
*/

#endif

