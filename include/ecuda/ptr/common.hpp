/*
Copyright (c) 2015, Scott Zuyderduyn
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
// ptr/common.hpp
//
// Classes and helper functions that are needed by smart and specialized
// pointers.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------
#ifndef ECUDA_PTR_COMMON_HPP
#define ECUDA_PTR_COMMON_HPP

#include "../global.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace detail {

// this is hacky structure that takes any pointer (const or not)
// and casts it to void* so it can be used by the deleter dispose() method
// and other places that call cudaFree
template<typename T> struct void_cast;
template<typename T> struct void_cast<const T*> { __HOST__ __DEVICE__ inline void* operator()( const T* ptr ) { return reinterpret_cast<void*>( const_cast<T*>(ptr) ); } };
template<typename T> struct void_cast           { __HOST__ __DEVICE__ inline void* operator()( T ptr        ) { return reinterpret_cast<void*>(ptr); } };

} // namespace detail
/// \endcond

///
/// \brief The default destruction policy used by smart pointers to device memory.
///
/// The CUDA API function cudaFree() is used to deallocate memory.
///
template<typename T>
struct default_device_delete {

	///
	/// \brief Constructs an ecuda::default_device_delete object.
	///
	__HOST__ __DEVICE__ default_device_delete() ECUDA__NOEXCEPT {}

	///
	/// \brief Constructs an ecuda::default_device_delete object from another one.
	///
	/// This constructor will only participate in overload resolution
	/// if U* is implicitly convertible to T*.
	///
	template<typename U> __HOST__ __DEVICE__ default_device_delete( const default_device_delete<U>& src ) ECUDA__NOEXCEPT {}

	///
	/// \brief Calls cudaFree() on a pointer.
	/// \param ptr an object or array to delete
	///
	__HOST__ __DEVICE__ inline void operator()( T* ptr ) const {
		#ifdef __CUDA_ARCH__
		//ptr = NULL;
		#else
		#ifndef __CUDACC__
		delete [] reinterpret_cast<char*>(ptr); // hacky as hell but this should be valid for most test cases
		#else
		if( ptr ) cudaFree( detail::void_cast<T*>()(ptr) );
		//if( ptr ) cudaFree(ptr);
		#endif // __CUDACC__
		#endif
	}

};

///
/// \brief The default destruction policy used by smart pointers to page-locked host memory.
///
/// The CUDA API function cudaFreeHost() is used to deallocate memory.
///
template<typename T>
struct default_host_delete {
	__HOST__ __DEVICE__ ECUDA__CONSTEXPR default_host_delete() ECUDA__NOEXCEPT {}
	template<typename U> __HOST__ __DEVICE__ default_host_delete( const default_host_delete<U>& src ) ECUDA__NOEXCEPT {}
	__HOST__ __DEVICE__ inline void operator()( T* ptr ) const {
		#ifdef __CUDA_ARCH__
		#else
		#ifndef __CUDACC__
		delete [] reinterpret_cast<char*>(ptr);
		#else
		if( ptr ) cudaFreeHost( detail::void_cast<T*>()(ptr) );
		#endif
		#endif
	}
};

} // namespace ecuda


#endif
