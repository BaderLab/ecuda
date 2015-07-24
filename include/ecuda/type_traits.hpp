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
// type_traits.hpp
//
// Metaprogramming templates to get type traits at compile-time.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_TYPE_TRAITS_HPP
#define ECUDA_TYPE_TRAITS_HPP

/// \cond DEVELOPER_DOCUMENTATION

#ifdef __CPP11_SUPPORTED__
#include <type_traits>
#else
namespace std {

template<typename T> struct remove_const { typedef T type; };
template<typename T> struct remove_const<const T> { typedef T type; };

template<typename T,typename U> struct is_same { enum { value = 0 }; };
template<typename T> struct is_same<T,T> { enum { value = 1 }; };

template<typename T> struct remove_reference { typedef T type; };
template<typename T> struct remove_reference<T&> { typedef T type; };
#ifdef __CPP11_SUPPORTED__
template<typename T> struct remove_reference<T&&> { typedef T type; };
#endif

template<typename T> struct add_lvalue_reference { typedef T& type; };
template<typename T> struct add_lvalue_reference<T&> { typedef T& type; };

template<bool B,typename T,typename F> struct conditional { typedef T type; };
template<typename T,typename F> struct conditional<false,T,F> { typedef F type; };

template<bool B,typename T=void> struct enable_if {};
template<typename T> struct enable_if<true,T> { typedef T type; };

} // namespace std
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

template<typename T> class naked_ptr; // forward declaration
template<typename T,typename U> class padded_ptr; // forward declaration
template<typename T> class shared_ptr; // forward declaration
template<typename T,typename U> class striding_ptr; // forward declaration
template<typename T,typename U> class unique_ptr; // forward declaration


template<typename T> struct dereference;
template<typename T> struct dereference<T*> { typedef T& type; };
template<typename T> struct dereference<T* const> { typedef const T& type; };
template<typename T> struct reference {
	typedef T* pointer_type;
	typedef T& reference_type;
	typedef T element_type;
};

template<typename T> struct type_traits { typedef T* pointer; };
template<typename T> struct type_traits<const T> { typedef const T* pointer; };

//template<typename T>
//struct pointer_traits {
//	typedef T* pointer;
//};
//
//template<typename T>
//struct pointer_traits<const T> {
//	typedef const T* pointer;
//};

template<typename T> struct pointer_traits;

template<typename T>
struct pointer_traits<T*> {
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T* naked_pointer;
	typedef T* modifiable_pointer;
	typedef char* char_pointer;
	__host__ __device__ inline naked_pointer undress( pointer ptr ) const { return ptr; }
	__host__ __device__ inline modifiable_pointer increment( pointer ptr, const int x ) const { return ptr + x; }
	__host__ __device__ inline modifiable_pointer cast_to_modifiable( pointer ptr ) const { return ptr; }
};

template<typename T>
struct pointer_traits<const T*> {
	typedef const T* pointer;
	typedef const T* const_pointer;
	typedef const T* naked_pointer;
	typedef const T* modifiable_pointer;
	typedef const char* char_pointer;
	__host__ __device__ inline naked_pointer undress( pointer ptr ) const { return ptr; }
	__host__ __device__ inline modifiable_pointer increment( pointer ptr, const int x ) const { return ptr + x; }
	__host__ __device__ inline modifiable_pointer cast_to_modifiable( pointer ptr ) const { return ptr; }
};

template<typename T>
struct pointer_traits< naked_ptr<T> > {
	typedef naked_ptr<T> pointer;
	typedef naked_ptr<const T> const_pointer;
	typedef typename naked_ptr<T>::pointer naked_pointer;
	typedef naked_ptr<T> modifiable_pointer;
	typedef char* char_pointer;
	__host__ __device__ inline naked_pointer undress( const pointer& ptr ) const { return ptr.get(); }
	__host__ __device__ inline modifiable_pointer increment( const pointer& ptr, const int x ) const { return ptr + x; }
	__host__ __device__ inline modifiable_pointer& cast_to_modifiable( pointer& ptr ) const { return ptr; }
};

template<typename T>
struct pointer_traits< const naked_ptr<T> > {
	typedef const naked_ptr<T> pointer;
	typedef const naked_ptr<const T> const_pointer;
	typedef typename naked_ptr<T>::pointer naked_pointer;
	typedef naked_pointer modifiable_pointer;
	typedef typename pointer_traits<naked_pointer>::char_pointer char_pointer;
	__host__ __device__ inline naked_pointer undress( const pointer& ptr ) const { return ptr.get(); }
	__host__ __device__ inline modifiable_pointer increment( const pointer& ptr, const int x ) const { return ptr.get() + x; }
	__host__ __device__ inline modifiable_pointer cast_to_modifiable( pointer& ptr ) const { return ptr.get(); }
};

template<typename T,typename U>
struct pointer_traits< unique_ptr<T,U> > {
	typedef unique_ptr<T,U> pointer;
	typedef unique_ptr<const T,U> const_pointer;
	typedef typename unique_ptr<T,U>::pointer naked_pointer;
	typedef naked_pointer modifiable_pointer;
	typedef typename pointer_traits<naked_pointer>::char_pointer char_pointer;
	__host__ __device__ inline naked_pointer undress( const pointer& ptr ) const { return ptr.get(); }
	__host__ __device__ inline modifiable_pointer increment( pointer& ptr, const int x ) const { return ptr.get() + x; }
	__host__ __device__ inline modifiable_pointer cast_to_modifiable( pointer& ptr ) const { return undress(ptr); }
};

template<typename T,typename U>
struct pointer_traits< const unique_ptr<T,U> > {
	typedef const unique_ptr<T,U> pointer;
	typedef const unique_ptr<const T,U> const_pointer;
	typedef typename unique_ptr<T,U>::pointer naked_pointer;
	typedef naked_pointer modifiable_pointer;
	typedef typename pointer_traits<naked_pointer>::char_pointer char_pointer;
	__host__ __device__ inline naked_pointer undress( const pointer& ptr ) const { return ptr.get(); }
	__host__ __device__ inline modifiable_pointer increment( pointer& ptr, const int x ) const { return ptr.get() + x; }
	__host__ __device__ inline modifiable_pointer cast_to_modifiable( pointer& ptr ) const { return undress(ptr); }
};

template<typename T>
struct pointer_traits< shared_ptr<T> > {
	typedef shared_ptr<T> pointer;
	typedef shared_ptr<const T> const_pointer;
	typedef T* naked_pointer;
	typedef naked_pointer modifiable_pointer;
	typedef typename pointer_traits<naked_pointer>::char_pointer char_pointer;
	__host__ __device__ inline naked_pointer undress( const pointer& ptr ) const { return ptr.get(); }
	__host__ __device__ inline modifiable_pointer increment( const shared_ptr<T> ptr, const int x ) const { return ptr.get() + x; }
	__host__ __device__ inline modifiable_pointer cast_to_modifiable( pointer ptr ) const { return undress(ptr); }
};

template<typename T>
struct pointer_traits< const shared_ptr<T> > {
	typedef const shared_ptr<T> pointer;
	typedef const shared_ptr<const T> const_pointer;
	typedef typename shared_ptr<T>::pointer naked_pointer;
	typedef typename pointer_traits<naked_pointer>::char_pointer char_pointer;
	__host__ __device__ inline naked_pointer undress( const pointer& ptr ) const { return ptr.get(); }
};

template<typename T,typename U>
struct pointer_traits< padded_ptr<T,U> > {
	typedef padded_ptr<T,U> pointer;
	typedef padded_ptr<const T,typename pointer_traits<U>::const_pointer> const_pointer;
	typedef typename pointer_traits<U>::naked_pointer naked_pointer;
	typedef padded_ptr<T,typename pointer_traits<U>::modifiable_pointer> modifiable_pointer;
	typedef typename pointer_traits<U>::char_pointer char_pointer;
	__host__ __device__ inline naked_pointer undress( const pointer& ptr ) const { return pointer_traits<U>().undress( ptr.get() ); }
	__host__ __device__ inline modifiable_pointer increment( const pointer& ptr, const int x ) const { return cast_to_modifiable(ptr)+x; }
	__host__ __device__ inline modifiable_pointer cast_to_modifiable( const pointer& ptr ) const {
		typename pointer_traits<U>::modifiable_pointer mp1 = pointer_traits<U>().cast_to_modifiable( ptr.get_edge() );
		typename pointer_traits<U>::modifiable_pointer mp2 = pointer_traits<U>().cast_to_modifiable( ptr.get() );
		return modifiable_pointer( mp1, ptr.get_pitch(), ptr.get_width(), mp2 );
	}
};

template<typename T,typename U>
struct pointer_traits< const padded_ptr<T,U> > {
	typedef const padded_ptr<T,U> pointer;
	typedef const padded_ptr<const T,typename pointer_traits<U>::const_pointer> const_pointer;
	typedef typename pointer_traits<U>::naked_pointer naked_pointer;
	typedef padded_ptr<const T,typename pointer_traits<typename pointer_traits<U>::const_pointer>::modifiable_pointer> modifiable_pointer;
	typedef typename pointer_traits<U>::char_pointer char_pointer;
	__host__ __device__ inline naked_pointer undress( const pointer& ptr ) const { return pointer_traits<U>().undress( ptr.get() ); }
	__host__ __device__ inline modifiable_pointer increment( const pointer& ptr, const int x ) const { return cast_to_modifiable(ptr)+x; }
	__host__ __device__ inline modifiable_pointer cast_to_modifiable( const pointer& ptr ) const {
		typename pointer_traits<U>::modifiable_pointer mp1 = pointer_traits<U>().cast_to_modifiable( ptr.get_edge() );
		typename pointer_traits<U>::modifiable_pointer mp2 = pointer_traits<U>().cast_to_modifiable( ptr.get() );
		return modifiable_pointer( mp1, ptr.get_pitch(), ptr.get_width(), mp2 );
	}
};

template<typename T,typename U>
struct pointer_traits< striding_ptr<T,U> > {
	typedef striding_ptr<T,U> pointer;
	typedef striding_ptr<const T,typename pointer_traits<U>::const_pointer> const_pointer;
	typedef typename pointer_traits<U>::naked_pointer naked_pointer;
	typedef striding_ptr<T,typename pointer_traits<U>::modifiable_pointer> modifiable_pointer;
	typedef typename pointer_traits<U>::char_pointer char_pointer;
	__host__ __device__ inline naked_pointer undress( const pointer& ptr ) const { return pointer_traits<U>().undress( ptr.get() ); }
	__host__ __device__ inline modifiable_pointer cast_to_modifiable( const pointer& ptr ) const { return modifiable_pointer( pointer_traits<U>().cast_to_modifiable( ptr.get() ), ptr.get_stride() ); }
	__host__ __device__ inline modifiable_pointer increment( const pointer& ptr, const int x ) const { return cast_to_modifiable(ptr)+x; }
};

template<typename T,typename U>
struct pointer_traits< const striding_ptr<T,U> > {
	typedef const striding_ptr<T,U> pointer;
	typedef const striding_ptr<const T,typename pointer_traits<U>::const_pointer> const_pointer;
	typedef typename pointer_traits<U>::naked_pointer naked_pointer;
	typedef striding_ptr<const T,typename pointer_traits<typename pointer_traits<U>::const_pointer>::modifiable_pointer> modifiable_pointer;
	typedef typename pointer_traits<U>::char_pointer char_pointer;
	__host__ __device__ inline naked_pointer undress( const pointer& ptr ) const { return pointer_traits<U>().undress( ptr.get() ); }
	__host__ __device__ inline modifiable_pointer cast_to_modifiable( const pointer& ptr ) const { return modifiable_pointer( pointer_traits<U>().cast_to_modifiable( ptr.get() ), ptr.get_stride() ); }
	__host__ __device__ inline modifiable_pointer increment( const pointer& ptr, const int x ) const { return cast_to_modifiable(ptr)+x; }
};

} // namespace ecuda

/// \endcond

#endif
