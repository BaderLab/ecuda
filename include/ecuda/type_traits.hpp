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
// type_traits.hpp
//
// Metaprogramming templates to get type traits at compile-time.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_TYPE_TRAITS_HPP
#define ECUDA_TYPE_TRAITS_HPP

#include "global.hpp"

/// \cond DEVELOPER_DOCUMENTATION

#ifdef ECUDA_CPP11_AVAILABLE
#include <type_traits>

namespace ecuda {

template<typename T> using add_const = std::add_const<T>;
template<typename T> using remove_const = std::remove_const<T>;
template<typename T,typename U> using is_same = std::is_same<T,U>;
template<typename T> using add_lvalue_reference = std::add_lvalue_reference<T>;

//template<typename T> struct add_const : std::add_const<T> { typedef std::add_const<T>::type type; };
//template<typename T> struct add_const<T&> : std::add_const<T&> { typedef std::add_const<T&>::type type; };
//template<typename T> struct remove_const : std::remove_const<T> { typedef std::remove_const<T>::type type; };
//template<typename T> struct remove_const<const T> : std::remove_const<const T> { typedef std::remove_const<const T>::type type; };
//template<typename T,typename U> struct is_same      { enum { value = 0 }; };
//template<typename T>            struct is_same<T,T> { enum { value = 1 }; };

template<bool B,typename T> using enable_if = std::enable_if<B,T>;

typedef std::true_type  true_type;
typedef std::false_type false_type;

template<typename T> using is_integral = std::is_integral<T>;
/*
template<typename T> struct is_integral           { typedef false_type type; };
template<> struct is_integral<bool>               { typedef true_type type; };
template<> struct is_integral<char>               { typedef true_type type; };
template<> struct is_integral<signed char>        { typedef true_type type; };
template<> struct is_integral<unsigned char>      { typedef true_type type; };
#ifdef _GLIBCXX_USE_WCHAR_T
template<> struct is_integral<wchar_t>            { typedef true_type type; };
#endif
template<> struct is_integral<char16_t>           { typedef true_type type; };
template<> struct is_integral<char32_t>           { typedef true_type type; };
template<> struct is_integral<short>              { typedef true_type type; };
template<> struct is_integral<unsigned short>     { typedef true_type type; };
template<> struct is_integral<int>                { typedef true_type type; };
template<> struct is_integral<unsigned int>       { typedef true_type type; };
template<> struct is_integral<long>               { typedef true_type type; };
template<> struct is_integral<unsigned long>      { typedef true_type type; };
template<> struct is_integral<long long>          { typedef true_type type; };
template<> struct is_integral<unsigned long long> { typedef true_type type; };
*/

template<typename T> using is_const = std::is_const<T>;

template<typename T> using add_pointer = std::add_pointer<T>;

//template<typename T> struct add_pointer : std::add_pointer<T> { typedef std::add_pointer<T>::type; };
//template<typename T> struct add_pointer<const T> : std::add_pointer<const T> { typedef std::add_pointer<const T>::type; };

} // namespace ecuda

#else
///
/// Reimplementations of various C++11 tools contained in the
/// <type_traits> header so that we can use them even if the
/// compiler is pre-C++11.
///
namespace ecuda {

template<typename T> struct add_const     { typedef const T type; };
template<typename T> struct add_const<T&> { typedef const T& type; };

template<typename T> struct remove_const          { typedef T type; };
template<typename T> struct remove_const<const T> { typedef T type; };

template<typename T,typename U> struct is_same      { enum { value = 0 }; };
template<typename T>            struct is_same<T,T> { enum { value = 1 }; };

template<typename T> struct remove_reference      { typedef T type; };
template<typename T> struct remove_reference<T&>  { typedef T type; };
#ifdef ECUDA_CPP11_AVAILABLE
template<typename T> struct remove_reference<T&&> { typedef T type; };
#endif

template<typename T> struct add_lvalue_reference     { typedef T& type; };
template<typename T> struct add_lvalue_reference<T&> { typedef T& type; };

template<bool B,typename T,typename F> struct conditional     { typedef T type; };
template<typename T,typename F> struct conditional<false,T,F> { typedef F type; };

template<bool B,typename T=void> struct enable_if {};
template<typename T> struct enable_if<true,T> { typedef T type; };

template<typename T,T v>
struct integral_constant {
	static const/*constexpr*/ T value = v;
	typedef T value_type;
	typedef integral_constant<T,v> type;
	/*constexpr*/ operator T() { return v; }
};

typedef integral_constant<bool,true>  true_type;
typedef integral_constant<bool,false> false_type;

template<typename T> struct is_integral           { typedef false_type type; };
template<> struct is_integral<bool>               { typedef true_type type; };
template<> struct is_integral<char>               { typedef true_type type; };
template<> struct is_integral<signed char>        { typedef true_type type; };
template<> struct is_integral<unsigned char>      { typedef true_type type; };
#ifdef _GLIBCXX_USE_WCHAR_T
template<> struct is_integral<wchar_t>            { typedef true_type type; };
#endif
#ifdef ECUDA_CPP11_AVAILABLE
template<> struct is_integral<char16_t>           { typedef true_type type; };
template<> struct is_integral<char32_t>           { typedef true_type type; };
#endif
template<> struct is_integral<short>              { typedef true_type type; };
template<> struct is_integral<unsigned short>     { typedef true_type type; };
template<> struct is_integral<int>                { typedef true_type type; };
template<> struct is_integral<unsigned int>       { typedef true_type type; };
template<> struct is_integral<long>               { typedef true_type type; };
template<> struct is_integral<unsigned long>      { typedef true_type type; };
template<> struct is_integral<long long>          { typedef true_type type; };
template<> struct is_integral<unsigned long long> { typedef true_type type; };

template<typename T> struct is_const : false_type {};
template<typename T> struct is_const<const T> : true_type {};

template<typename T> struct add_pointer          { typedef typename remove_reference<T>::type* type; };
//template<typename T> struct add_pointer<const T> { typedef const T* type; }; // confirm this is needed

template<typename T> struct remove_pointer                    { typedef T type; };
template<typename T> struct remove_pointer<T*>                { typedef T type; };
template<typename T> struct remove_pointer<T* const>          { typedef T type; };
template<typename T> struct remove_pointer<T* volatile>       { typedef T type; };
template<typename T> struct remove_pointer<T* const volatile> { typedef T type; };

} // namespace std
#endif

namespace ecuda {

///
/// Forward declarations of the four pointer specializations used in the API.
///
//template<typename T>            class naked_ptr;    // forward declaration // deprecated
template<typename T,typename P> class padded_ptr;   // forward declaration
template<typename T>            class shared_ptr;   // forward declaration
template<typename T,typename P> class striding_ptr; // forward declaration
template<typename T,typename P> class unique_ptr;   // forward declaration
template<typename T,typename P> class striding_padded_ptr; // forward declaration

///
/// Casts any raw or managed pointer, specialized pointer, or combination thereof to a naked pointer.
///
/// This is used in the API when a raw pointer is required, but the specific representation is unknown
/// in the implementation. Specifically, a raw pointer from T* is simply itself, but a raw pointer from
/// unique_ptr<T> (and other specializations) is retrieved via a call from the get() method.
///
/// The cast is guaranteed regardless of how many layers of pointer management or pointer specialization
/// are present. For example, a variable ptr of type padded_ptr< int, shared_ptr<int> > will be unwound
/// so that a cast to a naked pointer T* is achieved by reinterpret_cast<T*>(ptr.get().get()).
///
template<typename T,typename U>            __HOST__ __DEVICE__ T naked_cast( U* ptr )                       { return reinterpret_cast<T>(ptr); }
template<typename T>                       __HOST__ __DEVICE__ T naked_cast( T* ptr ) { return ptr; }
//template<typename T,typename U>            __HOST__ __DEVICE__ T naked_cast( const naked_ptr<U>& ptr )      { return naked_cast<T>(ptr.get()); }
template<typename T,typename U,typename V> __HOST__ __DEVICE__ T naked_cast( const unique_ptr<U,V>& ptr )   { return naked_cast<T>(ptr.get()); }
template<typename T,typename U>            __HOST__ __DEVICE__ T naked_cast( const shared_ptr<U>& ptr )     { return naked_cast<T>(ptr.get()); }
template<typename T,typename U,typename V> __HOST__ __DEVICE__ T naked_cast( const padded_ptr<U,V>& ptr )   { return naked_cast<T>(ptr.get()); }
template<typename T,typename U,typename V> __HOST__ __DEVICE__ T naked_cast( const striding_ptr<U,V>& ptr ) { return naked_cast<T>(ptr.get()); }

///
/// Gets the type of a pointer that is guaranteed to be free of any pointer management by unique_ptr or shared_ptr.
///
/// This is used in the API when a pointer that can be modified is needed. For example, a pointer to video memory
/// that represents a matrix may or may not be managed by unique_ptr, but if the pointer is used to create a temporary
/// structure to represent a row, then we must guarantee that the pointer is freed of any such management.
///
/// This operates recursively. For example:
/// \code{.cpp}
/// typedef shared_ptr<int> pointer_type1;
/// typedef typename remove_pointer_management<pointer_type1>::type unmanaged_pointer_type1; // is of type int*
/// typedef padded_ptr< int,shared_ptr<int> > pointer_type2;
/// typedef typename remove_pointer_management<pointer_type2>::type unmanaged_pointer_type2; // is of type padded_ptr<int,int*>
/// \endcode
///
template<typename T>            struct make_unmanaged;
template<typename T>            struct make_unmanaged< T*                      > { typedef T* type; };
template<typename T>            struct make_unmanaged< const T*                > { typedef const T* type; };
//template<typename T>            struct make_unmanaged< naked_ptr<T>            > { typedef naked_ptr<T> type; };
//template<typename T>            struct make_unmanaged< const naked_ptr<T>      > { typedef naked_ptr<T> type; };
template<typename T,typename U> struct make_unmanaged< unique_ptr<T,U>         > { typedef typename unique_ptr<T,U>::pointer type; };
template<typename T,typename U> struct make_unmanaged< const unique_ptr<T,U>   > { typedef typename unique_ptr<T,U>::pointer type; };
template<typename T>            struct make_unmanaged< shared_ptr<T>           > { typedef typename ecuda::add_pointer<T>::type type; };
template<typename T>            struct make_unmanaged< const shared_ptr<T>     > { typedef typename ecuda::add_pointer<T>::type type; };
template<typename T,typename U> struct make_unmanaged< padded_ptr<T,U>         > { typedef padded_ptr<T,typename make_unmanaged<U>::type> type; };
template<typename T,typename U> struct make_unmanaged< const padded_ptr<T,U>   > { typedef padded_ptr<T,typename make_unmanaged<U>::type> type; };
template<typename T,typename U> struct make_unmanaged< striding_ptr<T,U>       > { typedef striding_ptr<T,typename make_unmanaged<U>::type> type; };
template<typename T,typename U> struct make_unmanaged< const striding_ptr<T,U> > { typedef striding_ptr<T,typename make_unmanaged<U>::type> type; };
template<typename T,typename U> struct make_unmanaged< striding_padded_ptr<T,U> > { typedef striding_padded_ptr<T,typename make_unmanaged<U>::type> type; };
template<typename T,typename U> struct make_unmanaged< const striding_padded_ptr<T,U> > { typedef striding_padded_ptr<T,typename make_unmanaged<U>::type> type; };


///
/// Casts any raw or managed pointer, specialized pointer, or combination thereof to a type that is free of any
/// management from unique_ptr or shared_ptr.
///
/// The type of the resulting pointer can be determined with ecuda::make_unmanaged<T>::type.
///
/// The cast should be guaranteed regardless of how many layers of pointer management or
/// pointer specialization are present.
///
/// \code{.cpp}
/// padded_ptr< int,shared_ptr<int> > p;
/// int* q = naked_cast<int*>( p );
/// double* r = naked_cast<double*>( p ); // not sure why this would be needed, but it can be done
/// \endcode
///
template<typename T> __HOST__ __DEVICE__ inline typename make_unmanaged<T*>::type unmanaged_cast( T* ptr ) { return ptr; }

//template<typename T> __HOST__ __DEVICE__ inline typename make_unmanaged< naked_ptr<T> >::type unmanaged_cast( const naked_ptr<T>& ptr ) { return naked_ptr<T>(ptr); }

template<typename T,typename U>
__HOST__ __DEVICE__ inline
typename make_unmanaged<U>::type unmanaged_cast( const unique_ptr<T,U>& ptr )
{
	return typename make_unmanaged<U>::type( ptr.get() );
}

template<typename T>
__HOST__ __DEVICE__ inline
typename make_unmanaged< shared_ptr<T> >::type
unmanaged_cast( const shared_ptr<T>& ptr )
{
	return typename make_unmanaged< shared_ptr<T> >::type( ptr.get() );
}

template<typename T,typename U>
__HOST__ __DEVICE__ inline
padded_ptr<T,typename make_unmanaged<U>::type>
unmanaged_cast( const padded_ptr<T,U>& ptr )
{
	//typename make_unmanaged<U>::type mp1 = unmanaged_cast( ptr.get_edge() );
	typename make_unmanaged<U>::type mp = unmanaged_cast( ptr.get() );
	return padded_ptr<T,typename make_unmanaged<U>::type>( mp, ptr.get_pitch() ); //, ptr.get_width(), mp2 );
}

template<typename T,typename U>
__HOST__ __DEVICE__ inline
striding_ptr<T,typename make_unmanaged<U>::type>
unmanaged_cast( const striding_ptr<T,U>& ptr )
{
	typename make_unmanaged<U>::type mp = unmanaged_cast( ptr.get() );
	return striding_ptr<T,typename make_unmanaged<U>::type>( mp, ptr.get_stride() );
}

template<typename T,typename U>
__HOST__ __DEVICE__ inline
striding_padded_ptr<T,typename make_unmanaged<U>::type>
unmanaged_cast( const striding_padded_ptr<T,U>& ptr )
{
	typename make_unmanaged<U>::type mp = unmanaged_cast( ptr.get() );
	return striding_padded_ptr<T,typename make_unmanaged<U>::type>( mp, ptr.get_stride() );
}

///
/// Gets the const-equivalent type of any raw or managed pointer, pointer specialization, or combination thereof.
///
/// This is used in the API when a const-equivalent pointer is needed. For example, a pointer that is used to
/// construct a constant iterator. Normally, a raw pointer T* has a const-equivalent of const T*. However, the
/// unique_ptr<T> specialization has a const-equivalent of unique_ptr<const T>, NOT const unique_ptr<T>. The
/// make_const<T>::type abstracts this distinction away so the API can deal with different pointer types in
/// the appropriate way at compile time.
///
/// The evaluation also operates recursively. For example, padded_ptr< T,shared_ptr<T> > will be properly
/// evaluated as having a const-equivalent of padded_ptr< const T, shared_ptr<const T> >.
///
template<typename T>            struct make_const;
template<typename T>            struct make_const< T*                      > { typedef const T* type; };
template<typename T>            struct make_const< const T*                > { typedef const T* type; };
//template<typename T>            struct make_const< naked_ptr<T>            > { typedef naked_ptr<const T> type; };
//template<typename T>            struct make_const< naked_ptr<const T>      > { typedef naked_ptr<const T> type; };
template<typename T,typename U> struct make_const< unique_ptr<T,U>         > { typedef unique_ptr<const T,typename make_const<U>::type> type; };
template<typename T,typename U> struct make_const< unique_ptr<const T,U>   > { typedef unique_ptr<const T,typename make_const<U>::type> type; };
template<typename T>            struct make_const< shared_ptr<T>           > { typedef shared_ptr<const T> type; };
template<typename T>            struct make_const< shared_ptr<const T>     > { typedef shared_ptr<const T> type; };
template<typename T,typename U> struct make_const< padded_ptr<T,U>         > { typedef padded_ptr<const T,typename make_const<U>::type> type; };
template<typename T,typename U> struct make_const< padded_ptr<const T,U>   > { typedef padded_ptr<const T,typename make_const<U>::type> type; };
template<typename T,typename U> struct make_const< striding_ptr<T,U>       > { typedef striding_ptr<const T,typename make_const<U>::type> type; };
template<typename T,typename U> struct make_const< striding_ptr<const T,U> > { typedef striding_ptr<const T,typename make_const<U>::type> type; };

template<typename T,typename U> struct make_const< striding_padded_ptr<T,U>       > { typedef striding_padded_ptr<const T,typename make_const<U>::type> type; };
template<typename T,typename U> struct make_const< striding_padded_ptr<const T,U> > { typedef striding_padded_ptr<const T,typename make_const<U>::type> type; };

///
/// Gets the unmanaged const-equivalent type of any raw of managed pointer, pointer specialization, or combination thereof.
///
/// This is effectively just an alias for a combination of the make_unmanaged<T> and make_const<T> type modifications.
///
template<typename T> struct make_unmanaged_const { typedef typename make_unmanaged<typename make_const<T>::type>::type type; };

} // namespace ecuda

/// \endcond

#endif
