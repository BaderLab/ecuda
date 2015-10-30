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

#include "global.hpp"

/// \cond DEVELOPER_DOCUMENTATION

#ifdef __CPP11_SUPPORTED__
#include <type_traits>
#else
namespace std {

template<typename T> struct remove_const          { typedef T type; };
template<typename T> struct remove_const<const T> { typedef T type; };

template<typename T,typename U> struct is_same      { enum { value = 0 }; };
template<typename T>            struct is_same<T,T> { enum { value = 1 }; };

template<typename T> struct remove_reference      { typedef T type; };
template<typename T> struct remove_reference<T&>  { typedef T type; };
#ifdef __CPP11_SUPPORTED__
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

template<typename T> struct is_integral           { typedef std::false_type type; };
template<> struct is_integral<bool>               { typedef std::true_type type; };
template<> struct is_integral<char>               { typedef std::true_type type; };
template<> struct is_integral<signed char>        { typedef std::true_type type; };
template<> struct is_integral<unsigned char>      { typedef std::true_type type; };
#ifdef _GLIBCXX_USE_WCHAR_T
template<> struct is_integral<wchar_t>            { typedef std::true_type type; };
#endif
#ifdef __CPP11_SUPPORTED__
template<> struct is_integral<char16_t>           { typedef std::true_type type; };
template<> struct is_integral<char32_t>           { typedef std::true_type type; };
#endif
template<> struct is_integral<short>              { typedef std::true_type type; };
template<> struct is_integral<unsigned short>     { typedef std::true_type type; };
template<> struct is_integral<int>                { typedef std::true_type type; };
template<> struct is_integral<unsigned int>       { typedef std::true_type type; };
template<> struct is_integral<long>               { typedef std::true_type type; };
template<> struct is_integral<unsigned long>      { typedef std::true_type type; };
template<> struct is_integral<long long>          { typedef std::true_type type; };
template<> struct is_integral<unsigned long long> { typedef std::true_type type; };

template<typename T> struct add_pointer          { typedef T* type; };
template<typename T> struct add_pointer<const T> { typedef const T* type; }; // confirm this is needed

template<typename T> struct remove_pointer                    { typedef T type; };
template<typename T> struct remove_pointer<T*>                { typedef T type; };
template<typename T> struct remove_pointer<T* const>          { typedef T type; };
template<typename T> struct remove_pointer<T* volatile>       { typedef T type; };
template<typename T> struct remove_pointer<T* const volatile> { typedef T type; };

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


//template<typename T> struct dereference;
//template<typename T> struct dereference<T*> { typedef T& type; };
//template<typename T> struct dereference<T* const> { typedef const T& type; };
//template<typename T> struct reference {
//	typedef T* pointer_type;
//	typedef T& reference_type;
//	typedef T element_type;
//};

//template<typename T> struct type_traits { typedef T* pointer; };
//template<typename T> struct type_traits<const T> { typedef const T* pointer; };

//template<typename T>
//struct pointer_traits {
//	typedef T* pointer;
//};
//
//template<typename T>
//struct pointer_traits<const T> {
//	typedef const T* pointer;
//};

/*
template<typename T>            T*                                  make_naked( T* ptr )                       { return ptr; }
template<typename T>            const T*                            make_naked( const T* ptr )                 { return ptr; }
template<typename T>            typename naked_ptr<T>::pointer      make_naked( const naked_ptr<T>& ptr )      { return make_naked(ptr.get()); }
template<typename T,typename U> typename unique_ptr<T,U>::pointer   make_naked( const unique_ptr<T,U>& ptr )   { return make_naked(ptr.get()); }
template<typename T>            typename shared_ptr<T>::pointer     make_naked( const shared_ptr<T>& ptr )     { return make_naked(ptr.get()); }
template<typename T,typename U> typename padded_ptr<T,U>::pointer   make_naked( const padded_ptr<T,U>& ptr )   { return make_naked(ptr.get()); }
template<typename T,typename U> typename striding_ptr<T,U>::pointer make_naked( const striding_ptr<T,U>& ptr ) { return make_naked(ptr.get()); }
*/

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
template<typename T,typename U>            __HOST__ __DEVICE__ T naked_cast( const naked_ptr<U>& ptr )      { return naked_cast<T>(ptr.get()); }
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
template<typename T>            struct make_unmanaged<T*>                        { typedef T* type; };
template<typename T>            struct make_unmanaged<const T*>                  { typedef const T* type; };
template<typename T>            struct make_unmanaged< naked_ptr<T> >            { typedef naked_ptr<T> type; };
template<typename T>            struct make_unmanaged< const naked_ptr<T> >      { typedef naked_ptr<T> type; };
template<typename T,typename U> struct make_unmanaged< unique_ptr<T,U> >         { typedef typename unique_ptr<T,U>::pointer type; };
template<typename T,typename U> struct make_unmanaged< const unique_ptr<T,U> >   { typedef typename unique_ptr<T,U>::pointer type; };
template<typename T>            struct make_unmanaged< shared_ptr<T> >           { typedef typename std::add_pointer<T>::type type; };
template<typename T>            struct make_unmanaged< const shared_ptr<T> >     { typedef typename std::add_pointer<T>::type type; };
template<typename T,typename U> struct make_unmanaged< padded_ptr<T,U> >         { typedef padded_ptr<T,typename make_unmanaged<U>::type> type; };
template<typename T,typename U> struct make_unmanaged< const padded_ptr<T,U> >   { typedef padded_ptr<T,typename make_unmanaged<U>::type> type; };
template<typename T,typename U> struct make_unmanaged< striding_ptr<T,U> >       { typedef striding_ptr<T,typename make_unmanaged<U>::type> type; };
template<typename T,typename U> struct make_unmanaged< const striding_ptr<T,U> > { typedef striding_ptr<T,typename make_unmanaged<U>::type> type; };

///
/// Casts any raw or managed pointer, specialized pointer, or combination thereof to a type that is free of any management from unique_ptr or shared_ptr.
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

template<typename T> __HOST__ __DEVICE__ inline typename make_unmanaged< naked_ptr<T> >::type unmanaged_cast( const naked_ptr<T>& ptr ) { return naked_ptr<T>(ptr); }

template<typename T,typename U>
__HOST__ __DEVICE__
typename make_unmanaged<U>::type unmanaged_cast( const unique_ptr<T,U>& ptr ) {
	return typename make_unmanaged<U>::type( ptr.get() );
}

template<typename T>
__HOST__ __DEVICE__
typename make_unmanaged< shared_ptr<T> >::type
unmanaged_cast( const shared_ptr<T>& ptr ) {
	return typename make_unmanaged< shared_ptr<T> >::type( ptr.get() );
}

template<typename T,typename U>
__HOST__ __DEVICE__
padded_ptr<T,typename make_unmanaged<U>::type>
unmanaged_cast( const padded_ptr<T,U>& ptr ) {
	typename make_unmanaged<U>::type mp1 = unmanaged_cast( ptr.get_edge() );
	typename make_unmanaged<U>::type mp2 = unmanaged_cast( ptr.get() );
	return padded_ptr<T,typename make_unmanaged<U>::type>( mp1, ptr.get_pitch(), ptr.get_width(), mp2 );
}

template<typename T,typename U>
__HOST__ __DEVICE__
striding_ptr<T,typename make_unmanaged<U>::type>
unmanaged_cast( const striding_ptr<T,U>& ptr ) {
	typename make_unmanaged<U>::type mp = unmanaged_cast( ptr.get() );
	return striding_ptr<T,typename make_unmanaged<U>::type>( mp, ptr.get_stride() );
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
template<typename T>            struct make_const<T*>                        { typedef const T* type; };
template<typename T>            struct make_const<const T*>                  { typedef const T* type; };
template<typename T>            struct make_const< naked_ptr<T> >            { typedef naked_ptr<const T> type; };
template<typename T>            struct make_const< naked_ptr<const T> >      { typedef naked_ptr<const T> type; };
template<typename T,typename U> struct make_const< unique_ptr<T,U> >         { typedef unique_ptr<const T,typename make_const<U>::type> type; };
template<typename T,typename U> struct make_const< unique_ptr<const T,U> >   { typedef unique_ptr<const T,typename make_const<U>::type> type; };
template<typename T>            struct make_const< shared_ptr<T> >           { typedef shared_ptr<const T> type; };
template<typename T>            struct make_const< shared_ptr<const T> >     { typedef shared_ptr<const T> type; };
template<typename T,typename U> struct make_const< padded_ptr<T,U> >         { typedef padded_ptr<const T,typename make_const<U>::type> type; };
template<typename T,typename U> struct make_const< padded_ptr<const T,U> >   { typedef padded_ptr<const T,typename make_const<U>::type> type; };
template<typename T,typename U> struct make_const< striding_ptr<T,U> >       { typedef striding_ptr<const T,typename make_const<U>::type> type; };
template<typename T,typename U> struct make_const< striding_ptr<const T,U> > { typedef striding_ptr<const T,typename make_const<U>::type> type; };

///
/// Gets the unmanaged const-equivalent type of any raw of managed pointer, pointer specialization, or combination thereof.
///
/// This is effectively just an alias for a combination of the make_unmanaged<T> and make_const<T> type modifications.
///
template<typename T> struct make_unmanaged_const { typedef typename make_unmanaged<typename make_const<T>::type>::type type; };

/*
template<typename T> struct pointer_traits {
	template<typename U>
	static U cast_unmanaged( T ptr ) { return pointer_traits<T>().make_unmanaged(ptr); }
};

//template<typename T> typename pointer_traits<T>::unmanaged_pointer cast_unmanaged( T ptr );
//template<typename T> typename pointer_traits<T*>::unmanaged_pointer cast_unmanaged( T* ptr ) { return ptr; }

//template<typename T>


template<typename T>
struct pointer_traits<T*> {
	typedef T element_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T* naked_pointer;
	typedef T* unmanaged_pointer;
	typedef char* char_pointer;
	__HOST__ __DEVICE__ inline naked_pointer undress( pointer ptr ) const { return ptr; }
	__HOST__ __DEVICE__ inline unmanaged_pointer increment( pointer ptr, const int x ) const { return ptr + x; }
	__HOST__ __DEVICE__ inline unmanaged_pointer make_unmanaged( pointer ptr ) const { return ptr; }
	template<typename U> static __HOST__ __DEVICE__ typename pointer_traits<U*>::unmanaged_pointer cast_unmanaged( U* ptr ) { return ptr; }
	template<typename U> static __HOST__ __DEVICE__ typename pointer_traits<U*>::naked_pointer cast_naked( U* ptr ) { return ptr; }
};

template<typename T>
struct pointer_traits<const T*> {
	typedef const T element_type;
	typedef const T* pointer;
	typedef const T* const_pointer;
	typedef const T* naked_pointer;
	typedef const T* unmanaged_pointer;
	typedef const char* char_pointer;
	__HOST__ __DEVICE__ inline naked_pointer undress( pointer ptr ) const { return ptr; }
	__HOST__ __DEVICE__ inline unmanaged_pointer increment( pointer ptr, const int x ) const { return ptr + x; }
	__HOST__ __DEVICE__ inline unmanaged_pointer make_unmanaged( pointer ptr ) const { return ptr; }
	template<typename U> static __HOST__ __DEVICE__ typename pointer_traits<U*>::unmanaged_pointer cast_unmanaged( U* ptr ) { return ptr; }
	template<typename U> static __HOST__ __DEVICE__ typename pointer_traits<U*>::naked_pointer cast_naked( U* ptr ) { return ptr; }
};

template<typename T>
struct pointer_traits< naked_ptr<T> > {
	typedef typename naked_ptr<T>::element_type element_type;
	typedef naked_ptr<T> pointer;
	typedef naked_ptr<const T> const_pointer;
	typedef typename naked_ptr<T>::pointer naked_pointer;
	typedef naked_ptr<T> unmanaged_pointer;
	typedef char* char_pointer;
	__HOST__ __DEVICE__ inline naked_pointer undress( const pointer& ptr ) const { return ptr.get(); }
	__HOST__ __DEVICE__ inline unmanaged_pointer increment( const pointer& ptr, const int x ) const { return ptr + x; }
	__HOST__ __DEVICE__ inline unmanaged_pointer& make_unmanaged( pointer& ptr ) const { return ptr; }
	template<typename U> static __HOST__ __DEVICE__ typename pointer_traits< naked_ptr<U> >::unmanaged_pointer cast_unmanaged( const naked_ptr<U>& ptr ) { return ptr.get(); }
	template<typename U> static __HOST__ __DEVICE__ typename pointer_traits< naked_ptr<U> >::naked_pointer cast_naked( const naked_ptr<U>& ptr ) { return ptr.get(); }
};

template<typename T>
struct pointer_traits< const naked_ptr<T> > {
	typedef typename naked_ptr<T>::element_type element_type;
	typedef const naked_ptr<T> pointer;
	typedef const naked_ptr<const T> const_pointer;
	typedef typename naked_ptr<T>::pointer naked_pointer;
	typedef naked_pointer unmanaged_pointer;
	typedef typename pointer_traits<naked_pointer>::char_pointer char_pointer;
	__HOST__ __DEVICE__ inline naked_pointer undress( const pointer& ptr ) const { return ptr.get(); }
	__HOST__ __DEVICE__ inline unmanaged_pointer increment( const pointer& ptr, const int x ) const { return ptr.get() + x; }
	__HOST__ __DEVICE__ inline unmanaged_pointer make_unmanaged( pointer& ptr ) const { return ptr.get(); }
	template<typename U> static __HOST__ __DEVICE__ typename pointer_traits< const naked_ptr<U> >::unmanaged_pointer cast_unmanaged( const naked_ptr<U>& ptr ) { return ptr.get(); }
	template<typename U> static __HOST__ __DEVICE__ typename pointer_traits< const naked_ptr<U> >::naked_pointer cast_naked( const naked_ptr<U>& ptr ) { return ptr.get(); }
};

template<typename T,typename U>
struct pointer_traits< unique_ptr<T,U> > {
	typedef typename unique_ptr<T,U>::element_type element_type;
	typedef unique_ptr<T,U> pointer;
	typedef unique_ptr<const T,U> const_pointer;
	typedef typename unique_ptr<T,U>::pointer naked_pointer;
	typedef naked_pointer unmanaged_pointer;
	typedef typename pointer_traits<naked_pointer>::char_pointer char_pointer;
	__HOST__ __DEVICE__ inline naked_pointer undress( const pointer& ptr ) const { return ptr.get(); }
	__HOST__ __DEVICE__ inline unmanaged_pointer increment( pointer& ptr, const int x ) const { return ptr.get() + x; }
	__HOST__ __DEVICE__ inline unmanaged_pointer make_unmanaged( pointer& ptr ) const { return undress(ptr); }
	template<typename T2,typename U2> static __HOST__ __DEVICE__ typename pointer_traits< unique_ptr<T2,U2> >::unmanaged_pointer cast_unmanaged( const unique_ptr<T2,U2>& ptr ) { return ptr.get(); }
	template<typename T2,typename U2> static __HOST__ __DEVICE__ typename pointer_traits< unique_ptr<T2,U2> >::naked_pointer cast_naked( const unique_ptr<T2,U2>& ptr ) { return ptr.get(); }
};

template<typename T,typename U>
struct pointer_traits< const unique_ptr<T,U> > {
	typedef typename unique_ptr<T,U>::element_type element_type;
	typedef const unique_ptr<T,U> pointer;
	typedef const unique_ptr<const T,U> const_pointer;
	typedef typename unique_ptr<T,U>::pointer naked_pointer;
	typedef naked_pointer unmanaged_pointer;
	typedef typename pointer_traits<naked_pointer>::char_pointer char_pointer;
	__HOST__ __DEVICE__ inline naked_pointer undress( const pointer& ptr ) const { return ptr.get(); }
	__HOST__ __DEVICE__ inline unmanaged_pointer increment( pointer& ptr, const int x ) const { return ptr.get() + x; }
	__HOST__ __DEVICE__ inline unmanaged_pointer make_unmanaged( pointer& ptr ) const { return undress(ptr); }
	template<typename T2,typename U2> static __HOST__ __DEVICE__ typename pointer_traits< const unique_ptr<T2,U2> >::unmanaged_pointer cast_unmanaged( const unique_ptr<T2,U2>& ptr ) { return ptr.get(); }
	template<typename T2,typename U2> static __HOST__ __DEVICE__ typename pointer_traits< const unique_ptr<T2, U2> >::naked_pointer cast_naked( const unique_ptr<T2, U2>& ptr ) { return ptr.get(); }
};

template<typename T>
struct pointer_traits< shared_ptr<T> > {
	typedef typename shared_ptr<T>::element_type element_type;
	typedef shared_ptr<T> pointer;
	typedef shared_ptr<const T> const_pointer;
	typedef T* naked_pointer;
	typedef naked_pointer unmanaged_pointer;
	typedef typename pointer_traits<naked_pointer>::char_pointer char_pointer;
	__HOST__ __DEVICE__ inline naked_pointer undress( const pointer& ptr ) const { return ptr.get(); }
	__HOST__ __DEVICE__ inline unmanaged_pointer increment( const shared_ptr<T> ptr, const int x ) const { return ptr.get() + x; }
	__HOST__ __DEVICE__ inline unmanaged_pointer make_unmanaged( pointer ptr ) const { return undress(ptr); }
	template<typename U> static __HOST__ __DEVICE__ typename pointer_traits< shared_ptr<U> >::unmanaged_pointer cast_unmanaged( const shared_ptr<U>& ptr ) { return ptr.get(); }
	template<typename U> static __HOST__ __DEVICE__ typename pointer_traits< shared_ptr<U> >::naked_pointer cast_naked( const shared_ptr<U>& ptr ) { return ptr.get(); }
};

template<typename T>
struct pointer_traits< const shared_ptr<T> > {
	typedef typename shared_ptr<T>::element_type element_type;
	typedef const shared_ptr<T> pointer;
	typedef const shared_ptr<const T> const_pointer;
	typedef typename shared_ptr<T>::pointer naked_pointer;
	typedef typename pointer_traits<naked_pointer>::char_pointer char_pointer;
	__HOST__ __DEVICE__ inline naked_pointer undress( const pointer& ptr ) const { return ptr.get(); }
	template<typename U> static __HOST__ __DEVICE__ typename pointer_traits< const shared_ptr<U> >::unmanaged_pointer cast_unmanaged( const shared_ptr<U>& ptr ) { return ptr.get(); }
	template<typename U> static __HOST__ __DEVICE__ typename pointer_traits< const shared_ptr<U> >::naked_pointer cast_naked( const shared_ptr<U>& ptr ) { return ptr.get(); }
};

template<typename T,typename U>
struct pointer_traits< padded_ptr<T,U> > {
	typedef typename padded_ptr<T,U>::element_type element_type;
	typedef padded_ptr<T,U> pointer;
	typedef padded_ptr<const T,typename pointer_traits<U>::const_pointer> const_pointer;
	typedef typename pointer_traits<U>::naked_pointer naked_pointer;
	typedef padded_ptr<T,typename pointer_traits<U>::unmanaged_pointer> unmanaged_pointer;
	typedef typename pointer_traits<U>::char_pointer char_pointer;
	__HOST__ __DEVICE__ inline naked_pointer undress( const pointer& ptr ) const { return pointer_traits<U>().undress( ptr.get() ); }
	__HOST__ __DEVICE__ inline unmanaged_pointer increment( const pointer& ptr, const int x ) const { return make_unmanaged(ptr)+x; }
	__HOST__ __DEVICE__ inline unmanaged_pointer make_unmanaged( const pointer& ptr ) const {
		typename pointer_traits<U>::unmanaged_pointer mp1 = pointer_traits<U>().make_unmanaged( ptr.get_edge() );
		typename pointer_traits<U>::unmanaged_pointer mp2 = pointer_traits<U>().make_unmanaged( ptr.get() );
		return unmanaged_pointer( mp1, ptr.get_pitch(), ptr.get_width(), mp2 );
	}
	template<typename T2,typename U2>
	static __HOST__ __DEVICE__ typename pointer_traits< padded_ptr<T2,U2> >::unmanaged_pointer cast_unmanaged( const padded_ptr<T2,U2>& ptr ) {
		typename pointer_traits<U2>::unmanaged_pointer p1 = pointer_traits<U2>::cast_unmanaged( ptr.get_edge() );
		typename pointer_traits<U2>::unmanaged_pointer p2 = pointer_traits<U2>::cast_unmanaged( ptr.get() );
		return typename pointer_traits< padded_ptr<T2,U2> >::unmanaged_pointer( p1, ptr.get_pitch(), ptr.get_width(), p2 );
	}
	template<typename T2, typename U2> static __HOST__ __DEVICE__ typename pointer_traits< padded_ptr<T2, U2> >::naked_pointer cast_naked( const padded_ptr<T2, U2>& ptr ) { return pointer_traits<U2>::cast_naked( ptr.get() ); }
};

template<typename T,typename U>
struct pointer_traits< const padded_ptr<T,U> > {
	typedef typename padded_ptr<T,U>::element_type element_type;
	typedef const padded_ptr<T,U> pointer;
	typedef const padded_ptr<const T,typename pointer_traits<U>::const_pointer> const_pointer;
	typedef typename pointer_traits<U>::naked_pointer naked_pointer;
	typedef padded_ptr<const T,typename pointer_traits<typename pointer_traits<U>::const_pointer>::unmanaged_pointer> unmanaged_pointer;
	typedef typename pointer_traits<U>::char_pointer char_pointer;
	__HOST__ __DEVICE__ inline naked_pointer undress( const pointer& ptr ) const { return pointer_traits<U>().undress( ptr.get() ); }
	__HOST__ __DEVICE__ inline unmanaged_pointer increment( const pointer& ptr, const int x ) const { return make_unmanaged(ptr)+x; }
	__HOST__ __DEVICE__ inline unmanaged_pointer make_unmanaged( const pointer& ptr ) const {
		typename pointer_traits<U>::unmanaged_pointer mp1 = pointer_traits<U>().make_unmanaged( ptr.get_edge() );
		typename pointer_traits<U>::unmanaged_pointer mp2 = pointer_traits<U>().make_unmanaged( ptr.get() );
		return unmanaged_pointer( mp1, ptr.get_pitch(), ptr.get_width(), mp2 );
	}
	template<typename T2,typename U2>
	static __HOST__ __DEVICE__ typename pointer_traits< const padded_ptr<T2,U2> >::unmanaged_pointer cast_unmanaged( const padded_ptr<T2,U2>& ptr ) {
		typename pointer_traits<U2>::unmanaged_pointer p1 = pointer_traits<U2>::cast_unmanaged( ptr.get_edge() );
		typename pointer_traits<U2>::unmanaged_pointer p2 = pointer_traits<U2>::cast_unmanaged( ptr.get() );
		return typename pointer_traits<U2>::unmanaged_pointer( p1, ptr.get_pitch(), ptr.get_width(), p2 );
	}
	template<typename T2, typename U2> static __HOST__ __DEVICE__ typename pointer_traits< const padded_ptr<T2, U2> >::naked_pointer cast_naked( const padded_ptr<T2, U2>& ptr ) { return pointer_traits<U2>::cast_naked( ptr.get() ); }
};

template<typename T,typename U>
struct pointer_traits< striding_ptr<T,U> > {
	typedef typename striding_ptr<T,U>::element_type element_type;
	typedef striding_ptr<T,U> pointer;
	typedef striding_ptr<const T,typename pointer_traits<U>::const_pointer> const_pointer;
	typedef typename pointer_traits<U>::naked_pointer naked_pointer;
	typedef striding_ptr<T,typename pointer_traits<U>::unmanaged_pointer> unmanaged_pointer;
	typedef typename pointer_traits<U>::char_pointer char_pointer;
	__HOST__ __DEVICE__ inline naked_pointer undress( const pointer& ptr ) const { return pointer_traits<U>().undress( ptr.get() ); }
	__HOST__ __DEVICE__ inline unmanaged_pointer make_unmanaged( const pointer& ptr ) const { return unmanaged_pointer( pointer_traits<U>().make_unmanaged( ptr.get() ), ptr.get_stride() ); }
	__HOST__ __DEVICE__ inline unmanaged_pointer increment( const pointer& ptr, const int x ) const { return make_unmanaged(ptr)+x; }
	template<typename T2,typename U2>
	static __HOST__ __DEVICE__ typename pointer_traits< striding_ptr<T2,U2> >::unmanaged_pointer cast_unmanaged( const striding_ptr<T2,U2>& ptr ) {
		return typename pointer_traits< striding_ptr<T2,U2> >::unmanaged_pointer( pointer_traits<U2>::cast_unmanaged( ptr.get() ), ptr.get_stride() );
	}
	template<typename T2, typename U2> static __HOST__ __DEVICE__ typename pointer_traits< striding_ptr<T2, U2> >::naked_pointer cast_naked( const striding_ptr<T2, U2>& ptr ) { return pointer_traits<U2>::cast_naked( ptr.get() ); }
};

template<typename T,typename U>
struct pointer_traits< const striding_ptr<T,U> > {
	typedef typename striding_ptr<T,U>::element_type element_type;
	typedef const striding_ptr<T,U> pointer;
	typedef const striding_ptr<const T,typename pointer_traits<U>::const_pointer> const_pointer;
	typedef typename pointer_traits<U>::naked_pointer naked_pointer;
	typedef striding_ptr<const T,typename pointer_traits<typename pointer_traits<U>::const_pointer>::unmanaged_pointer> unmanaged_pointer;
	typedef typename pointer_traits<U>::char_pointer char_pointer;
	__HOST__ __DEVICE__ inline naked_pointer undress( const pointer& ptr ) const { return pointer_traits<U>().undress( ptr.get() ); }
	__HOST__ __DEVICE__ inline unmanaged_pointer make_unmanaged( const pointer& ptr ) const { return unmanaged_pointer( pointer_traits<U>().make_unmanaged( ptr.get() ), ptr.get_stride() ); }
	__HOST__ __DEVICE__ inline unmanaged_pointer increment( const pointer& ptr, const int x ) const { return make_unmanaged(ptr)+x; }
	template<typename T2,typename U2>
	static __HOST__ __DEVICE__ typename pointer_traits< const striding_ptr<T2,U2> >::unmanaged_pointer cast_unmanaged( const striding_ptr<T2,U2>& ptr ) {
		return typename pointer_traits<U2>::unmanaged_pointer( typename pointer_traits<U2>::cast_unmanaged( ptr.get() ), ptr.get_stride() );
	}
	template<typename T2, typename U2> static __HOST__ __DEVICE__ typename pointer_traits< const striding_ptr<T2, U2> >::naked_pointer cast_naked( const striding_ptr<T2, U2>& ptr ) { return pointer_traits<U2>::cast_naked( ptr.get() ); }
};

*/

} // namespace ecuda

/// \endcond

#endif
