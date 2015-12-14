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
// ptr/striding_padded_ptr.hpp
//
// A pointer specialization that describes device memory where elements are
// stored with a constant spacing in-between (hence incrementing/decrementing
// the pointer involves "striding" between elements).
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#ifndef ECUDA_PTR_STRIDING_PADDED_PTR_HPP
#define ECUDA_PTR_STRIDING_PADDED_PTR_HPP

#include <iostream>

#include "../global.hpp"
#include "../type_traits.hpp"
#include "../utility.hpp"
#include "common.hpp"

namespace ecuda {

///
/// \brief A specialized pointer to striding memory.
///
/// A specialized pointer to device memory where traversal of the data takes into a "stride", or a
/// fixed number of elements that are skipped each time the pointer is incremented.
///
/// The specialization is used to create certain views of a matrix or cube (e.g. single matrix column).
///
/// Memory use can be conceptualized as:
/// \code
///   |--- stride ----| // in multiples of sizeof(T)
///   +-+-------------+
///   | |xxxxxxxxxxxxx|
///   | |xxxxxxxxxxxxx| x = allocated but not used
///   | |xxxxxxxxxxxxx|
///   | |xxxxxxxxxxxxx|
///   | |xxxxxxxxxxxxx|
///   | |xxxxxxxxxxxxx| ... etc. (total size of the allocation is not known internally by striding_ptr)
///   +-+--------+----+
/// \endcode
///
/// For example, a pointer that will traverse the first column of a 10x5 matrix containing elements
/// of type T could be represented with striding_ptr<T>(ptr,5), where ptr points to the first element
/// of the matrix.
///
template<typename T,typename P=typename ecuda::add_pointer<T>::type>
class striding_padded_ptr
{

public:
	typedef T              element_type;
	typedef P              pointer;
	typedef T&             reference;
	typedef const T&       const_reference;
	typedef std::size_t    size_type;
	typedef std::ptrdiff_t difference_type;

private:
	pointer ptr;      //!< pointer to current element
	size_type stride; //!< amount (in bytes!) pointer should move to reach next element

private:
	template<typename U> struct char_ptr;
	template<typename U> struct char_ptr<U*> { typedef char* type; };
	template<typename U> struct char_ptr<const U*> { typedef const char* type; };

	template<typename T2,typename P2> friend class striding_padded_ptr;

public:
	__HOST__ __DEVICE__ striding_padded_ptr( pointer ptr = pointer(), size_type stride = 0 ) : ptr(ptr), stride(stride) {}

	__HOST__ __DEVICE__ striding_padded_ptr( const striding_padded_ptr& src ) : ptr(src.ptr), stride(src.stride) {}

	template<typename T2,typename P2>
	__HOST__ __DEVICE__ striding_padded_ptr( const striding_padded_ptr<T2,P2>& src ) : ptr(src.ptr), stride(src.stride) {}

	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ striding_padded_ptr( striding_padded_ptr&& src ) : ptr(std::move(src.ptr)), stride(std::move(src.stride)) {}

	__HOST__ __DEVICE__ striding_padded_ptr& operator=( striding_padded_ptr&& src )
	{
		ptr = std::move(src.ptr);
		stride = std::move(src.stride);
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline pointer get() const { return ptr; }

	__HOST__ __DEVICE__ inline size_type get_stride() const { return stride; }

	__DEVICE__ inline reference       operator*()                       { return *ptr; }
	__DEVICE__ inline const_reference operator*() const                 { return *ptr; }
	__DEVICE__ inline pointer         operator->() const                { return ptr; }
	__DEVICE__ inline reference       operator[]( std::size_t i )       { return striding_padded_ptr(*this).operator+=(i).operator*(); }
	__DEVICE__ inline const_reference operator[]( std::size_t i ) const { return striding_padded_ptr(*this).operator+=(i).operator*(); }

	__HOST__ __DEVICE__ inline striding_padded_ptr& operator++()
	{
		typedef typename char_ptr<pointer>::type char_pointer_type;
		ptr = reinterpret_cast<pointer>( reinterpret_cast<char_pointer_type>(ptr) + stride );
		return *this;
	}
	__HOST__ __DEVICE__ inline striding_padded_ptr& operator--()
	{
		typedef typename char_ptr<pointer>::type char_pointer_type;
		ptr = reinterpret_cast<pointer>( reinterpret_cast<char_pointer_type>(ptr) - stride );
		return *this;
	}
	__HOST__ __DEVICE__ inline striding_padded_ptr  operator++( int ) { striding_padded_ptr tmp(*this); ++(*this); return tmp; }
	__HOST__ __DEVICE__ inline striding_padded_ptr  operator--( int ) { striding_padded_ptr tmp(*this); --(*this); return tmp; }

	__HOST__ __DEVICE__ inline striding_padded_ptr& operator+=( int x )
	{
		typedef typename char_ptr<pointer>::type char_pointer_type;
		ptr = reinterpret_cast<pointer>( reinterpret_cast<char_pointer_type>(ptr) + x*stride );
		return *this;
	}
	__HOST__ __DEVICE__ inline striding_padded_ptr& operator-=( int x )
	{
		typedef typename char_ptr<pointer>::type char_pointer_type;
		ptr = reinterpret_cast<pointer>( reinterpret_cast<char_pointer_type>(ptr) - x*stride );
		return *this;
	}

	__HOST__ __DEVICE__ inline striding_padded_ptr operator+( int x ) const { striding_padded_ptr tmp(*this); tmp += x; return tmp; }
	__HOST__ __DEVICE__ inline striding_padded_ptr operator-( int x ) const { striding_padded_ptr tmp(*this); tmp -= x; return tmp; }

	template<typename T2,typename P2> __HOST__ __DEVICE__ inline bool operator==( const striding_padded_ptr<T2,P2>& other ) const { return ptr == other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ inline bool operator!=( const striding_padded_ptr<T2,P2>& other ) const { return ptr != other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ inline bool operator< ( const striding_padded_ptr<T2,P2>& other ) const { return ptr <  other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ inline bool operator> ( const striding_padded_ptr<T2,P2>& other ) const { return ptr >  other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ inline bool operator<=( const striding_padded_ptr<T2,P2>& other ) const { return ptr <= other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ inline bool operator>=( const striding_padded_ptr<T2,P2>& other ) const { return ptr >= other.ptr; }

	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const striding_padded_ptr& ptr )
	{
		out << "striding_padded_ptr(ptr=" << ptr.ptr << ";stride=" << ptr.stride << ")";
		return out;
	}

};

} // namespace ecuda

#endif
