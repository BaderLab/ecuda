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
// striding_ptr.hpp
//
// Pointer specialization that causes increment/decrement operations to
// skip (stride) a fixed number of units of the size of the type pointed to.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_STRIDING_PTR_HPP
#define ECUDA_STRIDING_PTR_HPP

#include "global.hpp"
#include "padded_ptr.hpp"

namespace ecuda {

///
/// \brief A specialized pointer to memory holding a sequence of data with fixed-sized spacing between each value.
///
/// Strided memory is a block of contiguous memory where elements are separated by
/// fixed-length padding.  Thus, one has to "stride" over the padding to reach the
/// next element.  The term was borrowed from the GNU Scientific Library.
///
/// For example, a pointer to memory representing a matrix of with N rows of M columns
/// can be given to a striding_ptr with the stride set to length M. When the striding_ptr
/// is incremented, the pointer moves column-wise.  This is useful for creating
/// flexible, iterable views of a fixed region of memory while minimizing overhead.
///
template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type>
class striding_ptr {

public:
	typedef T element_type; //!< data type represented in allocated memory
	typedef PointerType pointer; //!< data type pointer
	typedef T& reference; //!< data type reference
	typedef std::size_t size_type; //!< size type for pointer arithmetic and reference counting
	typedef std::ptrdiff_t difference_type; //!< signed integer type of the result of subtracting two pointers

private:
	pointer ptr;
	size_type stride; // stride in bytes

public:

	// NOTE: stride*sizeof(T) must be exact multiple of padded_ptr.get_width()
	template<typename PointerType2,std::size_t PaddingUnitBytes>
	HOST DEVICE striding_ptr( const padded_ptr<T,PointerType2,PaddingUnitBytes>& p, const size_type stride = 1 ) :
		ptr(p.get()),
		stride( stride*sizeof(T)+p.get_pitch()*stride/p.get_width() )
	{
	}

//	HOST DEVICE striding_ptr( pointer p = pointer(), const size_type stride = 1 ) : ptr(p), stride(stride*sizeof(T)) {}
	HOST DEVICE striding_ptr( const striding_ptr<T,PointerType>& src ) : ptr(src.ptr), stride(src.stride) {}
	//template<typename U,std::size_t StrideBytes2>
	//strided_ptr( const strided_ptr<U,StrideBytes2>& src ) : ptr(src.ptr), stride(src.stride) {}
	HOST DEVICE ~striding_ptr() {}

	HOST DEVICE inline size_type get_stride() const { return stride; }

	HOST DEVICE inline pointer get() const { return ptr; }
	HOST DEVICE inline operator bool() const { return ptr != nullptr; }

	///
	/// \brief operator T*
	///
	/// Allows this object to be type-cast to a naked pointer of type T*.
	/// \code{.cpp}
	/// static_cast<T*>(this)
	/// (T*)(*this)
	/// \endcode
	///
	HOST DEVICE inline operator typename ecuda::reference<element_type>::pointer_type() const { return ptr; }

	HOST DEVICE inline striding_ptr& operator++() {
		ptr = reinterpret_cast<pointer>( reinterpret_cast<typename cast_to_char<T*>::type>(ptr)+stride );
		//ptr += stride;
		return *this;
	}
	HOST DEVICE inline striding_ptr operator++( int ) {
		striding_ptr tmp(*this);
		++(*this);
		return tmp;
	}

	HOST DEVICE inline striding_ptr& operator--() {
		ptr = reinterpret_cast<pointer>( reinterpret_cast<typename cast_to_char<T*>::type>(ptr)-stride );
		//ptr -= stride;
		return *this;
	}
	HOST DEVICE inline striding_ptr operator--( int ) {
		striding_ptr tmp(*this);
		--(*this);
		return tmp;
	}

	HOST DEVICE inline striding_ptr& operator+=( const int strides ) {
		ptr = reinterpret_cast<pointer>( reinterpret_cast<typename cast_to_char<T*>::type>(ptr)+stride*strides );
		//ptr += stride*strides;
		return *this;
	}
	HOST DEVICE inline striding_ptr& operator-=( const int strides ) {
		ptr = reinterpret_cast<pointer>( reinterpret_cast<typename cast_to_char<T*>::type>(ptr)-stride*strides );
		//ptr -= stride*strides;
		return *this;
	}

	HOST DEVICE inline striding_ptr operator+( const int strides ) const {
		striding_ptr tmp(*this);
		tmp += strides;
		return tmp;
	}
	HOST DEVICE inline striding_ptr operator-( const int strides ) const {
		striding_ptr tmp(*this);
		tmp -= strides;
		return tmp;
	}

//	HOST DEVICE inline difference_type operator-( const striding_ptr& other ) const { return ptr-other.ptr; }

	DEVICE inline reference operator*() const { return *get(); }
	DEVICE inline pointer operator->() const { return get(); }

	HOST DEVICE inline bool operator==( const striding_ptr<T,PointerType>& other ) const { return ptr == other.ptr; }
	HOST DEVICE inline bool operator!=( const striding_ptr<T,PointerType>& other ) const { return ptr != other.ptr; }
	HOST DEVICE inline bool operator< ( const striding_ptr<T,PointerType>& other ) const { return ptr <  other.ptr; }
	HOST DEVICE inline bool operator> ( const striding_ptr<T,PointerType>& other ) const { return ptr >  other.ptr; }
	HOST DEVICE inline bool operator<=( const striding_ptr<T,PointerType>& other ) const { return ptr <= other.ptr; }
	HOST DEVICE inline bool operator>=( const striding_ptr<T,PointerType>& other ) const { return ptr >= other.ptr; }

	HOST DEVICE striding_ptr& operator=( const striding_ptr<T,PointerType>& other ) {
		ptr = other.ptr;
		stride = other.stride;
		return *this;
	}

	HOST DEVICE striding_ptr& operator=( PointerType& pt ) {
		ptr = pt;
		return *this;
	}

	HOST DEVICE striding_ptr& operator=( T* p ) {
		ptr = p;
		return *this;
	}

	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const striding_ptr& ptr ) {
		out << "striding_ptr(stride=" << ptr.stride << ";ptr=" << ptr.get() << ")";
		return out;
	}

};

} // namespace ecuda

#endif
