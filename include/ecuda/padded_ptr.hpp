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
// padded_ptr.hpp
//
// Pointer specialization that causes increment/decrement operations to
// ignore a fixed number of bytes after a certain number of units of
// the size of the type pointed to are read through.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_PADDED_PTR_HPP
#define ECUDA_PADDED_PTR_HPP

//#include <cstddef>
#include "global.hpp"
//#include "iterators.hpp"
//#include "device_ptr.hpp"

//#ifdef __CPP11_SUPPORTED__
//#include <memory>
//#endif

namespace ecuda {

///
/// Utility to convert arbitrary pointer type to char* whilst maintaining constness.
///
/// NOTE: C++11 has cool semantics via type_traits and enable_if that can accomplish this, but
///       this is a less elegant method that works with C98 and later.
///
template<typename T> struct cast_to_char;
template<typename T> struct cast_to_char<T*> { typedef char* type; };
template<typename T> struct cast_to_char<const T*> { typedef const char* type; };

template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type,std::size_t PaddingUnitBytes=sizeof(T)>
class padded_ptr {

public:
	typedef T element_type; //!< data type represented in allocated memory
	typedef PointerType pointer; //!< data type pointer
	typedef T& reference; //!< data type reference
	typedef std::size_t size_type; //!< size type for pointer arithmetic and reference counting
	typedef std::ptrdiff_t difference_type; //!< signed integer type of the result of subtracting two pointers

private:
	pointer ptr;
	const size_type data_length; // contiguous elements of data before pad, expressed in units of element_type
	const size_type padding_length;  // contiguous elements of padding after data, expressed in PaddingUnitBytes
	size_type distance_to_padding; // distance of current pointer from the padding, expressed in units of element_type

private:
	HOST DEVICE void jump_forward_pad_length() {
		T* p = static_cast<T*>(ptr);
		typename cast_to_char<T*>::type char_ptr = reinterpret_cast<typename cast_to_char<T*>::type>(p);
		//typename cast_to_char<T*>::type char_ptr = reinterpret_cast<typename cast_to_char<T*>::type>(ptr);
		char_ptr += padding_length*PaddingUnitBytes;
		ptr = reinterpret_cast<T*>(char_ptr);
	}
	HOST DEVICE void jump_backwards_pad_length() {
		T* p = static_cast<T*>(ptr);
		typename cast_to_char<T*>::type char_ptr = reinterpret_cast<typename cast_to_char<T*>::type>(p);
		//typename cast_to_char<T*>::type char_ptr = reinterpret_cast<typename cast_to_char<T*>::type>(ptr);
		char_ptr -= padding_length*PaddingUnitBytes;
		ptr = reinterpret_cast<T*>(char_ptr);
	}
	HOST DEVICE inline typename cast_to_char<T*>::type to_char_ptr() const {
		T* p = static_cast<T*>(ptr);
		return reinterpret_cast<typename cast_to_char<T*>::type>(p);
		//return reinterpret_cast<typename cast_to_char<T*>::type>(ptr);
	}

public:
	HOST DEVICE padded_ptr( pointer p = pointer(), const size_type data_length = 1, const size_type padding_length = 0, const size_type pointer_position = 0 ) :
		ptr(p),
		data_length(data_length),
		padding_length(padding_length),
		distance_to_padding(data_length-pointer_position)
	{
	}
	HOST DEVICE padded_ptr( const padded_ptr<T,PointerType,PaddingUnitBytes>& src ) : ptr(src.ptr), data_length(src.data_length), padding_length(src.padding_length), distance_to_padding(src.distance_to_padding) {}
	HOST DEVICE ~padded_ptr() {}

	HOST DEVICE inline size_type get_data_length() const { return data_length; }
	HOST DEVICE inline size_type get_padding_length() const { return padding_length; }
	HOST DEVICE inline __CONSTEXPR__ size_type get_pad_length_units() const { return PaddingUnitBytes; }
	HOST DEVICE inline size_type get_distance_to_padding() const { return distance_to_padding; }

	HOST DEVICE inline pointer get() const { return ptr; }
	HOST DEVICE inline operator bool() const { return ptr != nullptr; }
	HOST DEVICE inline operator pointer() const { return ptr; }

	HOST DEVICE inline padded_ptr& operator++() {
		++ptr;
		--distance_to_padding;
		if( !distance_to_padding ) {
			jump_forward_pad_length();
			distance_to_padding = data_length;
		}
		return *this;
	}
	HOST DEVICE inline padded_ptr operator++( int ) {
		padded_ptr tmp(*this);
		++(*this);
		return tmp;
	}

	HOST DEVICE inline padded_ptr& operator--() {
		if( distance_to_padding == data_length ) {
			jump_backwards_pad_length();
			distance_to_padding = 0;
		}
		--ptr;
		++distance_to_padding;
		return *this;
	}
	HOST DEVICE inline padded_ptr operator--( int ) {
		padded_ptr tmp(*this);
		--(*this);
		return tmp;
	}

	HOST DEVICE padded_ptr& operator+=( int units ) {
		if( units >= distance_to_padding ) {
			units -= distance_to_padding;
			ptr += distance_to_padding;
			jump_forward_pad_length();
			distance_to_padding = data_length;
			return operator+=( units );
		}
		ptr += units;
		distance_to_padding -= units;
		return *this;
	}

	HOST DEVICE padded_ptr& operator-=( int units ) {
		const difference_type distance_from_start = data_length-distance_to_padding;
		if( units > distance_from_start ) {
			units -= distance_from_start;
			ptr -= distance_from_start;
			jump_backwards_pad_length();
			distance_to_padding = 0;
			return operator-=( units );
		}
		ptr -= units;
		distance_to_padding += units;
		return *this;
	}

	HOST DEVICE inline padded_ptr operator+( const int units ) const { return padded_ptr(*this).operator+=(units); }
	HOST DEVICE inline padded_ptr operator-( const int units ) const { return padded_ptr(*this).operator-=(units); }

	///
	/// \brief operator-
	///
	/// Note this will always be expressed in bytes, regardless of the size of element_type.
	///
	/// \param other
	/// \return
	///
	HOST DEVICE inline difference_type operator-( const padded_ptr& other ) const { return to_char_ptr() - other.to_char_ptr(); }

	DEVICE inline reference operator*() const { return *get(); }
	DEVICE inline pointer operator->() const { return get(); }

	template<std::size_t PaddingUnitBytes2> HOST DEVICE inline bool operator==( const padded_ptr<T,PointerType,PaddingUnitBytes2>& other ) const { return ptr == other.ptr; }
	template<std::size_t PaddingUnitBytes2> HOST DEVICE inline bool operator!=( const padded_ptr<T,PointerType,PaddingUnitBytes2>& other ) const { return ptr != other.ptr; }
	template<std::size_t PaddingUnitBytes2> HOST DEVICE inline bool operator< ( const padded_ptr<T,PointerType,PaddingUnitBytes2>& other ) const { return ptr <  other.ptr; }
	template<std::size_t PaddingUnitBytes2> HOST DEVICE inline bool operator> ( const padded_ptr<T,PointerType,PaddingUnitBytes2>& other ) const { return ptr >  other.ptr; }
	template<std::size_t PaddingUnitBytes2> HOST DEVICE inline bool operator<=( const padded_ptr<T,PointerType,PaddingUnitBytes2>& other ) const { return ptr <= other.ptr; }
	template<std::size_t PaddingUnitBytes2> HOST DEVICE inline bool operator>=( const padded_ptr<T,PointerType,PaddingUnitBytes2>& other ) const { return ptr >= other.ptr; }

};

} // namespace ecuda

#endif
