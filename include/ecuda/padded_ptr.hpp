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

#include "global.hpp"

/*
namespace ecuda {

///
/// \brief A specialized pointer to memory with padding after a fixed-size sequence of data.
///
template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type,std::size_t PaddingUnitBytes=sizeof(T)>
class padded_ptr {

public:
	typedef T element_type; //!< data type represented in allocated memory
	typedef PointerType pointer; //!< data type pointer
	typedef T& reference; //!< data type reference
	typedef std::size_t size_type; //!< size type for pointer arithmetic and reference counting
	typedef std::ptrdiff_t difference_type; //!< signed integer type of the result of subtracting two pointers

private:
	pointer ptr; //!< underlying pointer
	const size_type padding; // padding memory width in bytes
	const size_type width; // width of grid in elements
	size_type current_position; // position of pointer in the grid

public:
	///
	/// \brief Default constructor.
	///
	/// The pointer p points to a location within a block of padded memory where
	/// every data_length contiguous elements of type T are followed by a block
	/// of ignorable memory that is padding_length units of size PaddingUnitBytes.
	/// pointer_position specifies the current location of the pointer within a
	/// data block.
	///
	/// \param p a pointer to an element of type T
	/// \param data_length the number of contiguous elements of type T that comprise the data block
	/// \param padding_length the number of PaddingUnitBytes-sized elements of padding following each data block
	/// \param pointer_position the index of the pointer within the current data block
	///
	///
	HOST DEVICE padded_ptr( pointer p, const size_type pitch, const size_type width, const size_type current_position = 0 ) :
		ptr(p),
		padding( (pitch*PaddingUnitBytes-width*sizeof(element_type))/PaddingUnitBytes ),
		width(width),
		current_position(current_position)
	{
	}

	///
	/// \brief Copy constructor.
	///
	HOST DEVICE padded_ptr( const padded_ptr& src ) : ptr(src.ptr), padding(src.padding), width(src.width), current_position(src.current_position) {}

	template<typename T2,typename PointerType2>
	HOST DEVICE padded_ptr( const padded_ptr<T2,PointerType2,PaddingUnitBytes>& src ) : ptr(src.get()), padding(src.get_padding()), width(src.get_width()), current_position(src.get_current_position()) {}

	HOST DEVICE inline size_type get_pitch() const { return sizeof(element_type)*width+padding; }
	HOST DEVICE inline size_type get_padding() const { return padding; }
	HOST DEVICE inline size_type get_width() const { return width; }
	HOST DEVICE inline size_type get_current_position() const { return current_position; }
	HOST DEVICE inline __CONSTEXPR__ size_type get_pitch_units() const { return PaddingUnitBytes; }

	HOST DEVICE inline pointer get() const { return ptr; }

	HOST DEVICE inline operator bool() const { return ptr != pointer(); } //nullptr; }

	HOST DEVICE inline operator pointer() const { return ptr; }

	HOST DEVICE inline padded_ptr& operator++() {
		++ptr;
		++current_position;
		if( current_position == width ) {
			T* p = static_cast<T*>(ptr);
			typename cast_to_char<T*>::type char_ptr = reinterpret_cast<typename cast_to_char<T*>::type>(p);
			char_ptr += padding;
			ptr = reinterpret_cast<T*>(char_ptr);
			current_position = 0;
		}
		return *this;
	}
	HOST DEVICE inline padded_ptr operator++( int ) {
		padded_ptr tmp(*this);
		++(*this);
		return tmp;
	}

	HOST DEVICE inline padded_ptr& operator--() {
		--ptr;
		if( current_position == 0 ) {
			T* p = static_cast<T*>(ptr);
			typename cast_to_char<T*>::type char_ptr = reinterpret_cast<typename cast_to_char<T*>::type>(p);
			char_ptr -= padding;
			ptr = reinterpret_cast<T*>(char_ptr);
			current_position = width-1;
		}
		return *this;
	}
	HOST DEVICE inline padded_ptr operator--( int ) {
		padded_ptr tmp(*this);
		--(*this);
		return tmp;
	}

	HOST DEVICE padded_ptr& operator+=( int units ) {
		if( (current_position+units) >= width ) {
			T* p = static_cast<T*>(ptr);
			typename cast_to_char<T*>::type char_ptr = reinterpret_cast<typename cast_to_char<T*>::type>(p);
			char_ptr += units/width*padding;
			ptr = reinterpret_cast<T*>(char_ptr);
		}
		current_position += units;
		current_position = current_position % width;
		ptr += units;
		return *this;
	}

	HOST DEVICE padded_ptr& operator-=( int units ) {
		if( units < current_position ) {
			T* p = static_cast<T*>(ptr);
			typename cast_to_char<T*>::type char_ptr = reinterpret_cast<typename cast_to_char<T*>::type>(p);
			char_ptr -= units/width*padding;
			ptr = reinterpret_cast<T*>(char_ptr);
		}
		current_position -= (width-(units % width));
		ptr -= units;
		return *this;
	}

	HOST DEVICE inline padded_ptr operator+( const int units ) const { return padded_ptr(*this).operator+=(units); }
	HOST DEVICE inline padded_ptr operator-( const int units ) const { return padded_ptr(*this).operator-=(units); }

	DEVICE inline reference operator*() const { return *get(); }
	DEVICE inline pointer operator->() const { return get(); }

	template<std::size_t PaddingUnitBytes2> HOST DEVICE inline bool operator==( const padded_ptr<T,PointerType,PaddingUnitBytes2>& other ) const { return ptr == other.ptr; }
	template<std::size_t PaddingUnitBytes2> HOST DEVICE inline bool operator!=( const padded_ptr<T,PointerType,PaddingUnitBytes2>& other ) const { return ptr != other.ptr; }
	template<std::size_t PaddingUnitBytes2> HOST DEVICE inline bool operator< ( const padded_ptr<T,PointerType,PaddingUnitBytes2>& other ) const { return ptr <  other.ptr; }
	template<std::size_t PaddingUnitBytes2> HOST DEVICE inline bool operator> ( const padded_ptr<T,PointerType,PaddingUnitBytes2>& other ) const { return ptr >  other.ptr; }
	template<std::size_t PaddingUnitBytes2> HOST DEVICE inline bool operator<=( const padded_ptr<T,PointerType,PaddingUnitBytes2>& other ) const { return ptr <= other.ptr; }
	template<std::size_t PaddingUnitBytes2> HOST DEVICE inline bool operator>=( const padded_ptr<T,PointerType,PaddingUnitBytes2>& other ) const { return ptr >= other.ptr; }

	//HOST DEVICE padded_ptr& operator=( const padded_ptr<T,PointerType,PaddingUnitBytes>& other ) {
	//	ptr = other.ptr;
		//padding = other.padding;
		//width = other.width;
	//	current_position = other.current_position;
	//	return *this;
	//}

	HOST DEVICE padded_ptr& operator=( PointerType& pt ) {
		ptr = pt;
		return *this;
	}

	HOST DEVICE padded_ptr& operator=( T* p ) {
		ptr = p;
		return *this;
	}

	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const padded_ptr& ptr ) {
		out << "padded_ptr(current_position=" << ptr.current_position;
		out << ";width=" << ptr.width;
		out << ";padding=" << ptr.padding;
		out << ";units=" << PaddingUnitBytes;
		out << ";ptr=" << ptr.get();
		out << ")";
		return out;
	}

};

} // namespace ecuda
*/

#endif
