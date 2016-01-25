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
// ptr/padded_ptr.hpp
//
// A pointer specialization that describes device memory that is padded to
// comform to a given memory alignment.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#ifndef ECUDA_PTR_PADDED_PTR_HPP
#define ECUDA_PTR_PADDED_PTR_HPP

#include <iostream>

#include "../global.hpp"
#include "../type_traits.hpp"
#include "../utility.hpp"
#include "common.hpp"

namespace ecuda {

///
/// \brief A specialized pointer to padded memory.
///
/// A specialized pointer to device memory which is aligned to a certain width. Although this class stores this
/// width it is the responsibility of the user of the class to account for this padding when traversing the
/// data.
///
/// The specialization is used to both represent 2D memory allocations using cudaMallocPitch().
///
/// Memory use can be conceptualized as:
/// \code
///   |- width --|      // in multiples of sizeof(T)
///   |---- pitch ----| // in bytes
///   +----------+----+
///   |          |xxxx|
///   |          |xxxx| x = allocated but not used
///   |          |xxxx|
///   |          |xxxx|
///   |          |xxxx|
///   |          |xxxx| ... etc. (total size of the allocation is not known internally by padded_ptr)
///   +----------+----+
/// \endcode
///
/// This class stores the pitch, but not the width nor the position within the range that the current pointed to
/// element resides. The caller should store the relative position within the padded region and utilize the get_pitch()
/// and skip_bytes() methods to skip over the padded region when needed.
///
template<typename T,class P=typename ecuda::add_pointer<T>::type>
class padded_ptr
{
public:
	typedef T              element_type;
	typedef P              pointer;
	typedef T&             reference;
	typedef const T&       const_reference;
	typedef std::size_t    size_type;
	typedef std::ptrdiff_t difference_type;

private:
	template<typename U> struct char_pointer;
	template<typename U> struct char_pointer<U*>       { typedef char* type; };
	template<typename U> struct char_pointer<const U*> { typedef const char* type; };
	template<typename U> __HOST__ __DEVICE__ typename char_pointer<U*>::type       char_cast( U* ptr ) const       { return reinterpret_cast<char*>(ptr); }
	template<typename U> __HOST__ __DEVICE__ typename char_pointer<const U*>::type char_cast( const U* ptr ) const { return reinterpret_cast<const char*>(ptr); }

private:
	pointer ptr;     //!< pointer to current element
	size_type pitch; //!< bytes of padding at the end of the contiguous region

public:
	__HOST__ __DEVICE__	padded_ptr( pointer ptr = pointer(), size_type pitch = size_type() ) : ptr(ptr), pitch(pitch) {}

	__HOST__ __DEVICE__ padded_ptr( const padded_ptr& src ) : ptr(src.ptr), pitch(src.pitch) {}

	__HOST__ __DEVICE__ padded_ptr& operator=( const padded_ptr& src )
	{
		ptr = src.ptr;
		pitch = src.pitch;
		return *this;
	}

	template<typename T2,class P2>
	__HOST__ __DEVICE__ padded_ptr( const padded_ptr<T2,P2>& src ) : ptr(src.get()), pitch(src.get_pitch()) {}

	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ __DEVICE__ padded_ptr( padded_ptr&& src ) : ptr(std::move(src.ptr)), pitch(std::move(src.pitch)) {}
	__HOST__ __DEVICE__ padded_ptr& operator=( padded_ptr&& src )
	{
		ptr = std::move(src.ptr);
		pitch = std::move(src.pitch);
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline size_type   get_pitch() const { return pitch; }
	__HOST__ __DEVICE__ inline pointer     get() const         { return ptr; }

	__DEVICE__ inline reference       operator*()                       { return *ptr; }
	__DEVICE__ inline const_reference operator*() const                 { return *ptr; }
	__DEVICE__ inline pointer         operator->() const                { return ptr; }
	__DEVICE__ inline reference       operator[]( std::size_t i )       { return padded_ptr(*this).operator+=(i).operator*(); }
	__DEVICE__ inline const_reference operator[]( std::size_t i ) const { return padded_ptr(*this).operator+=(i).operator*(); }

	#ifdef ECUDA_CPP11_AVAILABLE
	///
	/// \brief Checks if this stores a non-null pointer.
	///
	/// \return true if *this stores a pointer, false otherwise.
	///
	__HOST__ __DEVICE__ explicit operator bool() const __NOEXCEPT__ { return naked_cast<typename ecuda::add_pointer<const element_type>::type>( ptr ) != NULL; }
	#else
	///
	/// \brief Checks if this stores a non-null pointer.
	///
	/// \return true if *this stores a pointer, false otherwise.
	///
	__HOST__ __DEVICE__ operator bool() const __NOEXCEPT__ { return naked_cast<typename ecuda::add_pointer<const element_type>::type>( ptr ) != NULL; }
	#endif

	__HOST__ __DEVICE__ inline padded_ptr& operator++() { ++ptr; return *this; }
	__HOST__ __DEVICE__ inline padded_ptr& operator--() { --ptr; return *this; }

	__HOST__ __DEVICE__ inline padded_ptr operator++( int ) { padded_ptr tmp(*this); ++(*this); return tmp; }
	__HOST__ __DEVICE__ inline padded_ptr operator--( int ) { padded_ptr tmp(*this); --(*this); return tmp; }

	__HOST__ __DEVICE__ inline padded_ptr& operator+=( int x ) { ptr += x; return *this; }
	__HOST__ __DEVICE__ inline padded_ptr& operator-=( int x ) { ptr -= x; return *this; }

	__HOST__ __DEVICE__ inline padded_ptr operator+( int x ) const
	{
		padded_ptr tmp( *this );
		tmp += x;
		return tmp;
	}

	__HOST__ __DEVICE__ inline padded_ptr operator-( int x ) const
	{
		padded_ptr tmp( *this );
		tmp -= x;
		return tmp;
	}

	__HOST__ __DEVICE__ inline padded_ptr operator+( std::size_t x ) const { return operator+( static_cast<int>(x) ); }
	__HOST__ __DEVICE__ inline padded_ptr operator-( std::size_t x ) const { return operator-( static_cast<int>(x) ); }

	template<typename T2,typename P2> __HOST__ __DEVICE__ bool operator==( const padded_ptr<T2,P2>& other ) const { return ptr == other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ bool operator!=( const padded_ptr<T2,P2>& other ) const { return ptr != other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ bool operator< ( const padded_ptr<T2,P2>& other ) const { return ptr <  other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ bool operator> ( const padded_ptr<T2,P2>& other ) const { return ptr >  other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ bool operator<=( const padded_ptr<T2,P2>& other ) const { return ptr <= other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ bool operator>=( const padded_ptr<T2,P2>& other ) const { return ptr >= other.ptr; }

	///
	/// \brief Move the pointer some number of bytes.
	///
	/// Remember the movement is always in bytes and doesn't consider
	/// sizeof(element_type).
	///
	/// \param x Number of bytes to move the pointer.
	///
	__HOST__ __DEVICE__ void skip_bytes( difference_type x )
	{
		typedef typename ecuda::add_pointer<element_type>::type raw_pointer_type;
		typedef typename char_pointer<raw_pointer_type>::type char_pointer_type;
		char_pointer_type charPtr = char_cast( naked_cast<raw_pointer_type>(ptr) );
		charPtr += x;
		ptr = pointer( naked_cast<raw_pointer_type>(charPtr) );
	}

	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const padded_ptr& ptr )
	{
		out << "padded_ptr(ptr=" << ptr.ptr << ";pitch=" << ptr.pitch << ")";
		return out;
	}

};


} // namespace ecuda

#endif
