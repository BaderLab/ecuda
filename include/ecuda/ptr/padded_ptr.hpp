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
/// A specialized pointer to device memory where traversal of the data takes into account an ignorable region
/// of padding after every fixed number of sequential elements.
///
/// The specialization is used to both represent 2D memory allocations using cudaMallocPitch() and to
/// create certain views of a cube (e.g. single row or column).
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
/// The template parameter P is the base pointer type. It will be T* by default, but can refer to other pointer
/// specializations (i.e. shared_ptr). However, if any increment/decrement operation is intended on the padded_ptr
/// then the type P must have a constructor that can take a raw pointer of type T* (otherwise P can't be realigned
/// to the start of the next contiguous block of memory once it hits the padded region).
///
template<typename T,class P=typename std::add_pointer<T>::type>
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
	//typedef char* address_type;
	//typedef typename std::add_pointer<T>::type raw_pointer;

private:
	//address_type edge_address;
	pointer edge_ptr;
	size_type pitch;
	size_type width;
	pointer ptr;

private:
	template<typename U,typename V> struct change_type_keep_constness;
	template<typename U,typename V> struct change_type_keep_constness<      U*,V*      > { typedef V* type; };
	template<typename U,typename V> struct change_type_keep_constness<      U*,const V*> { typedef V* type; };
	template<typename U,typename V> struct change_type_keep_constness<const U*,V*      > { typedef const V* type; };
	template<typename U,typename V> struct change_type_keep_constness<const U*,const V*> { typedef const V* type; };

	template<typename U> struct char_pointer;
	template<typename U> struct char_pointer<U*>       { typedef char* type; };
	template<typename U> struct char_pointer<const U*> { typedef const char* type; };
	template<typename U> __HOST__ __DEVICE__ typename char_pointer<U*>::type       char_cast( U* ptr ) const       { return reinterpret_cast<char*>(ptr); }
	template<typename U> __HOST__ __DEVICE__ typename char_pointer<const U*>::type char_cast( const U* ptr ) const { return reinterpret_cast<const char*>(ptr); }

public:
	__HOST__ __DEVICE__
	padded_ptr( pointer edge_ptr = pointer(), size_type pitch = size_type(), size_type width = size_type(), pointer ptr = pointer() ) :
		edge_ptr(edge_ptr), pitch(pitch), width(width), ptr(ptr)
	{
		if( !ptr ) this->ptr = edge_ptr;
	}

	__HOST__ __DEVICE__ padded_ptr( const padded_ptr& src ) : edge_ptr(src.edge_ptr), pitch(src.pitch), width(src.width), ptr(src.ptr) {}

	__HOST__ __DEVICE__ padded_ptr& operator=( const padded_ptr& src ) {
		edge_ptr = src.edge_ptr;
		pitch = src.pitch;
		width = src.width;
		ptr = src.ptr;
		return *this;
	}

	template<typename T2,class P2>
	__HOST__ __DEVICE__ padded_ptr( const padded_ptr<T2,P2>& src ) : edge_ptr(src.get_edge()), pitch(src.get_pitch()), width(src.get_width()), ptr(src.get()) {}

	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ padded_ptr( padded_ptr&& src ) : edge_ptr(ecuda::move(src.edge_ptr)), pitch(ecuda::move(src.pitch)), width(ecuda::move(src.width)), ptr(ecuda::move(src.ptr)) {}
	__HOST__ __DEVICE__ padded_ptr& operator=( padded_ptr&& src )
	{
		edge_ptr = ecuda::move(src.edge_ptr);
		pitch = ecuda::move(src.pitch);
		width = ecuda::move(src.width);
		ptr = ecuda::move(src.ptr);
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline pointer   get_edge() const  { return edge_ptr; }
	__HOST__ __DEVICE__ inline size_type get_pitch() const { return pitch; }
	__HOST__ __DEVICE__ inline size_type get_width() const { return width; }
	__HOST__ __DEVICE__ inline pointer   get() const       { return ptr; }

	__HOST__ __DEVICE__ inline size_type get_remaining_width() const { return width-(ptr-edge_ptr); }

	__DEVICE__ inline reference       operator*()                       { return *ptr; }
	__DEVICE__ inline const_reference operator*() const                 { return *ptr; }
	__DEVICE__ inline pointer         operator->() const                { return ptr; }
	__DEVICE__ inline reference       operator[]( std::size_t i )       { return padded_ptr(*this).operator+=(i).operator*(); }
	__DEVICE__ inline const_reference operator[]( std::size_t i ) const { return padded_ptr(*this).operator+=(i).operator*(); }

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Checks if this stores a non-null pointer.
	///
	/// \return true if *this stores a pointer, false otherwise.
	///
	__HOST__ __DEVICE__ explicit operator bool() const __NOEXCEPT__ { return naked_cast<typename std::add_pointer<const element_type>::type>( ptr ) != NULL; }
	#else
	///
	/// \brief Checks if this stores a non-null pointer.
	///
	/// \return true if *this stores a pointer, false otherwise.
	///
	__HOST__ __DEVICE__ operator bool() const __NOEXCEPT__ { return naked_cast<typename std::add_pointer<const element_type>::type>( ptr ) != NULL; }
	#endif

	///
	/// This method makes several assumptions which, if violated, cause undefined behavior:
	/// 1) the width of this and other must be the same
	/// 2) the difference between the edge_ptr of this and other must be zero or strict multiple (positive or negative) of pitch/sizeof(T)
	///
	__HOST__ __DEVICE__ difference_type operator-( const padded_ptr& other ) const {
		//const difference_type remainder1 = static_cast<difference_type>(other.get_remaining_width());
		//const difference_type remainder2 = static_cast<difference_type>(ptr-edge_ptr);
		//const difference_type middle     = edge_ptr - other.edge_ptr - get_width();
		//return remainder1+remainder2+middle;
//TODO: look for optimizations here
		const difference_type leading  = static_cast<difference_type>( other.get_remaining_width() );
		const difference_type trailing = static_cast<difference_type>( get_width() - get_remaining_width() );
		const difference_type middle   = ( ( naked_cast<const char*>(edge_ptr) - naked_cast<const char*>(other.edge_ptr) ) / get_pitch() - 1 ) * get_width();
		return ( leading + middle + trailing );
/*
		const difference_type remainder1 = static_cast<difference_type>()
		std::cerr << "operator- other.get_remaining_width()=" << other.get_remaining_width() << std::endl;
		std::cerr << "operator- (ptr - other.edge_ptr)=" << ( ptr - other.edge_ptr ) << std::endl;
		std::cerr << "operator- get_width()=" << get_width() << std::endl;
		return static_cast<difference_type>(other.get_remaining_width()) + ( ptr - other.edge_ptr ) - get_width();
*/
	}

	/*
	template<class Q> // assumes sizeof(Q) is a strict multiple of sizeof(T) TODO: can probably enforce this with an enable_if
	__HOST__ __DEVICE__ inline difference_type operator-( Q other ) {
		//typename pointer_traits<pointer>::naked_pointer start = reinterpret_cast<typename pointer_traits<pointer>::naked_pointer>( pointer_traits<Q>().undress(other) );
		//typename pointer_traits<pointer>::naked_pointer stop  = pointer_traits<pointer>().undress(ptr);
		typedef typename std::add_pointer<element_type>::type naked_pointer_type;
		naked_pointer_type start = naked_cast<naked_pointer_type>( other );
		naked_pointer_type stop  = naked_cast<naked_pointer_type>( ptr );
		difference_type n = (stop-start)*sizeof(T); // bytes difference
		return n;
	}
	*/

/*
	template<class Q> // assumes sizeof(Q) is a strict multiple of sizeof(T) TODO: can probably enforce this with an enable_if
	__HOST__ __DEVICE__ inline difference_type operator-( Q p ) {
		typename pointer_traits<pointer>::naked_pointer start = reinterpret_cast<typename pointer_traits<pointer>::naked_pointer>( pointer_traits<Q>().undress(p) );
		typename pointer_traits<pointer>::naked_pointer stop = pointer_traits<pointer>().undress(ptr);
		difference_type n = (stop-start)*sizeof(T); // bytes difference
		n -= n/pitch * (pitch-width*sizeof(T)); // substract pads
		// assumption comes into play here, where n/sizeof(T) should evaluate exactly
		// i.e. ( n % sizeof(T) ) > 0 should never be true
		n /= sizeof(T);
		// delegate to underlying pointer specializations to allow it
		// a chance to apply logic
		return pointer(ptr+n)-p;
	}
*/
	__HOST__ __DEVICE__ inline padded_ptr& operator++()
	{
		++ptr;
		if( (ptr-edge_ptr) == width ) {
			// skip padding
			typedef typename std::add_pointer<element_type>::type raw_pointer_type;
			raw_pointer_type rawPtr = naked_cast<raw_pointer_type>(edge_ptr);
			typename char_pointer<raw_pointer_type>::type charPtr = char_cast( rawPtr );
			charPtr += pitch; // advance to start of next contiguous region
			ptr = pointer( naked_cast<raw_pointer_type>(charPtr) );
			edge_ptr = ptr;
			//ptr = pointer( naked_cast<typename std::add_pointer<element_type>::type>( naked_cast<typename change_type_keep_constness<pointer,char*>::type>(edge_ptr)+pitch ) );
			//edge_ptr = ptr;
		}
		return *this;
	}

	__HOST__ __DEVICE__ inline padded_ptr& operator--()
	{
		--ptr;
		if( ptr < edge_ptr ) {
			// skip padding
			edge_ptr = pointer( naked_cast<typename std::add_pointer<element_type>::type>( naked_cast<typename change_type_keep_constness<pointer,char*>::type>(edge_ptr)-pitch ) );
			ptr = edge_ptr + width - 1;
		}
		return *this;
	}

	__HOST__ __DEVICE__ inline padded_ptr operator++( int ) { padded_ptr tmp(*this); ++(*this); return tmp; }
	__HOST__ __DEVICE__ inline padded_ptr operator--( int ) { padded_ptr tmp(*this); --(*this); return tmp; }

	__HOST__ __DEVICE__ inline padded_ptr& operator+=( int x ) {
		ptr += x;
		if( (ptr-edge_ptr) >= width ) {
			// skip padding(s)
			//if( width == 0 ) throw std::runtime_error("width is zero");
			const size_type nskips = (ptr-edge_ptr) / width;
			const size_type offset = (ptr-edge_ptr) % width;
			typedef typename std::add_pointer<element_type>::type raw_pointer_type;
			raw_pointer_type rawPtr = naked_cast<raw_pointer_type>(edge_ptr);
			typename char_pointer<raw_pointer_type>::type charPtr = char_cast( rawPtr );
			charPtr += nskips * pitch;
			edge_ptr = pointer( naked_cast<raw_pointer_type>(charPtr) );
			charPtr += offset;
			ptr = edge_ptr + offset;
			//char* p = naked_cast<char*>( ptr );
			//p += nskips * pitch;
			//edge_ptr = pointer( p );
			//edge_ptr = pointer( naked_cast<typename std::add_pointer<element_type>::type>( naked_cast<typename change_type_keep_constness<pointer,char*>::type>(edge_ptr)+nskips*pitch ) );
			//ptr = edge_ptr + offset;
		}
		return *this;
	}

	__HOST__ __DEVICE__ inline padded_ptr& operator-=( int x ) {
		ptr -= x;
		if( ptr < edge_ptr ) {
			// skip padding(s)
			const size_type nskips = (edge_ptr-ptr) / width;
			const size_type offset = (edge_ptr-ptr) % width;
			edge_ptr = pointer( naked_cast<typename std::add_pointer<element_type>::type>( naked_cast<typename change_type_keep_constness<pointer,char*>::type>(edge_ptr)-nskips*pitch ) );
			ptr = edge_ptr + (width-offset);
		}
		return *this;
	}

	__HOST__ __DEVICE__ inline padded_ptr operator+( int x ) const {
		padded_ptr tmp( *this );
		tmp += x;
		return tmp;
	}

	__HOST__ __DEVICE__ inline padded_ptr operator-( int x ) const {
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

	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const padded_ptr& ptr ) {
		out << "padded_ptr(edge_ptr=" << ptr.edge_ptr << ";pitch=" << ptr.pitch << ";width=" << ptr.width << ";ptr=" << ptr.ptr << ")";
		return out;
	}

};


} // namespace ecuda

#endif
