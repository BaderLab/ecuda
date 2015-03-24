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
// memory.hpp
//
// Smart pointers to device memory and simple homespun unique_ptr in absense
// of C++11 memory library.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MEMORY_HPP
#define ECUDA_MEMORY_HPP

#include <cstddef>
#include "global.hpp"
#include "iterators.hpp"
#include "device_ptr.hpp"
#include "striding_ptr.hpp"
#include "padded_ptr.hpp"

#ifdef __CPP11_SUPPORTED__
#include <memory>
#endif

namespace ecuda {

///
/// A proxy to a pre-allocated block of contiguous memory.
///
/// The class merely holds the pointer and the length of the block
/// and provides the means to manipulate the sequence.  No
/// allocation/deallocation is done.
///
template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type>
class contiguous_memory_proxy
{
public:
	typedef T value_type; //!< The first template parameter (T)
	typedef PointerType pointer;
	//typedef value_type* pointer; //!< value_type*
	typedef value_type& reference; //!< value_type&
	//typedef const PointerType const_pointer;
	// nvcc V6.0.1 produced a warning "type qualifiers are meaningless here", but above replacement line is fine
	//typedef const pointer const_pointer; //!< const value_type*
	typedef const T& const_reference;
	// nvcc V6.0.1 produced a warning "type qualifiers are meaningless here", but above replacement line is fine
	//typedef const reference const_reference; //!< const value_type&
	typedef std::ptrdiff_t difference_type;
	typedef std::size_t size_type;

	typedef pointer_iterator<value_type,pointer> iterator;
	typedef pointer_iterator<const value_type,pointer> const_iterator;
	typedef pointer_reverse_iterator<iterator> reverse_iterator;
	typedef pointer_reverse_iterator<const_iterator> const_reverse_iterator;

protected:
	pointer ptr;
	size_type length;

public:
	HOST DEVICE contiguous_memory_proxy() : ptr(nullptr), length(0) {}
	template<typename T2,typename PointerType2>
	HOST DEVICE contiguous_memory_proxy( const contiguous_memory_proxy<T2,PointerType2>& src ) : ptr(src.data()), length(src.size()) {}
	HOST DEVICE contiguous_memory_proxy( pointer ptr, size_type length ) : ptr(ptr), length(length) {}
	/*
	template<class Container>
	HOST DEVICE contiguous_memory_proxy(
		Container& container,
		typename Container::size_type length, // = typename Container::size_type(),
		typename Container::size_type offset = typename Container::size_type()
	) : ptr(container.data()+offset), length(length) {}
	*/
	HOST DEVICE virtual ~contiguous_memory_proxy() {}

	HOST DEVICE pointer data() const { return ptr; }

	// iterators:
	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(ptr); }
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator(ptr+static_cast<int>(length)); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(ptr); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(ptr+static_cast<int>(length)); }
	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(iterator(ptr+static_cast<int>(length))); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(iterator(ptr)); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(ptr+static_cast<int>(length))); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(ptr)); }

	#ifdef __CPP11_SUPPORTED__
	HOST DEVICE inline const_iterator cbegin() const __NOEXCEPT__ { return const_iterator(ptr); }
	HOST DEVICE inline const_iterator cend() const __NOEXCEPT__ { return const_iterator(ptr+static_cast<int>(length)); }
	HOST DEVICE inline const_reverse_iterator crbegin() __NOEXCEPT__ { return const_reverse_iterator(const_iterator(ptr+static_cast<int>(length))); }
	HOST DEVICE inline const_reverse_iterator crend() __NOEXCEPT__ { return const_reverse_iterator(const_iterator(ptr)); }
	#endif

	// capacity:
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return length; }
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return length == 0; }

	// element access:
	DEVICE inline reference operator[]( size_type index ) { return *(ptr+static_cast<int>(index)); }
	DEVICE inline reference at( size_type index ) { return operator[]( index ); }
	DEVICE inline reference front() { return operator[](0); }
	DEVICE inline reference back() { return operator[](size()-1); }
	DEVICE inline const_reference operator[]( size_type index ) const {	return *(ptr+static_cast<int>(index)); }
	DEVICE inline const_reference at( size_type index ) const { return operator[]( index ); }
	DEVICE inline const_reference front() const { return operator[](0); }
	DEVICE inline const_reference back() const { return operator[](size()-1); }

	HOST DEVICE contiguous_memory_proxy& operator=( const contiguous_memory_proxy& other ) {
		ptr = other.ptr;
		length = other.length;
		return *this;
	}

};

///
///
///
template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type>
class contiguous_2d_memory_proxy : public contiguous_memory_proxy<T,PointerType>
{
protected:
	typedef contiguous_memory_proxy<T,PointerType> base_type;

public:
	typedef typename base_type::value_type value_type;
	typedef typename base_type::pointer pointer;
	typedef typename base_type::reference reference;
	//typedef typename base_type::const_pointer const_pointer;
	typedef typename base_type::const_reference const_reference;
	typedef typename base_type::difference_type difference_type;
	typedef typename base_type::size_type size_type;

	typedef typename base_type::iterator iterator;
	typedef typename base_type::const_iterator const_iterator;
	typedef typename base_type::reverse_iterator reverse_iterator;
	typedef typename base_type::const_reverse_iterator const_reverse_iterator;

	typedef contiguous_memory_proxy<T,PointerType> row_type;
	typedef contiguous_memory_proxy<const T,PointerType> const_row_type;
	//typedef strided_memory_proxy<T> column_type;

protected:
	size_type height; // cf. height
	//size_type pitch;

public:
	HOST DEVICE contiguous_2d_memory_proxy() : contiguous_memory_proxy<T>(), height(0) {}
	template<typename U>
	HOST DEVICE contiguous_2d_memory_proxy( const contiguous_2d_memory_proxy<U>& src ) : contiguous_memory_proxy<T,PointerType>(src), height(src.height) {}
	HOST DEVICE contiguous_2d_memory_proxy( pointer ptr, size_type width, size_type height ) : contiguous_memory_proxy<T,PointerType>(ptr,width*height), height(height) {}
	HOST DEVICE virtual ~contiguous_2d_memory_proxy() {}

	// capacity:
	HOST DEVICE inline size_type get_width() const { return contiguous_memory_proxy<T,PointerType>::size()/height; }
	HOST DEVICE inline size_type get_height() const { return height; }
	//HOST DEVICE inline size_type size() const { return height*base_type::size(); }

	// element access:
	HOST DEVICE inline row_type operator[]( size_type index ) {
		pointer ptr = base_type::data();
		ptr += index*get_width();
		return row_type( ptr, get_width() );
	}
	HOST DEVICE inline const_row_type operator[]( size_type index ) const {
		pointer ptr = base_type::data();
		ptr += index*get_width();
		return const_row_type( ptr, get_width() );
	}

	HOST DEVICE contiguous_2d_memory_proxy& operator=( const contiguous_2d_memory_proxy& other ) {
		base_type::operator=( other );
		height = other.height;
		return *this;
	}

};

#ifdef __CPP11_SUPPORTED__
// some future proofing for the glorious day when
// nvcc will support C++11 and we can just use the
// prepackaged implementations
template<typename T> typedef std::unique_ptr<T> unique_ptr<T>;
template<typename T> typedef std::unique_ptr<T[]> unique_ptr<T[]>;
#else
template<typename T>
class unique_ptr {

public:
	typedef T element_type;
	typedef T* pointer;
	typedef T& reference;

private:
	T* ptr;

public:
	HOST DEVICE unique_ptr( T* ptr=NULL ) : ptr(ptr) {}
	HOST DEVICE ~unique_ptr() {
		#ifndef __CUDA_ARCH__
		if( ptr ) delete ptr;
		#endif
	}

	HOST DEVICE inline pointer get() const { return ptr; }
	HOST DEVICE inline operator bool() const { return get() != NULL; }
	HOST DEVICE inline reference operator*() const { return *ptr; }
	HOST DEVICE inline pointer operator->() const { return ptr; }

	HOST DEVICE inline bool operator==( const unique_ptr<T>& other ) const { return ptr == other.ptr; }
	HOST DEVICE inline bool operator!=( const unique_ptr<T>& other ) const { return ptr != other.ptr; }
	HOST DEVICE inline bool operator<( const unique_ptr<T>& other ) const { return ptr < other.ptr; }
	HOST DEVICE inline bool operator>( const unique_ptr<T>& other ) const { return ptr > other.ptr; }
	HOST DEVICE inline bool operator<=( const unique_ptr<T>& other ) const { return ptr <= other.ptr; }
	HOST DEVICE inline bool operator>=( const unique_ptr<T>& other ) const { return ptr >= other.ptr; }

	HOST DEVICE unique_ptr<T>& operator=( T* p ) {
		ptr = p;
		return *this;
	}

};

template<typename T>
class unique_ptr<T[]> {

public:
	typedef T element_type;
	typedef T* pointer;
	typedef T& reference;
	typedef std::size_t size_type;

private:
	T* ptr;

public:
	HOST DEVICE unique_ptr<T[]>( T* ptr=NULL ) : ptr(ptr) {}
	//unique_ptr<T[]>( const unique_ptr<T[]>& src ) : ptr(src.ptr) {}
	HOST DEVICE ~unique_ptr<T[]>() { if( ptr ) delete [] ptr; }

	HOST DEVICE inline pointer get() const { return ptr; }
	HOST DEVICE inline operator bool() const { return get() != NULL; }
	HOST DEVICE inline reference operator[]( const size_type index ) const { return *(ptr+index); }

	HOST DEVICE inline bool operator==( const unique_ptr<T[]>& other ) const { return ptr == other.ptr; }
	HOST DEVICE inline bool operator!=( const unique_ptr<T[]>& other ) const { return ptr != other.ptr; }
	HOST DEVICE inline bool operator<( const unique_ptr<T[]>& other ) const { return ptr < other.ptr; }
	HOST DEVICE inline bool operator>( const unique_ptr<T[]>& other ) const { return ptr > other.ptr; }
	HOST DEVICE inline bool operator<=( const unique_ptr<T[]>& other ) const { return ptr <= other.ptr; }
	HOST DEVICE inline bool operator>=( const unique_ptr<T[]>& other ) const { return ptr >= other.ptr; }

	HOST DEVICE unique_ptr<T>& operator=( T* p ) {
		ptr = p;
		return *this;
	}

};
#endif

} // namespace ecuda

#endif
