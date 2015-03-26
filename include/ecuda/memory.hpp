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
#include "apiwrappers.hpp"
#include "iterators.hpp"
#include "device_ptr.hpp"
#include "striding_ptr.hpp"
#include "padded_ptr.hpp"

namespace ecuda {

///
/// \brief A temporary array comprised of a pointer and size.
///
/// Acts as a standalone representation of a linear fixed-size series of values
/// given a pointer and the desired size. Used to generate subsequences from
/// a larger memory structure (e.g. an individual row of a larger matrix). This
/// is a contrived structure to provide array-like operations, no
/// allocation/deallocation is done.
///
template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type>
class temporary_array
{
public:
	typedef T value_type; //!< element data type
	typedef PointerType pointer; //!< element pointer type
	typedef value_type& reference; //!< element reference type
	typedef const value_type& const_reference; //!< const element reference type
	typedef std::size_t size_type; //!< unsigned integral type
	typedef std::ptrdiff_t difference_type; //!< signed integral type

	typedef pointer_iterator<value_type,pointer> iterator; //!< iterator type
	typedef pointer_iterator<const value_type,const pointer> const_iterator; //!< const iterator type
	typedef pointer_reverse_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef pointer_reverse_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

protected:
	pointer ptr; //!< pointer to the start of the array
	size_type length; //!< number of elements in the array

public:
	HOST DEVICE temporary_array() : ptr(nullptr), length(0) {}
	template<typename T2,typename PointerType2>
	HOST DEVICE temporary_array( const temporary_array<T2,PointerType2>& src ) : ptr(src.data()), length(src.size()) {}
	HOST DEVICE temporary_array( pointer ptr, size_type length ) : ptr(ptr), length(length) {}
	HOST DEVICE ~temporary_array() {}

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

	HOST DEVICE temporary_array& operator=( const temporary_array& other ) {
		ptr = other.ptr;
		length = other.length;
		return *this;
	}

};


///
/// \brief A specialized temporary array guaranteed to be comprised of a contiguous sequence of memory.
///
/// This is a subclass of temporary_array that imposes the requirement that the
/// underlying pointer refers to a contiguous block of memory.  Thus, PointerType
/// (the second template parameter of temporary_array) is strictly defined as a
/// naked pointer of type T*.
///
/// This allows the assign method to be made available safely.
///
template<typename T>
class contiguous_temporary_array : public temporary_array<T,T*>
{
private:
	typedef temporary_array<T,T*> base_type;

public:
	typedef typename temporary_array<T,T*>::value_type value_type; //!< element data type
	typedef typename temporary_array<T,T*>::pointer pointer; //!< element pointer type
	typedef typename temporary_array<T,T*>::reference reference; //!< element reference type
	typedef typename temporary_array<T,T*>::const_reference const_reference; //!< const element reference type
	typedef typename temporary_array<T,T*>::size_type size_type; //!< unsigned integral type
	typedef typename temporary_array<T,T*>::difference_type difference_type; //!< signed integral type

	typedef contiguous_pointer_iterator<T> iterator; //!< iterator type
	typedef contiguous_pointer_iterator<const T> const_iterator; //!< const iterator type
	typedef pointer_reverse_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef pointer_reverse_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

public:
	HOST DEVICE contiguous_temporary_array() : temporary_array<T,T*>() {}
	template<typename T2>
	HOST DEVICE contiguous_temporary_array( const contiguous_temporary_array<T2>& src ) : temporary_array<T2,T2*>( src ) {}
	HOST DEVICE contiguous_temporary_array( pointer ptr, size_type length ) : temporary_array<T,T*>( ptr, length ) {}
	HOST DEVICE ~contiguous_temporary_array() {}

	template<class InputIterator>
	HOST void assign_from_host( InputIterator begin, InputIterator end ) {
		std::vector<value_type> v( begin, end );
		CUDA_CALL( cudaMemcpy<value_type>( reinterpret_cast<value_type*>(temporary_array<T,T*>::data()), &v.front(), std::min(v.size(),temporary_array<T,T*>::size()), cudaMemcpyHostToDevice ) );
	}

	template<class ContiguousIterator>
	HOST void assign_from_device( ContiguousIterator begin, ContiguousIterator end ) {
		const std::ptrdiff_t n = end-begin;
		if( n < 0 ) throw std::length_error( "ecuda::contiguous_temporary_array::assign() given iterator-based range oriented in wrong direction (are begin and end mixed up?)" );
		CUDA_CALL( cudaMemcpy<value_type>( base_type::data(), begin, std::min(static_cast<typename base_type::size_type>(n),base_type::size()), cudaMemcpyDeviceToDevice ) );
	}

};

///
/// \brief A temporary matrix comprised of a pointer, width and height.
///
/// Acts as a standalone representation of a fixed-size matrix of values
/// given a pointer and the desired dimensions. Used to generate submatrices from
/// a larger memory structure (e.g. an individual slice of a larger cube). This
/// is a contrived structure to provide matrix-like operations, no
/// allocation/deallocation is done.
///
template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type,class RowType=temporary_array<T,PointerType> >
class temporary_matrix : public RowType //temporary_array<T,PointerType>
{
private:
	//typedef temporary_array<T,PointerType> base_type;
	typedef RowType base_type;

public:
	typedef typename base_type::value_type value_type; //!< element data type
	typedef typename base_type::pointer pointer; //!< element pointer type
	typedef typename base_type::reference reference; //!< element reference type
	typedef typename base_type::const_reference const_reference; //!< const element reference type
	typedef typename base_type::size_type size_type; //!< unsigned integral type
	typedef typename base_type::difference_type difference_type; //!< signed integral type

	typedef typename base_type::iterator iterator; //!< iterator type
	typedef typename base_type::const_iterator const_iterator; //!< const iterator type
	typedef typename base_type::reverse_iterator reverse_iterator; //!< reverse iterator type
	typedef typename base_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

	typedef temporary_array<value_type,pointer> row_type;
	typedef temporary_array<const value_type,const pointer> const_row_type;
	typedef temporary_array< value_type, striding_ptr<value_type,pointer> > column_type;
	typedef temporary_array< const value_type, striding_ptr<const value_type,const pointer> > const_column_type;

private:
	size_type height;

public:
	HOST DEVICE temporary_matrix() : base_type(), height(0) {} //temporary_array<T>(), height(0) {}
	template<typename U>
	HOST DEVICE temporary_matrix( const temporary_matrix<U>& src ) : base_type(src), height(src.height) {} // temporary_array<T,PointerType>(src), height(src.height) {}
	HOST DEVICE temporary_matrix( pointer ptr, size_type width, size_type height ) : base_type(ptr,width*height), height(height) {} //temporary_array<T,PointerType>(ptr,width*height), height(height) {}
	HOST DEVICE ~temporary_matrix() {}

	// capacity:
	HOST DEVICE inline size_type get_width() const { return base_type::size()/height; }
	HOST DEVICE inline size_type get_height() const { return height; }

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

	HOST DEVICE inline row_type get_row( size_type rowIndex ) { return operator[]( rowIndex ); }
	HOST DEVICE inline const_row_type get_row( size_type rowIndex ) const { return operator[]( rowIndex ); }

	HOST DEVICE inline column_type get_column( size_type columnIndex ) {
		pointer ptr = base_type::data();
		ptr += columnIndex;
		return column_type( striding_ptr<value_type,pointer>( ptr, get_width() ), get_height() );
	}

	HOST DEVICE inline const_column_type get_column( size_type columnIndex ) const {
		pointer ptr = base_type::data();
		ptr += columnIndex;
		return const_column_type( striding_ptr<const value_type,const pointer>( ptr, get_width() ), get_height() );
	}

	HOST DEVICE temporary_matrix& operator=( const temporary_matrix& other ) {
		base_type::operator=( other );
		height = other.height;
		return *this;
	}

};

template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type>
class contiguous_temporary_matrix : private temporary_matrix< T, PointerType, contiguous_temporary_array<T> >
{
private:
	typedef temporary_matrix< T, PointerType, contiguous_temporary_array<T> > base_type;

public:
	typedef typename base_type::value_type value_type; //!< element data type
	typedef typename base_type::pointer pointer; //!< element pointer type
	typedef typename base_type::reference reference; //!< element reference type
	typedef typename base_type::const_reference const_reference; //!< const element reference type
	typedef typename base_type::size_type size_type; //!< unsigned integral type
	typedef typename base_type::difference_type difference_type; //!< signed integral type

	typedef typename base_type::iterator iterator; //!< iterator type
	typedef typename base_type::const_iterator const_iterator; //!< const iterator type
	typedef typename base_type::reverse_iterator reverse_iterator; //!< reverse iterator type
	typedef typename base_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

	typedef contiguous_temporary_array<value_type> row_type;
	typedef contiguous_temporary_array<const value_type> const_row_type;
	typedef typename base_type::column_type column_type;
	typedef typename base_type::const_column_type const_column_type;

public:
	HOST DEVICE contiguous_temporary_matrix() : base_type() {}
	template<typename U>
	HOST DEVICE contiguous_temporary_matrix( const contiguous_temporary_matrix<U>& src ) : base_type(src) {}
	HOST DEVICE contiguous_temporary_matrix( pointer ptr, size_type width, size_type height ) : base_type(ptr,width,height) {}
	HOST DEVICE ~contiguous_temporary_matrix() {}

	HOST DEVICE inline size_type get_width() const { return base_type::get_width(); }
	HOST DEVICE inline size_type get_height() const { return base_type::get_height(); }

	HOST DEVICE inline row_type operator[]( size_type index ) {
		pointer p = base_type::data();
		p += index*base_type::get_height();
		typename row_type::pointer np = p;
		return row_type( np, base_type::get_height() );
	}

	HOST DEVICE inline const_row_type operator[]( size_type index ) const {
		pointer p = base_type::data();
		p += index*base_type::get_height();
		typename const_row_type::pointer np = p;
		return const_row_type( np, base_type::get_height() );
	}

	HOST DEVICE inline row_type get_row( size_type rowIndex ) { return operator[]( rowIndex ); }
	HOST DEVICE inline const_row_type get_row( size_type rowIndex ) const { return operator[]( rowIndex ); }

	HOST DEVICE inline column_type get_column( size_type columnIndex ) { return base_type::get_column(); }
	HOST DEVICE inline const_column_type get_column( size_type columnIndex ) const { return base_type::get_column(); }

	HOST DEVICE contiguous_temporary_matrix& operator=( const contiguous_temporary_matrix& other ) {
		base_type::operator=( other );
		return *this;
	}

};

} // namespace ecuda

#endif
