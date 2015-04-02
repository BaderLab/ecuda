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
// views.hpp
//
// Provides specialized view of the data within a container.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_VIEWS_HPP
#define ECUDA_VIEWS_HPP

#include <cstddef>
#include <vector>

#include "global.hpp"
#include "apiwrappers.hpp"
#include "iterators.hpp"
#include "device_ptr.hpp"
#include "striding_ptr.hpp"
#include "padded_ptr.hpp"

namespace ecuda {

struct contiguous_sequence_tag {};
struct noncontiguous_sequence_tag {};

template<typename T,typename PointerType=typename reference<T>::pointer_type,typename Category=contiguous_sequence_tag>
class __device_sequence
{
public:
	typedef T value_type;
	typedef PointerType pointer;
	typedef Category category;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

	typedef contiguous_device_iterator<value_type> iterator;
	typedef contiguous_device_iterator<const value_type> const_iterator;
	typedef reverse_device_iterator<iterator> reverse_iterator;
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator;

private:
	pointer ptr;
	size_type length;

private:
	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, contiguous_sequence_tag, std::random_access_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		// will never be called
		#else
		const typename std::iterator_traits<Iterator>::difference_type n = std::distance(first,last);
		if( n < 0 or static_cast<size_type>(n) != length ) throw std::length_error( "__device_sequence::assign first,last does not span the correct number of elements" );
		CUDA_CALL( cudaMemcpy<value_type>( ptr, first.operator->(), length, cudaMemcpyHostToDevice ) );
		#endif
	}

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, contiguous_sequence_tag, std::bidirectional_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		// will never be called
		#else
		std::vector< value_type, host_allocator<value_type> > v( first, last );
		if( v.size() != length ) throw std::length_error( "__device_sequence::assign first,last does not span the correct number of elements" );
		CUDA_CALL( cudaMemcpy<value_type>( ptr, &v.front(), length, cudaMemcpyHostToDevice ) );
		#endif
	}

	template<class Iterator> HOST DEVICE inline void assign( Iterator first, Iterator last, contiguous_sequence_tag, std::forward_iterator_tag ) { assign( first, last, std::bidirectional_iterator_tag() ); }
	template<class Iterator> HOST DEVICE inline void assign( Iterator first, Iterator last, contiguous_sequence_tag, std::input_iterator_tag ) { assign( first, last, std::bidirectional_iterator_tag() ); }

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, contiguous_sequence_tag, contiguous_device_iterator_tag ) {
		const typename std::iterator_traits<Iterator>::difference_type n = last-first;
		#ifdef __CUDA_ARCH__
		if( n < 0 or static_cast<size_type>(n) != length ) return; // nothing happens
		for( pointer p = ptr; first != last; ++p, ++first ) *p = *first;
		#else
		if( n < 0 or static_cast<size_type>(n) != length ) throw std::length_error( "__device_sequence::assign first,last does not span the correct number of elements" );
		CUDA_CALL( cudaMemcpy<value_type>( ptr, first.operator->(), length, cudaMemcpyDeviceToDevice ) );
		#endif

	}

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, noncontiguous_sequence_tag, contiguous_device_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		const typename std::iterator_traits<Iterator>::difference_type n = last-first;
		if( n < 0 or static_cast<size_type>(n) != length ) return; // nothing happens
		for( iterator dest = begin(); dest != end(); ++dest, ++first ) *dest = *first;
		#else
		throw cuda_error( cudaErrorInvalidDevicePointer, "__device_sequence::assign cannot assign range to noncontiguous memory" );
		#endif
	}

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, contiguous_sequence_tag, device_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		for( iterator dest = begin(); dest != end() and first != last; ++dest, ++first ) *dest = *first;
		#else
		throw cuda_error( cudaErrorInvalidDevicePointer, "__device_sequence::assign cannot assign range from noncontiguous memory" );
		#endif
	}

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, noncontiguous_sequence_tag, device_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		assign( first, last, contiguous_sequence_tag(), device_iterator_tag() );
		#else
		throw cuda_error( cudaErrorInvalidDevicePointer, "__device_sequence::assign cannot assign range to and from noncontiguous memory" );
		#endif
	}

public:
	HOST DEVICE __device_sequence( pointer ptr, size_type length ) : ptr(ptr), length(length) {}
	HOST DEVICE __device_sequence( const __device_sequence<T>& src ) : ptr(src.ptr), length(src.length) {}

	HOST DEVICE inline pointer data() const __NOEXCEPT__ { return ptr; }
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return length; }

	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(ptr); }
	HOST DEVICE inline iterator end() __NOEXCEPT__  { return iterator(ptr+length); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(ptr); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(ptr+length); }

	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(end()); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(begin()); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(end()); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(begin()); }

	DEVICE inline reference operator[]( const size_type index ) { return *(ptr+index); }
	DEVICE inline const_reference operator[]( const size_type index ) const { return *(ptr+index); }

	template<class Iterator>
	HOST DEVICE inline void assign( Iterator first, Iterator last ) { assign( first, last, category(), typename std::iterator_traits<Iterator>::iterator_category() ); }

};

template<typename T>
class __device_grid : private __device_sequence<typename cast_to_char<T>::type>
{
private:
	typedef __device_sequence<typename cast_to_char<T>::type> base_type;

public:
	typedef T value_type;
	typedef value_type* pointer;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

	typedef device_iterator< value_type, padded_ptr<value_type,pointer,1> > iterator;
	typedef device_iterator< const value_type, padded_ptr<const value_type,const pointer,1> > const_iterator;
	typedef reverse_device_iterator<iterator> reverse_iterator;
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator;

	typedef __device_sequence<value_type> row_type;
	typedef const __device_sequence<const value_type> const_row_type;
	typedef __device_sequence<value_type,padded_ptr<value_type,striding_ptr<value_type>,1>,noncontiguous_sequence_tag> column_type;
	typedef const __device_sequence<const value_type,padded_ptr<const value_type,striding_ptr<const value_type>,1>,noncontiguous_sequence_tag> const_column_type;

private:
	size_type numberRows;
	size_type numberColumns;

private:
	HOST DEVICE inline size_type get_pitch() const __NOEXCEPT__ { return base_type::size()/numberRows; }

public:
	HOST DEVICE __device_grid( pointer ptr, size_type numberRows, size_type numberColumns, size_type pitch ) :
		base_type( reinterpret_cast<typename base_type::pointer>(ptr), pitch*numberRows ), numberRows(numberRows), numberColumns(numberColumns) {}
	HOST DEVICE __device_grid( const __device_grid<T>& src ) : base_type(src), numberRows(src.numberRows), numberColumns(src.numberColumns) {}

	HOST DEVICE inline pointer data() const __NOEXCEPT__ { return reinterpret_cast<pointer>(base_type::data()); }
	HOST DEVICE inline size_type number_rows() const __NOEXCEPT__ { return numberRows; }
	HOST DEVICE inline size_type number_columns() const __NOEXCEPT__ { return numberColumns; }
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return numberRows*numberColumns; }

	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator( padded_ptr<value_type,pointer,1>( data(), number_columns(), get_pitch()-sizeof(value_type)*number_columns() ) ); }
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator( padded_ptr<value_type,pointer,1>( reinterpret_cast<value_type>(base_type::end().operator->()), number_columns(), get_pitch()-sizeof(value_type)*number_columns() ) ); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator( padded_ptr<const value_type,const pointer,1>( data(), number_columns(), get_pitch()-sizeof(value_type)*number_columns() ) ); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator( padded_ptr<const value_type,const pointer,1>( reinterpret_cast<value_type>(base_type::end().operator->()), number_columns(), get_pitch()-sizeof(value_type)*number_columns() ) ); }

	HOST DEVICE inline row_type get_row( const size_type rowIndex ) { return row_type( reinterpret_cast<pointer>( base_type::data()+(get_pitch()*rowIndex) ), number_columns() ); }
	HOST DEVICE inline const_row_type get_row( const size_type rowIndex ) const { return const_row_type( reinterpret_cast<pointer>( base_type::data()+(get_pitch()*rowIndex) ), number_columns() ); }
	HOST DEVICE inline column_type get_column( const size_type columnIndex ) {
		striding_ptr<value_type> sp( data(), number_columns() );
		padded_ptr<value_type,striding_ptr<value_type>,1> pp( sp, 1, get_pitch()-sizeof(value_type)*number_columns() );
		return column_type( pp, number_rows() );
	}
	HOST DEVICE inline const_column_type get_column( const size_type columnIndex ) const {
		striding_ptr<const value_type> sp( data(), number_columns() );
		padded_ptr<const value_type,striding_ptr<const value_type>,1> pp( sp, 1, get_pitch()-sizeof(value_type)*number_columns() );
		return const_column_type( pp, number_rows() );
	}

	DEVICE inline row_type operator[]( const size_type index ) { return get_row(index); }
	DEVICE inline const_row_type operator[]( const size_type index ) const { return get_row(index); }

	template<Iterator>
	HOST DEVICE inline assign( Iterator first, Iterator last ) { assign( first, last, typename std::iterator_traits<Iterator>::iterator_category() ); }

};


///
/// \brief View of data sequence given a pointer and size.
///
/// Acts as a standalone representation of a linear fixed-size series of values
/// given a pointer and the desired size. Used to generate subsequences from
/// a larger memory structure (e.g. an individual row of a larger matrix). This
/// is a contrived structure to provide array-like operations, no
/// allocation/deallocation is done.
///
template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type>
class sequence_view
{
public:
	typedef T value_type; //!< element data type
	typedef PointerType pointer; //!< element pointer type
	typedef value_type& reference; //!< element reference type
	typedef const value_type& const_reference; //!< const element reference type
	typedef std::size_t size_type; //!< unsigned integral type
	typedef std::ptrdiff_t difference_type; //!< signed integral type

	typedef device_iterator<value_type,pointer> iterator; //!< iterator type
	typedef device_iterator<const value_type,/*const*/ pointer> const_iterator; //!< const iterator type
	typedef reverse_device_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

protected:
	pointer ptr; //!< pointer to the start of the array
	size_type length; //!< number of elements in the array

public:
	HOST DEVICE sequence_view() : ptr(nullptr), length(0) {}
	template<typename T2,typename PointerType2>
	HOST DEVICE sequence_view( const sequence_view<T2,PointerType2>& src ) : ptr(src.data()), length(src.size()) {}
	HOST DEVICE sequence_view( pointer ptr, size_type length ) : ptr(ptr), length(length) {}
	HOST DEVICE ~sequence_view() {}

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
	//DEVICE inline reference at( size_type index ) { return operator[]( index ); }
	DEVICE inline reference front() { return operator[](0); }
	DEVICE inline reference back() { return operator[](size()-1); }
	DEVICE inline const_reference operator[]( size_type index ) const {	return *(ptr+static_cast<int>(index)); }
	//DEVICE inline const_reference at( size_type index ) const { return operator[]( index ); }
	DEVICE inline const_reference front() const { return operator[](0); }
	DEVICE inline const_reference back() const { return operator[](size()-1); }

	DEVICE void fill( const value_type& value ) {
		iterator iter = begin();
		while( iter != end() ) { *iter = value; ++iter; }
	}

	HOST DEVICE sequence_view& operator=( const sequence_view& other ) {
		ptr = other.ptr;
		length = other.length;
		return *this;
	}

};


///
/// \brief View of data sequence residing in contiguous memory given a pointer and size.
///
/// This is a subclass of sequence_view that imposes the requirement that the
/// underlying pointer refers to a contiguous block of memory.  Thus, PointerType
/// (the second template parameter of sequence_view) is strictly defined as a
/// naked pointer of type T*.
///
/// This allows the assign method to be made available safely.
///
template<typename T>
class contiguous_sequence_view : public sequence_view<T,T*>
{
private:
	typedef sequence_view<T,T*> base_type;

public:
	typedef typename sequence_view<T,T*>::value_type value_type; //!< element data type
	typedef typename sequence_view<T,T*>::pointer pointer; //!< element pointer type
	typedef typename sequence_view<T,T*>::reference reference; //!< element reference type
	typedef typename sequence_view<T,T*>::const_reference const_reference; //!< const element reference type
	typedef typename sequence_view<T,T*>::size_type size_type; //!< unsigned integral type
	typedef typename sequence_view<T,T*>::difference_type difference_type; //!< signed integral type

	typedef contiguous_device_iterator<T> iterator; //!< iterator type
	typedef contiguous_device_iterator<const T> const_iterator; //!< const iterator type
	typedef reverse_device_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

public:
	HOST DEVICE contiguous_sequence_view() : sequence_view<T,T*>() {}
	template<typename T2>
	HOST DEVICE contiguous_sequence_view( const contiguous_sequence_view<T2>& src ) : sequence_view<T,T*>( src ) {}
	HOST DEVICE contiguous_sequence_view( pointer ptr, size_type length ) : sequence_view<T,T*>( ptr, length ) {}
	HOST DEVICE ~contiguous_sequence_view() {}

	// iterators:
	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(base_type::data()); }
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator(base_type::data()+static_cast<int>(base_type::size())); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(base_type::data()); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(base_type::data()+static_cast<int>(base_type::size())); }
	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(iterator(base_type::data()+static_cast<int>(base_type::size()))); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(iterator(base_type::data())); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(base_type::data()+static_cast<int>(base_type::size()))); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(base_type::data())); }

private:
	template<class Iterator>
	HOST void assign( Iterator first, Iterator last, std::random_access_iterator_tag ) {
		const typename std::iterator_traits<Iterator>::difference_type n = std::distance(first,last);
		if( n < 0 or static_cast<size_type>(n) != base_type::size() ) throw std::length_error( "ecuda::vector::sequence_view(first,last) the number of elements to assign does not match the length of this sequence" );
		CUDA_CALL( cudaMemcpy<value_type>( base_type::data(), first.operator->(), base_type::size(), cudaMemcpyHostToDevice ) );
	}

	template<class Iterator> HOST inline void assign( Iterator first, Iterator last, std::bidirectional_iterator_tag ) {
		std::vector< value_type, host_allocator<value_type> > v( first, last );
		if( v.size() != base_type::size() ) throw std::length_error( "ecuda::vector::sequence_view(first,last) the number of elements to assign does not match the length of this sequence" );
		CUDA_CALL( cudaMemcpy<value_type>( base_type::data(), &v.front(), base_type::size(), cudaMemcpyHostToDevice ) );
	}

	template<class Iterator>
	HOST void assign( Iterator first, Iterator last, contiguous_device_iterator_tag ) {
		const typename std::iterator_traits<Iterator>::difference_type n = last-first;
		if( n < 0 or static_cast<size_type>(n) != base_type::size() )  throw std::length_error( "ecuda::vector::sequence_view(first,last) the number of elements to assign does not match the length of this sequence" );
		CUDA_CALL( cudaMemcpy<value_type>( base_type::data(), first.operator->(), base_type::size(), cudaMemcpyDeviceToDevice ) );
	}

public:

	template<class Iterator>
	HOST inline void assign( Iterator first, Iterator last ) { assign( first, last, typename std::iterator_traits<Iterator>::iterator_category() ); }

	HOST DEVICE void fill( const value_type& value ) {
		#ifdef __CUDA_ARCH__
		for( iterator iter = begin(); iter != end(); ++iter ) *iter = value;
		#else
		CUDA_CALL( cudaMemset<value_type>( base_type::data(), value, base_type::size() ) );
		#endif
	}

	HOST DEVICE contiguous_sequence_view& operator=( const contiguous_sequence_view& other ) {
		base_type::operator=( other );
		return *this;
	}


};

///
/// \brief View of data matrix residing given a pointer and dimensions.
///
/// Acts as a standalone representation of a fixed-size matrix of values
/// given a pointer and the desired dimensions. Used to generate submatrices from
/// a larger memory structure (e.g. an individual slice of a larger cube). This
/// is a contrived structure to provide matrix-like operations, no
/// allocation/deallocation is done.
///
template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type,class RowType=sequence_view<T,PointerType> >
class matrix_view : public RowType //sequence_view<T,PointerType>
{
private:
	//typedef sequence_view<T,PointerType> base_type;
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

	typedef sequence_view<value_type,pointer> row_type;
	typedef sequence_view<const value_type,/*const*/ pointer> const_row_type;
	typedef sequence_view< value_type, striding_ptr<value_type,pointer> > column_type;
	typedef sequence_view< const value_type, striding_ptr<const value_type,/*const*/ pointer> > const_column_type;

private:
	size_type height;

public:
	HOST DEVICE matrix_view() : base_type(), height(0) {} //sequence_view<T>(), height(0) {}
	template<typename U>
	HOST DEVICE matrix_view( const matrix_view<U>& src ) : base_type(src), height(src.height) {} // sequence_view<T,PointerType>(src), height(src.height) {}
	HOST DEVICE matrix_view( pointer ptr, size_type width, size_type height ) : base_type(ptr,width*height), height(height) {} //sequence_view<T,PointerType>(ptr,width*height), height(height) {}
	HOST DEVICE ~matrix_view() {}

	// capacity:
	HOST DEVICE inline size_type get_width() const { return base_type::size()/height; }
	HOST DEVICE inline size_type get_height() const { return height; }

	// iterators:
	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(base_type::data()); }
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator(base_type::data()+static_cast<int>(base_type::size())); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(base_type::data()); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(base_type::data()+static_cast<int>(base_type::size())); }
	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(iterator(base_type::data()+static_cast<int>(base_type::size()))); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(iterator(base_type::data())); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(base_type::data()+static_cast<int>(base_type::size()))); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(base_type::data())); }

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

	DEVICE void fill( const value_type& value ) {
		iterator iter = begin();
		while( iter != end() ) { *iter = value; ++iter; }
	}


	HOST DEVICE matrix_view& operator=( const matrix_view& other ) {
		base_type::operator=( other );
		height = other.height;
		return *this;
	}

};

///
/// \brief View of data matrix residing in contiguous memory given a pointer and dimensions.
///
template<typename T>
class contiguous_matrix_view : private matrix_view< T, padded_ptr<T,T*,1>, contiguous_sequence_view<T> >
{
private:
	typedef matrix_view< T, padded_ptr<T,T*,1>, contiguous_sequence_view<T> > base_type;

public:
	typedef typename base_type::value_type value_type; //!< element data type
	typedef T* pointer; //!< element pointer type
	//typedef typename base_type::pointer pointer; //!< element pointer type
	typedef typename base_type::reference reference; //!< element reference type
	typedef typename base_type::const_reference const_reference; //!< const element reference type
	typedef typename base_type::size_type size_type; //!< unsigned integral type
	typedef typename base_type::difference_type difference_type; //!< signed integral type

	typedef typename base_type::iterator iterator; //!< iterator type
	typedef typename base_type::const_iterator const_iterator; //!< const iterator type
	typedef typename base_type::reverse_iterator reverse_iterator; //!< reverse iterator type
	typedef typename base_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

	typedef contiguous_sequence_view<value_type> row_type;
	typedef contiguous_sequence_view<const value_type> const_row_type;
	typedef typename base_type::column_type column_type;
	typedef typename base_type::const_column_type const_column_type;

	typedef contiguous_device_iterator<const T> ContiguousDeviceIterator;

public:
	HOST DEVICE contiguous_matrix_view() : base_type() {}
	template<typename U>
	HOST DEVICE contiguous_matrix_view( const contiguous_matrix_view<U>& src ) : base_type(src) {}
	HOST DEVICE contiguous_matrix_view( pointer ptr, size_type width, size_type height, size_type paddingBytes=0 ) :
		base_type( padded_ptr<T,T*,1>( ptr, width, paddingBytes ), width, height ) {}
	HOST DEVICE ~contiguous_matrix_view() {}

	HOST DEVICE inline size_type get_width() const { return base_type::get_width(); }
	HOST DEVICE inline size_type get_height() const { return base_type::get_height(); }
	HOST DEVICE inline size_type get_pitch() const {
		padded_ptr<T,T*,1> ptr = base_type::data();
		const typename base_type::size_type pitch = ptr.get_data_length()*sizeof(value_type) + ptr.get_padding_length()*ptr.get_pad_length_units();
		return pitch;
	}

	// iterators:
	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(base_type::data()); }
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator(base_type::data()+static_cast<int>(base_type::size())); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(base_type::data()); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(base_type::data()+static_cast<int>(base_type::size())); }
	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(iterator(base_type::data()+static_cast<int>(base_type::size()))); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(iterator(base_type::data())); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(base_type::data()+static_cast<int>(base_type::size()))); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(base_type::data())); }

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

private:
	template<class Iterator>
	HOST void assign( Iterator first, Iterator last, std::random_access_iterator_tag ) {
		const typename std::iterator_traits<Iterator>::difference_type n = std::distance(first,last);
		if( n < 0 or static_cast<size_type>(n) != (get_width()*get_height()) ) throw std::length_error( "ecuda::contiguous_matrix_view::assign() given iterator-based range that does not have width x height elements" );
		CUDA_CALL( cudaMemcpy2D<value_type>( base_type::data(), get_pitch(), first.operator->(), get_width()*sizeof(value_type), get_width(), get_height(), cudaMemcpyHostToDevice ) );
	}

	template<class Iterator>
	HOST void assign( Iterator first, Iterator last, std::bidirectional_iterator_tag ) {
		std::vector< value_type, host_allocator<value_type> > v( first, last );
		if( v.size() != (get_width()*get_height()) ) throw std::length_error( "ecuda::contiguous_matrix_view::assign() given iterator-based range that does not have width x height elements" );
		CUDA_CALL( cudaMemcpy2D<value_type>( base_type::data(), get_pitch(), &v.front(), get_width()*sizeof(value_type), get_width(), get_height(), cudaMemcpyHostToDevice ) );
	}

	template<class Iterator>
	HOST void assign( Iterator first, Iterator last, contiguous_device_iterator_tag ) {
		const typename std::iterator_traits<Iterator>::difference_type n = last-first;
		if( n < 0 or static_cast<size_type>(n) != (get_width()*get_height()) ) throw std::length_error( "ecuda::contiguous_matrix_view::assign() given iterator-based range that does not have width x height elements" );
		CUDA_CALL( cudaMemcpy2D<value_type>( base_type::data(), get_pitch(), first.operator->(), get_width()*sizeof(value_type), get_width(), get_height(), cudaMemcpyDeviceToDevice ) );
	}

public:
	template<class Iterator>
	HOST inline void assign( Iterator first, Iterator last ) { assign( first, last, typename std::iterator_traits<Iterator>::iterator_category() ); }

	HOST DEVICE void fill( const value_type& value ) {
		#ifdef __CUDA_ARCH__
		iterator iter = begin();
		while( iter != end() ) { *begin = value; ++iter; }
		#else
		CUDA_CALL( cudaMemset2D<value_type>( base_type::data(), get_pitch(), value, get_width(), get_height() ) );
		#endif
	}

	HOST DEVICE contiguous_matrix_view& operator=( const contiguous_matrix_view& other ) {
		base_type::operator=( other );
		return *this;
	}

};

} // namespace ecuda

#endif
