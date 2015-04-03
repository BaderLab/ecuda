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

struct contiguous_memory_tag {};
struct noncontiguous_memory_tag {};

template<typename T,typename PointerType,typename Category> struct __device_sequence_iterator_traits;
template<typename T,typename PointerType> struct __device_sequence_iterator_traits<T,PointerType,contiguous_memory_tag> {
	typedef contiguous_device_iterator<T> iterator;
	typedef contiguous_device_iterator<const T> const_iterator;
};
template<typename T,typename PointerType> struct __device_sequence_iterator_traits<T,PointerType,noncontiguous_memory_tag> {
	typedef device_iterator<T,PointerType> iterator;
	typedef device_iterator<const T,PointerType> const_iterator;
};

template<typename T,typename PointerType,typename CategoryRow,typename CategoryColumn> struct __device_grid_iterator_traits {
	typedef device_iterator<T,PointerType> iterator;
	typedef device_iterator<const T,PointerType> const_iterator;
};
template<typename T,typename PointerType> struct __device_grid_iterator_traits<T,PointerType,contiguous_memory_tag,contiguous_memory_tag> {
	typedef contiguous_device_iterator<T> iterator;
	typedef contiguous_device_iterator<const T> const_iterator;
};

struct root_container_tag {};
struct child_container_tag {};

template<typename T,typename PointerType,typename Category,typename ContainerType> struct __device_sequence_traits;
template<typename T,typename PointerType,typename Category> struct __device_sequence_traits<T,PointerType,Category,root_container_tag> {
	typedef typename PointerType::pointer pointer;
	typedef typename __device_sequence_iterator_traits<T,PointerType,Category>::iterator iterator;
	typedef typename __device_sequence_iterator_traits<T,PointerType,Category>::const_iterator const_iterator;
};
template<typename T,typename PointerType,typename Category> struct __device_sequence_traits<T,PointerType,Category,child_container_tag> {
	typedef PointerType pointer;
	typedef typename __device_sequence_iterator_traits<T,PointerType,Category>::iterator iterator;
	typedef typename __device_sequence_iterator_traits<T,PointerType,Category>::const_iterator const_iterator;
};

template<typename T,typename PointerType,typename CategoryRow,typename CategoryColumn,typename ContainerType> struct __device_grid_traits;
template<typename T,typename PointerType,typename CategoryRow,typename CategoryColumn> struct __device_grid_traits<T,PointerType,CategoryRow,CategoryColumn,root_container_tag> {
	typedef typename PointerType::pointer pointer;
	typedef typename __device_grid_iterator_traits<T,PointerType,CategoryRow,CategoryColumn>::iterator iterator;
	typedef typename __device_grid_iterator_traits<T,PointerType,CategoryRow,CategoryColumn>::const_iterator const_iterator;
};

template<typename T,typename PointerType,typename CategoryRow,typename CategoryColumn> struct __device_grid_traits<T,PointerType,CategoryRow,CategoryColumn,child_container_tag> {
	typedef PointerType pointer;
	typedef typename __device_grid_iterator_traits<T,PointerType,CategoryRow,CategoryColumn>::iterator iterator;
	typedef typename __device_grid_iterator_traits<T,PointerType,CategoryRow,CategoryColumn>::const_iterator const_iterator;
};

template<typename T,typename PointerType=typename reference<T>::pointer_type,typename Category=contiguous_memory_tag,typename ContainerType=root_container_tag>
class __device_sequence
{
public:
	typedef T value_type;
	typedef typename __device_sequence_traits<T,PointerType,Category,ContainerType>::pointer pointer;
	typedef Category category;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

private:
	typedef PointerType managed_pointer;

public:
	typedef typename __device_sequence_traits<T,PointerType,Category,ContainerType>::iterator iterator;
	typedef typename __device_sequence_traits<T,PointerType,Category,ContainerType>::const_iterator const_iterator;
	typedef reverse_device_iterator<iterator> reverse_iterator;
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator;

private:
	managed_pointer ptr;
	size_type length;

private:
	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, contiguous_memory_tag, std::random_access_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		// will never be called
		#else
		const typename std::iterator_traits<Iterator>::difference_type n = std::distance(first,last);
		if( n < 0 or static_cast<size_type>(n) != length ) throw std::length_error( EXCEPTION_MSG("__device_sequence::assign first,last does not span the correct number of elements") );
		typename std::remove_const<value_type>::type* p = ptr;
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( p, first.operator->(), length, cudaMemcpyHostToDevice ) );
		CUDA_CHECK_ERRORS();
		#endif
	}

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, contiguous_memory_tag, std::bidirectional_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		// will never be called
		#else
		std::vector< value_type, host_allocator<value_type> > v( first, last );
		if( v.size() != length ) throw std::length_error( EXCEPTION_MSG("__device_sequence::assign first,last does not span the correct number of elements") );
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( ptr, &v.front(), length, cudaMemcpyHostToDevice ) );
		#endif
	}

	template<class Iterator> HOST DEVICE inline void assign( Iterator first, Iterator last, contiguous_memory_tag, std::forward_iterator_tag ) { assign( first, last, contiguous_memory_tag(), std::bidirectional_iterator_tag() ); }
	template<class Iterator> HOST DEVICE inline void assign( Iterator first, Iterator last, contiguous_memory_tag, std::input_iterator_tag ) { assign( first, last, contiguous_memory_tag(), std::bidirectional_iterator_tag() ); }

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, contiguous_memory_tag, contiguous_device_iterator_tag ) {
		const typename std::iterator_traits<Iterator>::difference_type n = last-first;
		#ifdef __CUDA_ARCH__
		if( n < 0 or static_cast<size_type>(n) != length ) return; // nothing happens
		for( pointer p = ptr; first != last; ++p, ++first ) *p = *first;
		#else
		if( n < 0 or static_cast<size_type>(n) != length ) throw std::length_error( EXCEPTION_MSG("__device_sequence::assign first,last does not span the correct number of elements") );
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( ptr, first.operator->(), length, cudaMemcpyDeviceToDevice ) );
		#endif

	}

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, noncontiguous_memory_tag, contiguous_device_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		const typename std::iterator_traits<Iterator>::difference_type n = last-first;
		if( n < 0 or static_cast<size_type>(n) != length ) return; // nothing happens
		for( iterator dest = begin(); dest != end(); ++dest, ++first ) *dest = *first;
		#else
		throw cuda_error( cudaErrorInvalidDevicePointer, EXCEPTION_MSG("__device_sequence::assign cannot assign range to noncontiguous memory") );
		#endif
	}

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, contiguous_memory_tag, device_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		for( iterator dest = begin(); dest != end() and first != last; ++dest, ++first ) *dest = *first;
		#else
		throw cuda_error( cudaErrorInvalidDevicePointer, EXCEPTION_MSG("__device_sequence::assign cannot assign range from noncontiguous memory") );
		#endif
	}

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, noncontiguous_memory_tag, device_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		assign( first, last, contiguous_sequence_tag(), device_iterator_tag() );
		#else
		throw cuda_error( cudaErrorInvalidDevicePointer, EXCEPTION_MSG("__device_sequence::assign cannot assign range to and from noncontiguous memory") );
		#endif
	}

	HOST DEVICE void fill( const value_type& value, contiguous_memory_tag ) {
		#ifdef __CUDA_ARCH__
		for( iterator iter = begin(); iter != end(); ++iter ) *iter = value;
		#else
		CUDA_CALL( cudaMemset<value_type>( data(), value, size() ) );
		#endif
	}

	HOST DEVICE void fill( const value_type& value, noncontiguous_memory_tag ) {
		#ifdef __CUDA_ARCH__
		for( iterator iter = begin(); iter != end(); ++iter ) *iter = value;
		#else
		throw cuda_error( cudaErrorInvalidDevicePointer, EXCEPTION_MSG("__device_sequence::fill cannot fill non-contiguous device memory from host") );
		#endif
	}

	template<class Container>
	HOST void copy_to( Container& container, contiguous_memory_tag, std::random_access_iterator_tag ) const {
		const typename std::iterator_traits<typename Container::iterator>::difference_type n = std::distance( container.begin(), container.end() );
		if( n < 0 or static_cast<size_type>(n) < size() ) throw std::length_error( EXCEPTION_MSG("__device_sequence::operator>> target container does not have sufficient space") );
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( container.begin().operator->(), data(), size(), cudaMemcpyDeviceToHost ) );
	}

	template<class Container>
	HOST void copy_to( Container& container, contiguous_memory_tag, std::bidirectional_iterator_tag ) const {
		if( container.size() != size() ) throw std::length_error( EXCEPTION_MSG("__device_sequence::operator>> target container does not have sufficient space") );
		std::vector< value_type, host_allocator<value_type> > v( size() );
		operator>>( v );
		typename Container::iterator dest = container.begin();
		for( typename std::vector< value_type, host_allocator<value_type> >::const_iterator src = v.begin(); src != v.end(); ++src, ++dest ) *dest = *src;
	}

	template<class Container> HOST inline void copy_to( Container& container, contiguous_memory_tag, std::forward_iterator_tag ) const { copy_to( container, contiguous_memory_tag(), std::bidirectional_iterator_tag() ); }
	template<class Container> HOST inline void copy_to( Container& container, contiguous_memory_tag, std::output_iterator_tag ) const { copy_to( container, contiguous_memory_tag(), std::bidirectional_iterator_tag() ); }

	template<class Container>
	HOST void copy_to( Container& container, contiguous_memory_tag, contiguous_device_iterator_tag ) const {
		const typename std::iterator_traits<typename Container::iterator>::difference_type n = container.end()-container.begin();
		if( n < 0 or static_cast<size_type>(n) < size() ) throw std::length_error( EXCEPTION_MSG("__device_sequence::operator>> target container does not have sufficient space") );
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( container.begin().operator->(), data(), size(), cudaMemcpyDeviceToDevice ) );
	}

public:
	HOST DEVICE explicit __device_sequence( managed_pointer ptr, size_type length ) : ptr(ptr), length(length) {}
	HOST DEVICE __device_sequence( const __device_sequence<T,PointerType,Category>& src ) : ptr(src.ptr), length(src.length) {}

	HOST DEVICE inline pointer data() const __NOEXCEPT__ { return ptr; }
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return length; }

	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(ptr); }
	HOST DEVICE inline iterator end() __NOEXCEPT__  {
		// NOTE: Important to pre-cast this __device_sequence's pointer to the
		//       pointer type of the iterator BEFORE applying the addition of
		//       the length value.  For example, if this is a padded_ptr
		//       and the iterator is a regular pointer (because we accept
		//       responsibility for ensuring the the range does not cross the
		//       padding region), then failure to pre-cast to the iterator pointer
		//       type will cause the pointer operator+() to set the location
		//       at the start of the next row _after_ the padding.  This will
		//       screw up iter.operator-(otheriter).
		return iterator( static_cast<typename iterator::pointer>(ptr)+static_cast<int>(length) );
	}
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(ptr); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ {
		// NOTE: Important to pre-cast this __device_sequence's pointer to the
		//       pointer type of the iterator BEFORE applying the addition of
		//       the length value.  For example, if this is a padded_ptr
		//       and the iterator is a regular pointer (because we accept
		//       responsibility for ensuring the the range does not cross the
		//       padding region), then failure to pre-cast to the iterator pointer
		//       type will cause the pointer operator+() to set the location
		//       at the start of the next row _after_ the padding.  This will
		//       screw up iter.operator-(otheriter).
		return const_iterator( static_cast<typename const_iterator::pointer>(ptr)+static_cast<int>(length) );
	}

	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(end()); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(begin()); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(end()); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(begin()); }

	DEVICE inline reference operator[]( const size_type index ) { return *(ptr+index); }
	DEVICE inline const_reference operator[]( const size_type index ) const { return *(ptr+index); }

	template<class Iterator>
	HOST DEVICE inline void assign( Iterator first, Iterator last ) { assign( first, last, category(), typename std::iterator_traits<Iterator>::iterator_category() ); }

	HOST DEVICE inline void fill( const value_type& value ) { fill( value, category() ); }

	template<class Container>
	HOST inline const __device_sequence& operator>>( Container& container ) const {
		copy_to( container, category(), typename std::iterator_traits<typename Container::iterator>::iterator_category() );
		return *this;
	}

};

template<typename T,typename PointerType=typename reference<T>::pointer_type,class CategoryRow=noncontiguous_memory_tag,class CategoryColumn=contiguous_memory_tag,class ContainerType=root_container_tag>
class __device_grid : private __device_sequence<T,PointerType,CategoryColumn,ContainerType>
{
private:
	typedef __device_sequence<T,PointerType,CategoryColumn,ContainerType> base_type;

public:
	typedef T value_type;
	typedef typename __device_sequence_traits<T,PointerType,CategoryColumn,ContainerType>::pointer pointer;
	typedef CategoryRow row_category;
	typedef CategoryColumn column_category;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

private:
	typedef PointerType managed_pointer;

public:
	typedef typename __device_grid_traits<T,PointerType,CategoryRow,CategoryColumn,ContainerType>::iterator iterator;
	typedef typename __device_grid_traits<T,PointerType,CategoryRow,CategoryColumn,ContainerType>::const_iterator const_iterator;
	//typedef typename __device_grid_iterator_traits<value_type,pointer,row_category,column_category>::iterator iterator;
	//typedef typename __device_grid_iterator_traits<value_type,pointer,row_category,column_category>::const_iterator const_iterator;
	typedef reverse_device_iterator<iterator> reverse_iterator;
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator;

	typedef __device_sequence<value_type,pointer,column_category,child_container_tag> row_type;
	typedef const __device_sequence<const value_type,pointer,column_category,child_container_tag> const_row_type;
	typedef __device_sequence<value_type,striding_ptr<value_type,pointer>,row_category,child_container_tag> column_type;
	typedef const __device_sequence<const value_type,striding_ptr<value_type,pointer>,row_category,child_container_tag> const_column_type;

private:
	size_type numberRows;

private:
	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, contiguous_memory_tag, std::random_access_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		// will never be called
		#else
		const typename std::iterator_traits<Iterator>::difference_type n = std::distance(first,last);
		if( n < 0 or static_cast<size_type>(n) != size() ) throw std::length_error( EXCEPTION_MSG("__device_grid::assign first,last does not span the correct number of elements") );
		for( size_type i = 0; i < number_rows(); ++i, first += number_columns() ) get_row(i).assign( first, first+number_columns() );
		#endif
	}

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, contiguous_memory_tag, std::bidirectional_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		// will never be called
		#else
		std::vector< value_type, host_allocator<value_type> > v( first, last );
		if( v.size() != size() ) throw std::length_error( EXCEPTION_MSG("__device_grid::assign first,last does not span the correct number of elements") );
		size_type i = 0;
		for( typename std::vector< value_type, host_allocator<value_type> >::const_iterator iter = v.begin(); iter != v.end(); iter += number_columns(), ++i ) get_row(i).assign( iter, iter+number_columns() );
		#endif
	}

	template<class Iterator> HOST DEVICE inline void assign( Iterator first, Iterator last, contiguous_memory_tag, std::forward_iterator_tag ) { assign( first, last, contiguous_memory_tag(), std::bidirectional_iterator_tag() ); }
	template<class Iterator> HOST DEVICE inline void assign( Iterator first, Iterator last, contiguous_memory_tag, std::input_iterator_tag ) { assign( first, last, contiguous_memory_tag(), std::bidirectional_iterator_tag() ); }

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, contiguous_memory_tag, contiguous_device_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		const typename std::iterator_traits<Iterator>::difference_type n = last-first;
		if( n < 0 or static_cast<n> != size() ) return; // nothing happens
		for( size_type i = 0; i < number_rows(); ++i ) get_row(i).assign( first, first+number_columns() );
		#else
		const typename std::iterator_traits<Iterator>::difference_type n = last-first;
		if( n < 0 or static_cast<size_type>(n) != size() ) throw std::length_error( EXCEPTION_MSG("__device_grid::assign first,last does not span the correct number of elements") );
		for( size_type i = 0; i < number_rows(); ++i, first += number_columns() ) get_row(i).assign( first, first+number_columns() );
		#endif
	}

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, noncontiguous_memory_tag, contiguous_device_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		const typename std::iterator_traits<Iterator>::difference_type n = last-first;
		if( n < 0 or static_cast<n> != size() ) return; // nothing happens
		for( size_type i = 0; i < number_rows(); ++i ) get_row(i).assign( first, first+number_columns() );
		#else
		throw cuda_error( cudaErrorInvalidDevicePointer, EXCEPTION_MSG("__device_grid::assign cannot assign range to noncontiguous memory") );
		#endif
	}

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, contiguous_memory_tag, device_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		for( iterator dest = begin(); dest != end() and first != last; ++dest, ++first ) *dest = *first;
		#else
		throw cuda_error( cudaErrorInvalidDevicePointer, EXCEPTION_MSG("__device_grid::assign cannot assign range from noncontiguous memory") );
		#endif
	}

	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last, noncontiguous_memory_tag, device_iterator_tag ) {
		#ifdef __CUDA_ARCH__
		assign( first, last, contiguous_sequence_tag(), device_iterator_tag() );
		#else
		throw cuda_error( cudaErrorInvalidDevicePointer, EXCEPTION_MSG("__device_grid::assign cannot assign range to and from noncontiguous memory") );
		#endif
	}

	HOST DEVICE void fill( const value_type& value, contiguous_memory_tag ) {
		#ifdef __CUDA_ARCH__
		for( iterator iter = begin(); iter != end(); ++iter ) *iter = value;
		#else
		for( size_type i = 0; i < number_rows(); ++i ) get_row(i).fill( value );
		#endif
	}

	HOST DEVICE void fill( const value_type& value, noncontiguous_memory_tag ) {
		#ifdef __CUDA_ARCH__
		for( iterator iter = begin(); iter != end(); ++iter ) *iter = value;
		#else
		throw cuda_error( cudaErrorInvalidDevicePointer, EXCEPTION_MSG("__device_grid::fill cannot fill non-contiguous device memory from host") );
		#endif
	}

	template<class Container>
	HOST void copy_to( Container& container, contiguous_memory_tag, std::random_access_iterator_tag ) const {
		const typename std::iterator_traits<typename Container::iterator>::difference_type n = std::distance( container.begin(), container.end() );
		if( n < 0 or static_cast<size_type>(n) < size() ) throw std::length_error( EXCEPTION_MSG("__device_grid::operator>> target container does not have sufficient space") );
		typename Container::iterator dest = container.begin();
		//TODO: should be able to optimize this, so that transfer occurs without staging vector v
		std::vector< value_type, host_allocator<value_type> > v( number_columns() );
		for( size_type i = 0; i < number_rows(); ++i ) {
			get_row(i).operator>>( v );
			for( typename std::vector< value_type, host_allocator<value_type> >::const_iterator iter = v.begin(); iter != v.end(); ++iter, ++dest ) *dest = *iter;
		}
	}

	template<class Container>
	HOST void copy_to( Container& container, contiguous_memory_tag, std::bidirectional_iterator_tag ) const {
		if( container.size() != size() ) throw std::length_error( EXCEPTION_MSG("__device_grid::operator>> target container does not have sufficient space") );
		std::vector< value_type, host_allocator<value_type> > v( size() );
		operator>>( v );
		typename Container::iterator dest = container.begin();
		for( typename std::vector< value_type, host_allocator<value_type> >::const_iterator src = v.begin(); src != v.end(); ++src, ++dest ) *dest = *src;
	}

	template<class Container> HOST inline void copy_to( Container& container, contiguous_memory_tag, std::forward_iterator_tag ) const { copy_to( container, contiguous_memory_tag(), std::bidirectional_iterator_tag() ); }
	template<class Container> HOST inline void copy_to( Container& container, contiguous_memory_tag, std::output_iterator_tag ) const { copy_to( container, contiguous_memory_tag(), std::bidirectional_iterator_tag() ); }

	template<class Container>
	HOST void copy_to( Container& container, contiguous_memory_tag, contiguous_device_iterator_tag ) const {
		const typename std::iterator_traits<typename Container::iterator>::difference_type n = container.end()-container.begin();
		if( n < 0 or static_cast<size_type>(n) < size() ) throw std::length_error( EXCEPTION_MSG("__device_grid::operator>> target container does not have sufficient space") );
		typename Container::iterator dest = container.begin();
		for( const_iterator src = begin(); src != end(); src += number_columns(), dest += number_columns() ) {
			CUDA_CALL( cudaMemcpy<value_type>( dest.operator->(), src.operator->(), number_columns(), cudaMemcpyDeviceToDevice ) );
		}
	}

	template<typename T2,typename PointerType2,typename RowCategory2,typename ColumnCategory2,typename ContainerType2>
	HOST void copy_to( __device_grid<T2,PointerType2,RowCategory2,ColumnCategory2,ContainerType2>& grid, contiguous_memory_tag, contiguous_device_iterator_tag, device_iterator_tag ) const {
		if( grid.number_rows() != number_rows() or grid.number_columns() != number_columns() ) throw std::length_error( EXCEPTION_MSG("__device_grid::operator>> target __device_grid does not match the size of source __target_grid") );
		for( size_type i = 0; i < number_rows(); ++i ) {
			typename __device_grid<T2,PointerType2,RowCategory2,ColumnCategory2,ContainerType2>::row_type dest = grid.get_row(i);
			const_row_type src = get_row(i);
			dest.assign( src.begin(), src.end() );
		}
	}

public:
	HOST DEVICE explicit __device_grid( managed_pointer ptr, size_type numberRows, size_type numberColumns ) : base_type( ptr, numberRows*numberColumns ), numberRows(numberRows) {}
	HOST DEVICE __device_grid( const __device_grid<T,PointerType,CategoryRow,CategoryColumn>& src ) : base_type(src), numberRows(src.numberRows) {}

	HOST DEVICE inline pointer data() const __NOEXCEPT__ { return base_type::data(); }
	HOST DEVICE inline size_type number_rows() const __NOEXCEPT__ { return numberRows; }
	HOST DEVICE inline size_type number_columns() const __NOEXCEPT__ { return base_type::size()/number_rows(); }
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return base_type::size(); }

	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(data()); }
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator(data()+static_cast<int>(size())); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(data()); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(data()+static_cast<int>(size())); }

	HOST DEVICE inline row_type get_row( const size_type rowIndex ) { return row_type( data()+static_cast<int>(number_columns()*rowIndex), number_columns() ); }
	HOST DEVICE inline const_row_type get_row( const size_type rowIndex ) const { return const_row_type( data()+static_cast<int>(number_columns()*rowIndex), number_columns() ); }
	HOST DEVICE inline column_type get_column( const size_type columnIndex ) { return column_type( striding_ptr<value_type,managed_pointer>( data()+static_cast<int>(columnIndex), number_columns() ), number_rows() ); }
	HOST DEVICE inline const_column_type get_column( const size_type columnIndex ) const { return const_column_type( striding_ptr<const value_type,managed_pointer>( data()+static_cast<int>(columnIndex), number_columns() ), number_rows() ); }

	HOST DEVICE inline row_type operator[]( const size_type index ) { return get_row(index); }
	HOST DEVICE inline const_row_type operator[]( const size_type index ) const { return get_row(index); }

	template<class Iterator>
	HOST DEVICE inline void assign( Iterator first, Iterator last ) { assign( first, last, column_category(), typename std::iterator_traits<Iterator>::iterator_category() ); }

	HOST DEVICE inline void fill( const value_type& value ) { fill( value, column_category() ); }

	template<class Container>
	HOST inline const __device_grid& operator>>( Container& container ) const {
		copy_to( container, column_category(), typename std::iterator_traits<typename Container::iterator>::iterator_category() );
		return *this;
	}

	template<typename T2,typename PointerType2,typename RowCategory2,typename ColumnCategory2,typename ContainerType2>
	HOST inline const __device_grid& operator>>( __device_grid<T2,PointerType2,RowCategory2,ColumnCategory2,ContainerType2>& container ) const {
		copy_to(
			container,
			column_category(),
			typename std::iterator_traits<typename __device_grid<T2,PointerType2,RowCategory2,ColumnCategory2,ContainerType2>::row_type::iterator>::iterator_category(),
			typename std::iterator_traits<typename __device_grid<T2,PointerType2,RowCategory2,ColumnCategory2,ContainerType2>::column_type::iterator>::iterator_category()
		);
		return *this;
	}

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
