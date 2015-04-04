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

#ifdef __CPP11_SUPPORTED__
#include <initializer_list>
#endif

#include "global.hpp"
#include "apiwrappers.hpp"
#include "iterators.hpp"
#include "device_ptr.hpp"
#include "striding_ptr.hpp"
#include "padded_ptr.hpp"

namespace ecuda {

struct __dimension_contiguous_tag {};
struct __dimension_noncontiguous_tag {};

struct __container_type_base_tag {};
struct __container_type_derived_tag {};

template<typename T,typename PointerType,typename DimensionType,typename ContainerType> struct __device_sequence_traits;
template<typename T,typename PointerType> struct __device_sequence_traits<T,PointerType,__dimension_contiguous_tag,__container_type_base_tag> {
	typedef typename PointerType::pointer pointer;
	typedef contiguous_device_iterator<T> iterator;
	typedef contiguous_device_iterator<const T> const_iterator;
};
template<typename T,typename PointerType> struct __device_sequence_traits<T,PointerType,__dimension_noncontiguous_tag,__container_type_base_tag> {
	typedef typename PointerType::pointer pointer;
	typedef device_iterator<T,PointerType> iterator;
	typedef device_iterator<const T,PointerType> const_iterator;
};
template<typename T,typename PointerType> struct __device_sequence_traits<T,PointerType,__dimension_contiguous_tag,__container_type_derived_tag> {
	typedef PointerType pointer;
	typedef contiguous_device_iterator<T> iterator;
	typedef contiguous_device_iterator<const T> const_iterator;
};
template<typename T,typename PointerType> struct __device_sequence_traits<T,PointerType,__dimension_noncontiguous_tag,__container_type_derived_tag> {
	typedef PointerType pointer;
	typedef device_iterator<T,PointerType> iterator;
	typedef device_iterator<const T,PointerType> const_iterator;
};

template<typename T,typename PointerType,typename RowDimensionType,typename ColumnDimensionType,typename ContainerType> struct __device_grid_traits;
template<typename T,typename PointerType,typename RowDimensionType,typename ColumnDimensionType> struct __device_grid_traits<T,PointerType,RowDimensionType,ColumnDimensionType,__container_type_base_tag> {
	typedef typename PointerType::pointer pointer;
	typedef device_iterator<T,PointerType> iterator;
	typedef device_iterator<const T,PointerType> const_iterator;
};
template<typename T,typename PointerType> struct __device_grid_traits<T,PointerType,__dimension_contiguous_tag,__dimension_contiguous_tag,__container_type_base_tag> {
	typedef typename PointerType::pointer pointer;
	typedef contiguous_device_iterator<T> iterator;
	typedef contiguous_device_iterator<const T> const_iterator;
};
template<typename T,typename PointerType,typename RowDimensionType,typename ColumnDimensionType> struct __device_grid_traits<T,PointerType,RowDimensionType,ColumnDimensionType,__container_type_derived_tag> {
	typedef PointerType pointer;
	typedef device_iterator<T,PointerType> iterator;
	typedef device_iterator<const T,PointerType> const_iterator;
};
template<typename T,typename PointerType> struct __device_grid_traits<T,PointerType,__dimension_contiguous_tag,__dimension_contiguous_tag,__container_type_derived_tag> {
	typedef PointerType pointer;
	typedef contiguous_device_iterator<T> iterator;
	typedef contiguous_device_iterator<const T> const_iterator;
};


template<typename T,typename PointerType=typename reference<T>::pointer_type,typename DimensionType=__dimension_contiguous_tag,typename ContainerType=__container_type_base_tag>
class __device_sequence
{
public:
	typedef T value_type;
	typedef typename __device_sequence_traits<T,PointerType,DimensionType,ContainerType>::pointer pointer;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

private:
	typedef PointerType managed_pointer;
	typedef DimensionType dimension_type;

public:
	typedef typename __device_sequence_traits<T,PointerType,DimensionType,ContainerType>::iterator iterator;
	typedef typename __device_sequence_traits<T,PointerType,DimensionType,ContainerType>::const_iterator const_iterator;
	typedef reverse_device_iterator<iterator> reverse_iterator;
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator;

private:
	managed_pointer ptr;
	size_type length;

private:
	HOST DEVICE void fill( const value_type& value, __dimension_contiguous_tag ) {
		#ifdef __CUDA_ARCH__
		for( iterator iter = begin(); iter != end(); ++iter ) *iter = value;
		#else
		CUDA_CALL( cudaMemset<value_type>( data(), value, size() ) );
		#endif
	}

	HOST DEVICE void fill( const value_type& value, __dimension_noncontiguous_tag ) {
		#ifdef __CUDA_ARCH__
		for( iterator iter = begin(); iter != end(); ++iter ) *iter = value;
		#else
		throw cuda_error( cudaErrorInvalidDevicePointer, EXCEPTION_MSG("__device_sequence::fill cannot fill non-contiguous device memory from host") );
		#endif
	}

	template<class Iterator>
	HOST void copy_range_to( const_iterator first, const_iterator last, Iterator output, contiguous_device_iterator_tag, std::random_access_iterator_tag ) const {
		const typename std::iterator_traits<const_iterator>::difference_type n = last-first;
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( output.operator->(), first.operator->(), n, cudaMemcpyDeviceToHost ) );
	}

	template<class Iterator>
	HOST void copy_range_to( const_iterator first, const_iterator last, Iterator output, contiguous_device_iterator_tag, std::bidirectional_iterator_tag ) const {
		const typename std::iterator_traits<const_iterator>::difference_type n = last-first;
		std::vector< value_type, host_allocator<value_type> > v( n );
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( output.operator->(), &v.front(), n, cudaMemcpyDeviceToHost ) );
		std::copy( v.begin(), v.end(), output );
	}

	template<class Iterator> HOST inline void copy_range_to( const_iterator first, const_iterator last, Iterator output, contiguous_device_iterator_tag, std::forward_iterator_tag ) const { copy_range_to( first, last, output, std::bidirectional_iterator_tag() ); }
	template<class Iterator> HOST inline void copy_range_to( const_iterator first, const_iterator last, Iterator output, contiguous_device_iterator_tag, std::output_iterator_tag ) const { copy_range_to( first, last, output, std::bidirectional_iterator_tag() ); }

	template<class Iterator>
	HOST void copy_range_to( const_iterator first, const_iterator last, Iterator output, contiguous_device_iterator_tag, contiguous_device_iterator_tag ) const {
		const typename std::iterator_traits<const_iterator>::difference_type n = last-first;
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( output.operator->(), first.operator->(), n, cudaMemcpyDeviceToDevice ) );
	}


	template<class Iterator>
	HOST void copy_range_from( Iterator first, Iterator last, iterator output, std::random_access_iterator_tag, contiguous_device_iterator_tag ) {
		const typename std::iterator_traits<Iterator>::difference_type n = std::distance(first,last);
		std::cerr << "first.operator->()=" << first.operator->() << std::endl;
		std::cerr << "output.operator->()=" << output.operator->() << std::endl;
		std::cerr << "n=" << n << std::endl;
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( output.operator->(), first.operator->(), n, cudaMemcpyHostToDevice ) );
	}

	template<class Iterator>
	HOST void copy_range_from( Iterator first, Iterator last, iterator output, std::bidirectional_iterator_tag, contiguous_device_iterator_tag ) {
		std::vector< value_type, host_allocator<value_type> > v( first, last );
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( output.operator->(), &v.front(), v.size(), cudaMemcpyHostToDevice ) );
	}

	template<class Iterator> HOST inline void copy_range_from( Iterator first, Iterator last, iterator output, std::forward_iterator_tag, contiguous_device_iterator_tag ) { copy_range_from( first, last, output, std::bidirectional_iterator_tag() ); }
	template<class Iterator> HOST inline void copy_range_from( Iterator first, Iterator last, iterator output, std::input_iterator_tag, contiguous_device_iterator_tag ) { copy_range_from( first, last, output, std::bidirectional_iterator_tag() ); }

	template<class Iterator>
	HOST void copy_range_from( Iterator first, Iterator last, iterator output, contiguous_device_iterator_tag, contiguous_device_iterator_tag ) {
		const typename std::iterator_traits<const_iterator>::difference_type n = last-first;
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( output.operator->(), first.operator->(), n, cudaMemcpyDeviceToDevice ) );
	}

protected:
public: // NOTE: for debugging
	template<class Iterator>
	HOST inline void copy_range_to( const_iterator first, const_iterator last, Iterator output ) const {
		copy_range_to( first, last, output, typename std::iterator_traits<const_iterator>::iterator_category(), typename std::iterator_traits<Iterator>::iterator_category() );
	}

	template<class Iterator>
	HOST inline void copy_range_from( Iterator first, Iterator last, iterator output ) {
		copy_range_from( first, last, output, typename std::iterator_traits<Iterator>::iterator_category(), typename std::iterator_traits<const_iterator>::iterator_category() );
	}

public:
	HOST DEVICE explicit __device_sequence( managed_pointer ptr = managed_pointer(), size_type length = 0 ) : ptr(ptr), length(length) {}
	HOST DEVICE __device_sequence( const __device_sequence<T,PointerType,DimensionType,ContainerType>& src ) : ptr(src.ptr), length(src.length) {}

	HOST DEVICE inline pointer data() const __NOEXCEPT__ { return ptr; }
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return length; }

	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(data()); }
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
		return iterator( static_cast<typename iterator::pointer>(data())+static_cast<int>(size()) );
	}
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(data()); }
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
		return const_iterator( static_cast<typename const_iterator::pointer>(data())+static_cast<int>(size()) );
	}

	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(end()); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(begin()); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(end()); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(begin()); }

	DEVICE inline reference operator[]( const size_type index ) { return *(data()+static_cast<int>(index)); }
	DEVICE inline const_reference operator[]( const size_type index ) const { return *(data()+static_cast<int>(index)); }

	HOST DEVICE inline void fill( const value_type& value ) { fill( value, dimension_type() ); }

	HOST DEVICE inline void swap( __device_sequence& other ) {
		#ifdef __CUDA_ARCH__
		ecuda::swap( ptr, other.ptr );
		ecuda::swap( length, other.length );
		#else
		std::swap( ptr, other.ptr );
		std::swap( length, other.length );
		#endif
	}

	HOST DEVICE bool operator==( const __device_sequence& other ) const {
		if( size() != other.size() ) return false;
		#ifdef __CUDA_ARCH__
		const_iterator iter2 = other.begin();
		for( const_iterator iter1 = begin(); iter1 != end(); ++iter1, ++iter2 ) if( !(*iter1 == *iter2) ) return false;
		return true;
		#else
		std::vector< value_type, host_allocator<value_type> > v1( size() );
		std::vector< value_type, host_allocator<value_type> > v2( other.size() );
		operator>>( v1 );
		other.operator>>( v2 );
		return v1 == v2;
		#endif
	}

	HOST DEVICE inline bool operator!=( const __device_sequence& other ) const { return !operator==( other ); }

	HOST DEVICE bool operator<( const __device_sequence& other ) const {
		#ifdef __CUDA_ARCH__
		return ecuda::lexicographical_compare( begin(), end(), other.begin(), other.end() );
		#else
		std::vector< value_type, host_allocator<value_type> > v1( size() );
		std::vector< value_type, host_allocator<value_type> > v2( other.size() );
		operator>>( v1 );
		other.operator>>( v2 );
		return v1 < v2;
		#endif
	}

	HOST DEVICE bool operator>( const __device_sequence& other ) const {
		#ifdef __CUDA_ARCH__
		return ecuda::lexicographical_compare( other.begin(), other.end(), begin(), end() );
		#else
		std::vector< value_type, host_allocator<value_type> > v1( size() );
		std::vector< value_type, host_allocator<value_type> > v2( other.size() );
		operator>>( v1 );
		other.operator>>( v2 );
		return v1 > v2;
		#endif
	}

	HOST DEVICE inline bool operator<=( const __device_sequence& other ) const { return !operator>( other ); }

	HOST DEVICE inline bool operator>=( const __device_sequence& other ) const { return !operator<( other ); }

	template<class Container>
	HOST inline const __device_sequence& operator>>( Container& container ) const {
		copy_range_to( begin(), end(), container.begin() );
		return *this;
	}

	HOST DEVICE __device_sequence& operator=( __device_sequence& other ) {
		ptr = other.ptr;
		length = other.length;
		return *this;
	}

};

template<typename T,typename PointerType=typename reference<T>::pointer_type,class RowDimensionType=__dimension_noncontiguous_tag,class ColumnDimensionType=__dimension_contiguous_tag,class ContainerType=__container_type_base_tag>
class __device_grid : private __device_sequence<T,PointerType,ColumnDimensionType,ContainerType>
{
private:
	typedef __device_sequence<T,PointerType,ColumnDimensionType,ContainerType> base_type;

public:
	typedef T value_type;
	typedef typename __device_sequence_traits<T,PointerType,ColumnDimensionType,ContainerType>::pointer pointer;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

private:
	typedef PointerType managed_pointer;
	typedef RowDimensionType row_dimension_type;
	typedef ColumnDimensionType column_dimension_type;

public:
	typedef typename __device_grid_traits<T,PointerType,RowDimensionType,ColumnDimensionType,ContainerType>::iterator iterator;
	typedef typename __device_grid_traits<T,PointerType,RowDimensionType,ColumnDimensionType,ContainerType>::const_iterator const_iterator;
	typedef reverse_device_iterator<iterator> reverse_iterator;
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator;

	typedef __device_sequence<value_type,pointer,column_dimension_type,__container_type_derived_tag> row_type;
	typedef const __device_sequence<const value_type,pointer,column_dimension_type,__container_type_derived_tag> const_row_type;
	typedef __device_sequence<value_type,striding_ptr<value_type,pointer>,row_dimension_type,__container_type_derived_tag> column_type;
	typedef const __device_sequence<const value_type,striding_ptr<value_type,pointer>,row_dimension_type,__container_type_derived_tag> const_column_type;

private:
	size_type numberRows;

/*
private:
	template<class Container>
	HOST void copy_to( Container& container, __dimension_contiguous_tag, std::random_access_iterator_tag ) const {
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
	HOST void copy_to( Container& container, __dimension_contiguous_tag, std::bidirectional_iterator_tag ) const {
		if( container.size() != size() ) throw std::length_error( EXCEPTION_MSG("__device_grid::operator>> target container does not have sufficient space") );
		std::vector< value_type, host_allocator<value_type> > v( size() );
		operator>>( v );
		typename Container::iterator dest = container.begin();
		for( typename std::vector< value_type, host_allocator<value_type> >::const_iterator src = v.begin(); src != v.end(); ++src, ++dest ) *dest = *src;
	}

	template<class Container> HOST inline void copy_to( Container& container, __dimension_contiguous_tag, std::forward_iterator_tag ) const { copy_to( container, __dimension_contiguous_tag(), std::bidirectional_iterator_tag() ); }
	template<class Container> HOST inline void copy_to( Container& container, __dimension_contiguous_tag, std::output_iterator_tag ) const { copy_to( container, __dimension_contiguous_tag(), std::bidirectional_iterator_tag() ); }

	template<class Container>
	HOST void copy_to( Container& container, __dimension_contiguous_tag, contiguous_device_iterator_tag ) const {
		const typename std::iterator_traits<typename Container::iterator>::difference_type n = container.end()-container.begin();
		if( n < 0 or static_cast<size_type>(n) < size() ) throw std::length_error( EXCEPTION_MSG("__device_grid::operator>> target container does not have sufficient space") );
		typename Container::iterator dest = container.begin();
		for( const_iterator src = begin(); src != end(); src += number_columns(), dest += number_columns() ) {
			CUDA_CALL( cudaMemcpy<value_type>( dest.operator->(), src.operator->(), number_columns(), cudaMemcpyDeviceToDevice ) );
		}
	}

	template<typename T2,typename PointerType2,typename RowCategory2,typename ColumnCategory2,typename ContainerType2>
	HOST void copy_to( __device_grid<T2,PointerType2,RowCategory2,ColumnCategory2,ContainerType2>& grid, __dimension_contiguous_tag, contiguous_device_iterator_tag, device_iterator_tag ) const {
		if( grid.number_rows() != number_rows() or grid.number_columns() != number_columns() ) throw std::length_error( EXCEPTION_MSG("__device_grid::operator>> target __device_grid does not match the size of source __target_grid") );
		for( size_type i = 0; i < number_rows(); ++i ) {
			typename __device_grid<T2,PointerType2,RowCategory2,ColumnCategory2,ContainerType2>::row_type dest = grid.get_row(i);
			const_row_type src = get_row(i);
			dest.assign( src.begin(), src.end() );
		}
	}
*/
private:
	HOST void copy_to( __device_grid& other, __dimension_noncontiguous_tag, __dimension_contiguous_tag ) const {
		// assume a pitched memory model
		padded_ptr<const value_type,const value_type*,1> src( data() );
		padded_ptr<value_type,value_type*,1> dest( other.data() );
		CUDA_CALL( cudaMemcpy2D<value_type>( dest, dest.get_pitch(), src, src.get_pitch(), number_columns(), number_rows(), cudaMemcpyDeviceToDevice ) );
	}

public:
	HOST DEVICE explicit __device_grid( managed_pointer ptr = managed_pointer(), size_type numberRows = 0, size_type numberColumns = 0 ) : base_type( ptr, numberRows*numberColumns ), numberRows(numberRows) {}
	HOST DEVICE __device_grid( const __device_grid<T,PointerType,RowDimensionType,ColumnDimensionType,ContainerType>& src ) : base_type(src), numberRows(src.numberRows) {}

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
	HOST DEVICE inline column_type get_column( const size_type columnIndex ) { return column_type( striding_ptr<value_type,pointer>( data()+static_cast<int>(columnIndex), number_columns() ), number_rows() ); }
	HOST DEVICE inline const_column_type get_column( const size_type columnIndex ) const { return const_column_type( striding_ptr<const value_type,pointer>( data()+static_cast<int>(columnIndex), number_columns() ), number_rows() ); }

	HOST DEVICE inline row_type operator[]( const size_type index ) { return get_row(index); }
	HOST DEVICE inline const_row_type operator[]( const size_type index ) const { return get_row(index); }

	HOST DEVICE inline reference at( const size_type rowIndex, const size_type columnIndex ) { return *(data()+(number_columns()*rowIndex+columnIndex)); }
	HOST DEVICE inline const_reference at( const size_type rowIndex, const size_type columnIndex ) const { return *(data()+(number_columns()*rowIndex+columnIndex)); }

	HOST DEVICE void fill( const value_type& value ) {
		//TODO: utilize cudaMemset2D when appropriate
		for( size_type i = 0; i < number_rows(); ++i ) get_row(i).fill( value );
	}

	template<class Container>
	HOST inline const __device_grid& operator>>( Container& container ) const {
		typename Container::iterator iter = container.begin();
		for( size_type i = 0; i < number_rows(); ++i, iter += number_columns() ) {
			const_row_type row = get_row(i);
			row.copy_range_to( row.begin(), row.end(), iter );
		}
		return *this;
	}

	HOST inline const __device_grid& operator>>( __device_grid& other ) const {
		if( number_rows() != other.number_rows() or number_columns() != other.number_columns() )
			throw std::length_error( EXCEPTION_MSG("ecuda::__device_grid::operator>>() target __device_grid has different dimensions") );
		copy_to( other, row_dimension_type(), column_dimension_type() );
		return *this;
	}

	template<typename T2,typename PointerType2,class RowDimensionType2,class ColumnDimensionType2,class ContainerType2>
	HOST inline const __device_grid& operator>>( __device_grid<T2,PointerType2,RowDimensionType2,ColumnDimensionType2,ContainerType2>& other ) const {
		if( number_rows() != other.number_rows() or number_columns() != other.number_columns() )
			throw std::length_error( EXCEPTION_MSG("ecuda::__device_grid::operator>>() target __device_grid has different dimensions") );
		for( size_type i = 0; i < number_rows(); ++i ) {
			const_row_type src = get_row(i);
			typename __device_grid<T2,PointerType2,RowDimensionType2,ColumnDimensionType2,ContainerType2>::row_type dest = other.get_row(i);
			src.copy_range_to( src.begin(), src.end(), dest.begin() );
		}
		return *this;
	}

//	template<typename T2,typename PointerType2,typename RowCategory2,typename ColumnCategory2,typename ContainerType2>
//	HOST inline const __device_grid& operator>>( __device_grid<T2,PointerType2,RowCategory2,ColumnCategory2,ContainerType2>& container ) const {
//		copy_to(
//			container,
//			column_dimension_type(),
//			typename std::iterator_traits<typename __device_grid<T2,PointerType2,RowCategory2,ColumnCategory2,ContainerType2>::row_type::iterator>::iterator_category(),
//			typename std::iterator_traits<typename __device_grid<T2,PointerType2,RowCategory2,ColumnCategory2,ContainerType2>::column_type::iterator>::iterator_category()
//		);
//		return *this;
//	}

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
