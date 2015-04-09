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
// models.hpp
//
// The core memory models that provide basic operations on device memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MODELS_HPP
#define ECUDA_MODELS_HPP

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
#include "type_traits.hpp"

namespace ecuda {

///
/// \brief The core device memory model for data sequences used in the API.
///
/// A __device_sequence provides core functionality for a sequence of data.
/// Template parameters are used as trait tags to enable/disable/alter certain
/// functions.
///
/// For example, if two sequences are contiguous (DimensionType=__dimension_contiguous_tag)
/// then they could copy their contents from one to another using cudaMemcpy and only their
/// stored pointers.  If one sequence is contiguous, and the other is not, then copying from
/// the non-contiguous sequence to the contiguous sequence requires the use of a temporary
/// contiguous staging container.  On the other hand, attempting to copy from a contiguous to
/// non-contiguous sequence will result in an exception.
///
/// The types of iterators are also used to change the behaviour of the container at
/// compile-time.  For example, if a contiguous __device_sequence is provided with an iterator
/// from a std::vector, the action (i.e. host=>device transfer) will be different from the same
/// sequence being provided with an iterator to another __device_sequence (i.e. device=>device
/// transfer).
///
/// The purpose of this container is to abstract away as much core functionality as possible
/// while remaining flexible.  Any future architectural changes can ideally be made at this level
/// while the specific containers (e.g. array, vector) that an API user will use are derived from
/// __device_sequence and should remain fairly stable.
///
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

	typedef PointerType managed_pointer;

private:
	typedef DimensionType dimension_type;

public:
	typedef typename __device_sequence_traits<T,PointerType,DimensionType,ContainerType>::iterator iterator;
	typedef typename __device_sequence_traits<T,PointerType,DimensionType,ContainerType>::const_iterator const_iterator;
	typedef reverse_device_iterator<iterator> reverse_iterator;
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator;

private:
	// REMEMBER: length altered on device memory won't be reflected in the host object.
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

	HOST void copy_range_to( const_iterator first, const_iterator last, value_type* output, contiguous_device_iterator_tag, std::random_access_iterator_tag ) const {
		const typename std::iterator_traits<const_iterator>::difference_type n = last-first;
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( output, first.operator->(), n, cudaMemcpyDeviceToHost ) );
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
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( output.operator->(), first.operator->(), std::min(size(),static_cast<size_type>(n)), cudaMemcpyHostToDevice ) );
	}

	template<class Iterator>
	HOST void copy_range_from( Iterator first, Iterator last, iterator output, std::bidirectional_iterator_tag, contiguous_device_iterator_tag ) {
		std::vector< value_type, host_allocator<value_type> > v( first, last );
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( output.operator->(), &v.front(), std::min(v.size(),size()), cudaMemcpyHostToDevice ) );
	}

	template<class Iterator> HOST inline void copy_range_from( Iterator first, Iterator last, iterator output, std::forward_iterator_tag, contiguous_device_iterator_tag ) { copy_range_from( first, last, output, std::bidirectional_iterator_tag() ); }
	template<class Iterator> HOST inline void copy_range_from( Iterator first, Iterator last, iterator output, std::input_iterator_tag, contiguous_device_iterator_tag ) { copy_range_from( first, last, output, std::bidirectional_iterator_tag() ); }

	template<class Iterator>
	HOST void copy_range_from( Iterator first, Iterator last, iterator output, contiguous_device_iterator_tag, contiguous_device_iterator_tag ) {
		const typename std::iterator_traits<const_iterator>::difference_type n = last-first;
		CUDA_CALL( cudaMemcpy<typename std::remove_const<value_type>::type>( output.operator->(), first.operator->(), std::min(size(),static_cast<size_type>(n)), cudaMemcpyDeviceToDevice ) );
	}

public:
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
	template<typename T2>
	HOST DEVICE __device_sequence( const __device_sequence<T2,PointerType,DimensionType,ContainerType>& src ) : ptr(src.get_managed_pointer()), length(src.size()) {}

	HOST DEVICE inline pointer data() const __NOEXCEPT__ { return ptr; }
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return length; }

	HOST DEVICE inline managed_pointer get_managed_pointer() const __NOEXCEPT__ { return ptr; }

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

private:
	template<class Iterator>
	HOST void assign( Iterator first, Iterator last, contiguous_device_iterator_tag ) {
		const typename std::iterator_traits<Iterator>::difference_type n = last-first;
		if( n < 0 or static_cast<size_type>(n) != size() )
			throw std::length_error( EXCEPTION_MSG("ecuda::__device_sequence::assign range of first,last does not match sequence size" ) );
		copy_range_from( first, last, begin() );
	}

	template<class Iterator> HOST inline void assign( Iterator first, Iterator last, std::random_access_iterator_tag ) { assign( first, last, contiguous_device_iterator_tag() ); }

	template<class Iterator,typename FallthroughArgument> HOST inline void assign( Iterator first, Iterator last, FallthroughArgument ) {
		throw std::domain_error( EXCEPTION_MSG("ecuda::__device_sequence::assign is not usable with non-contigous iterator" ) );
	}

public:
	template<class Iterator>
	HOST inline void assign( Iterator first, Iterator last ) { assign( first, last, typename std::iterator_traits<Iterator>::iterator_category() );	}

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

	template<class Container>
	HOST inline __device_sequence& operator<<( Container& container ) {
		copy_range_from( container.begin(), container.end(), begin() );
		return *this;
	}

	HOST DEVICE __device_sequence& operator=( __device_sequence& other ) {
		ptr = other.ptr;
		length = other.length;
		return *this;
	}

};

///
/// \brief The core device memory model for data grids (2D memory blocks) used in the API.
///
/// A __device_grid provides core functionality for a 2D block of data. Template parameters
/// are used as trait tags to enable/disable/alter certain functions.
///
/// For example, if two blocks have contiguous columns and non-contiguous rows (which
/// is the case when pitched 2D memory allocated with cudaMallocPitch is used), then a
/// cudaMemcpy2D call is possible.
///
/// Models of individual rows and columns can be provided by instantiating a __device_sequence
/// with a length and a properly traversing pointer.
///
/// This model is effectively an extension to __device_sequence where rows and columns are
/// used to determine the index in the underlying sequence.
///
/// The purpose of this container is to abstract away as much core functionality as possible
/// while remaining flexible.  Any future architectural changes can ideally be made at this level
/// while the specific containers (e.g. matrix, cube) that an API user will use are derived from
/// __device_grid and should remain fairly stable.
///
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

	typedef PointerType managed_pointer;

private:
	typedef RowDimensionType row_dimension_type;
	typedef ColumnDimensionType column_dimension_type;

public:
	typedef typename __device_grid_traits<T,PointerType,RowDimensionType,ColumnDimensionType,ContainerType>::iterator iterator;
	typedef typename __device_grid_traits<T,PointerType,RowDimensionType,ColumnDimensionType,ContainerType>::const_iterator const_iterator;
	typedef reverse_device_iterator<iterator> reverse_iterator;
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator;

	typedef       __device_sequence<      value_type,pointer,column_dimension_type,__container_type_derived_tag> row_type;
	typedef       __device_sequence<      value_type,striding_ptr<value_type,value_type*>,row_dimension_type,__container_type_derived_tag> column_type;
	typedef const __device_sequence<const value_type,typename __pointer_traits<pointer>::const_pointer,                               column_dimension_type,__container_type_derived_tag> const_row_type;
	typedef const __device_sequence<const value_type,striding_ptr<const value_type,const value_type*>,row_dimension_type,__container_type_derived_tag> const_column_type;

private:
	// REMEMBER: numberRows altered on device memory won't be reflected inn the host object.
	size_type numberRows;

private:
	HOST void copy_to( __device_grid& other, __dimension_noncontiguous_tag, __dimension_contiguous_tag ) const {
		// assume a pitched memory model
		padded_ptr<const value_type,const value_type*,1> src( data() );
		padded_ptr<value_type,value_type*,1> dest( other.data() );
		CUDA_CALL( cudaMemcpy2D<value_type>( dest, dest.get_pitch(), src, src.get_pitch(), number_columns(), number_rows(), cudaMemcpyDeviceToDevice ) );
	}

	template<typename T2,typename PointerType2,typename ContainerType2>
	HOST void copy_to( __device_sequence<T2,PointerType2,__dimension_contiguous_tag,ContainerType2>& other, __dimension_noncontiguous_tag, __dimension_contiguous_tag ) const {
		typename __device_sequence<T2,PointerType2,__dimension_contiguous_tag,ContainerType2>::iterator dest = other.begin();
		for( size_type i = 0; i < number_rows(); ++i, dest += number_columns() ) {
			const_row_type row = get_row(i);
			row.copy_range_to( row.begin(), row.end(), dest );
		}
	}

	template<typename T2,typename PointerType2,typename ContainerType2>
	HOST void copy_to( __device_sequence<T2,PointerType2,__dimension_contiguous_tag,ContainerType2>& other, __dimension_contiguous_tag, __dimension_contiguous_tag ) const {
		typename __device_sequence<T2,PointerType2,__dimension_contiguous_tag,ContainerType2>::iterator dest = other.begin();
		CUDA_CALL( cudaMemcpy2D<value_type>( dest.operator->(), number_columns()*sizeof(value_type), data(), data().get_pitch(), number_columns(), number_rows(), cudaMemcpyDeviceToDevice ) );
	}

	template<typename T2,typename PointerType2,typename DimensionType2,typename ContainerType2,typename RowDimensionType2,typename ColumnDimensionType2>
	HOST inline void copy_to( __device_sequence<T2,PointerType2,DimensionType2,ContainerType2>& other, RowDimensionType2, ColumnDimensionType2 ) const {
		throw cuda_error( cudaErrorInvalidDevicePointer, EXCEPTION_MSG("ecuda::__device_sequence::copy_to() cannot copy to or from a non-contiguous range of elements") );
	}

	HOST DEVICE void fill( const value_type& value, padded_ptr<value_type,value_type*,1>& dest, __dimension_contiguous_tag ) {
		// this fill method is called iff. the underlying memory is a contiguous pitched memory block
		CUDA_CALL( cudaMemset2D<value_type>( dest, dest.get_pitch(), value, number_columns(), number_rows() ) );
	}

	HOST DEVICE void fill( const value_type& value, value_type*, __dimension_contiguous_tag ) {
		// this fill method is called iff. the underlying memory is a contiguous linear memory sequence
		// so it delegates the task to the base linear sequence container class
		base_type::fill( value );
	}

	template<class IrrelevantArgument1,class IrrelevantArgument2>
	HOST DEVICE void fill( const value_type& value, IrrelevantArgument1, IrrelevantArgument2 ) {
		// this is the default by-row fill method
		for( size_type i = 0; i < number_rows(); ++i ) get_row(i).fill( value );
	}

public:
	HOST DEVICE explicit __device_grid( managed_pointer ptr = managed_pointer(), size_type numberRows = 0, size_type numberColumns = 0 ) : base_type( ptr, numberRows*numberColumns ), numberRows(numberRows) {}
	template<typename T2,typename PointerType2>
	HOST DEVICE __device_grid( const __device_grid<T2,PointerType2,RowDimensionType,ColumnDimensionType,ContainerType>& src ) : base_type( src.get_managed_pointer(), src.number_rows()*src.number_columns() ), numberRows(src.number_rows()) {}
	HOST DEVICE __device_grid( const __device_grid<T,PointerType,RowDimensionType,ColumnDimensionType,ContainerType>& src ) : base_type(src), numberRows(src.numberRows) {}

	HOST DEVICE inline pointer data() const __NOEXCEPT__ { return base_type::data(); }
	HOST DEVICE inline size_type number_rows() const __NOEXCEPT__ { return numberRows; }
	HOST DEVICE inline size_type number_columns() const __NOEXCEPT__ { return base_type::size() == 0 ? 0 : base_type::size()/number_rows(); }
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return base_type::size(); }

	HOST DEVICE inline managed_pointer get_managed_pointer() const __NOEXCEPT__ { return base_type::get_managed_pointer(); }

	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(data()); }
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator(data()+static_cast<int>(size())); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(data()); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(data()+static_cast<int>(size())); }

	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(end()); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(begin()); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(end()); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(begin()); }

	HOST DEVICE inline row_type get_row( const size_type rowIndex ) { return row_type( data()+static_cast<int>(number_columns()*rowIndex), number_columns() ); }
	HOST DEVICE inline const_row_type get_row( const size_type rowIndex ) const { return const_row_type( data()+static_cast<int>(number_columns()*rowIndex), number_columns() ); }
	HOST DEVICE inline column_type get_column( const size_type columnIndex ) {
		return column_type( column_type::pointer( data()+static_cast<int>(columnIndex), number_columns() ), number_rows() );
		//return column_type( striding_ptr<value_type,pointer>( data()+static_cast<int>(columnIndex), number_columns() ), number_rows() );
	}
	HOST DEVICE inline const_column_type get_column( const size_type columnIndex ) const {
		return const_column_type( const_column_type::pointer( data()+static_cast<int>(columnIndex), number_columns() ), number_rows() );
		//return const_column_type( striding_ptr<const value_type,pointer>( data()+static_cast<int>(columnIndex), number_columns() ), number_rows() );
	}

	HOST DEVICE inline row_type operator[]( const size_type index ) { return get_row(index); }
	HOST DEVICE inline const_row_type operator[]( const size_type index ) const { return get_row(index); }

	HOST DEVICE inline reference at( const size_type rowIndex, const size_type columnIndex ) { return *(data()+static_cast<int>(number_columns()*rowIndex+columnIndex)); }
	HOST DEVICE inline const_reference at( const size_type rowIndex, const size_type columnIndex ) const { return *(data()+static_cast<int>(number_columns()*rowIndex+columnIndex)); }

	HOST DEVICE inline void fill( const value_type& value ) { fill( value, data(), column_dimension_type() ); }

	HOST DEVICE inline void swap( __device_grid& other ) {
		base_type::swap( other );
		#ifdef __CUDA_ARCH__
		ecuda::swap( numberRows, other.numberRows );
		#else
		std::swap( numberRows, other.numberRows );
		#endif
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

	template<typename T2,typename PointerType2,typename DimensionType2,typename ContainerType2>
	HOST inline const __device_grid& operator>>( __device_sequence<T2,PointerType2,DimensionType2,ContainerType2>& other ) const {
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

};

///
/// \brief Simple utility container to make C-style arrays useable by ecuda containers.
///
/// Given a C-style array consisting of a pointer and a length, this is a simple wrapper
/// that provides the C-style array with the ability to produce iterators so that
/// ecuda containers can use them to copy to/from.
///
/// This is handy for C-style APIs that might be useful, such as the GNU Scientific
/// Library.
///
/// \code{.cpp}
/// gsl_matrix* mat = gsl_matrix_alloc( 10, 20 );
/// // ... prepare matrix values
/// ecuda::host_array_proxy<double> proxy( mat->data, 10*20 );
/// ecuda::matrix<double> deviceMatrix( 10, 20 );
/// deviceMatrix.assign( proxy.begin(), proxy.end() ); // copies gsl_matrix to device matrix
/// deviceMatrix >> proxy; // copies device matrix to gsl_matrix
/// // proxy container can now be safely discarded, since the
/// // manipulations of the data are reflected in the original gsl_matrix.
/// \endcode
///
template<class T>
class host_array_proxy {
public:
	typedef T value_type;
	typedef T* pointer;
	typedef T& reference;
	typedef const T& const_reference;
	typedef std::ptrdiff_t difference_type;
	typedef std::size_t size_type;

	typedef host_pointer_iterator<value_type> iterator;
	typedef host_pointer_iterator<const value_type> const_iterator;

private:
	pointer ptr;
	size_type len;
public:
	host_array_proxy( T* ptr, const size_type len ) : ptr(ptr), len(len) {}

	inline iterator begin() { return iterator(ptr); }
	inline iterator end() { return iterator(ptr+len); }
	inline const_iterator begin() const { return const_iterator(ptr); }
	inline const_iterator end() const { return const_iterator(ptr+len); }

};


} // namespace ecuda

#endif
