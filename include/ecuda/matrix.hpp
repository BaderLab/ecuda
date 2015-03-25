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
// matrix.hpp
// An STL-like structure that resides in video memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MATRIX_HPP
#define ECUDA_MATRIX_HPP

#include <vector>

#include "config.hpp"
#if HAVE_ESTD_LIBRARY > 0
#include <estd/matrix.hpp>
#endif
#include "global.hpp"
#include "allocators.hpp"
#include "apiwrappers.hpp"
#include "memory.hpp"
#include "device_ptr.hpp"
#include "padded_ptr.hpp"
#include "striding_ptr.hpp"

namespace ecuda {

///
/// \brief A video-memory bound matrix container.
///
/// A matrix is defined as a 2D structure of dimensions rows*columns.  The default implementation
/// uses pitched memory where a 2D block of video memory is allocated with width=columns and height=rows.
/// Pitched memory is aligned in a device-dependent manner so that calls to individual elements can be
/// threaded more efficiently (i.e. minimizing the number of read operations required to supply data to
/// multiple threads). Consult the CUDA API documentation for a more verbose explanation.
///
/// Memory use can be conceptualized as:
/// \code
///       |- columns -|
///       |---- pitch -----|
///    _  +-----------+----+
///   |   |           |xxxx| x = allocated but not used, just padding to
///   |   |           |xxxx|     enforce an efficient memory alignment
///  rows |           |xxxx|
///   |   |           |xxxx|
///   |_  +-----------+----+
/// \endcode
///
/// As a result, it is highly desirable for threading to utilize a column-wise orientation.
/// For example, a good kernel to perform an operation on the elements of a matrix might be:
///
/// \code{.cpp}
/// template<typename T> __global__ void doMatrixOperation( ecuda::matrix<T> matrix ) {
///    const int row = blockIdx.x;
///    const int col = blockDim.y*gridDim.y; // each thread gets a different column value
///    if( row < matrix.number_rows() and col < matrix.number_columns() ) {
///       T& value = matrix[row][col];
///       // ... do work on value
///    }
/// }
/// \endcode
///
/// This could be called from host code like:
/// \code{.cpp}
/// ecuda::matrix<double> matrix( 100, 1000 );
/// // ... fill matrix with data
/// dim3 grid( 100, 1 ), block( 1, 1000 );
/// doMatrixOperation<<<grid,block>>>( matrix );
/// \endcode
///
/// Unfortunately, CUDA solutions are very problem specific, so there is no generally applicable example for
/// specifying how thread blocks should be defined.  The size of the matrix, hardware limitations, CUDA API
/// limitations, etc. all play a part.  For example, the above implementation won't work in earlier versions
/// of CUDA since blockDim.y is limited to 512.
///
/// Just keep in mind that the column dimension lies in contiguous memory, and the row dimension is contiguous
/// blocks of columns; thus, an implementation that aims to have concurrently running threads accessing
/// column >>>> row will run much more efficiently.
///
template< typename T, class Alloc=DevicePitchAllocator<T> >
class matrix {

public:
	typedef T value_type; //!< cell data type
	typedef Alloc allocator_type; //!< allocator type
	typedef std::size_t size_type; //!< index data type
	typedef std::ptrdiff_t difference_type; //!<
	typedef value_type& reference; //!< cell reference type
	typedef const value_type& const_reference; //!< cell const reference type
	typedef value_type* pointer; //!< cell pointer type
	typedef const value_type* const_pointer; //!< cell const pointer type

	typedef contiguous_memory_proxy< value_type, pointer > row_type; //!< matrix row container type
	typedef contiguous_memory_proxy< value_type, padded_ptr<value_type,striding_ptr<value_type>,1> > column_type; //!< matrix column container type
	typedef contiguous_memory_proxy< const value_type, const_pointer > const_row_type; //!< matrix const row container type
	typedef contiguous_memory_proxy< const value_type, padded_ptr<const value_type,striding_ptr<const value_type>,1> > const_column_type; //!< matrix const column container type

	typedef pointer_iterator< value_type, padded_ptr<value_type,pointer,1> > iterator; //!< iterator type
	typedef pointer_iterator< const value_type, padded_ptr<const value_type,const_pointer,1> > const_iterator; //!< const iterator type
	typedef pointer_reverse_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef pointer_reverse_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type


private:
	// REMEMBER: numberRows, numberColumns, and pitch altered on device memory won't be
	//           reflected on the host object. Don't allow the device to perform any operations that
	//           change their value.
	size_type numberRows; //!< number of matrix rows
	size_type numberColumns; //!< number of matrix columns
	size_type pitch; //!< the padded width of the 2D memory allocation in bytes
	device_ptr<T> deviceMemory; //!< smart pointer to video card memory
	allocator_type allocator;

public:
	HOST matrix( const size_type numberRows=0, const size_type numberColumns=0, const_reference value = T(), const Alloc& allocator = Alloc() );

	HOST DEVICE matrix( const matrix<T>& src );

	#if HAVE_ESTD_LIBRARY > 0
	template<typename U,typename V>
	HOST matrix( const estd::matrix<T,U,V>& src, const Alloc& allocator = Alloc() );
	#endif

	HOST DEVICE virtual ~matrix() {}

	HOST inline allocator_type get_allocator() const { return allocator; }
	HOST DEVICE inline size_type get_pitch() const { return pitch; }

	template<class RandomAccessIterator>
	HOST void assign( RandomAccessIterator begin, RandomAccessIterator end );

	/*
	 * Deprecating this function since the STL standard seems to specify that the at() accessor
	 * must implement range checking that throws an exception on failure.  Since exceptions are
	 * not supported within a CUDA kernel, this cannot be satisfied.
	 *
	DEVICE inline reference at( size_type rowIndex, size_type columnIndex ) {
		//if( rowIndex >= row_size() ) throw std::out_of_range( "ecuda::matrix::at() rowIndex parameter is out of range" );	
		//if( columnIndex >= column_size() ) throw std::out_of_range( "ecuda::matrix::at() columnIndex parameter is out of range" );	
		return *allocator.address( data(), rowIndex, columnIndex, pitch ); 
	}
	DEVICE inline reference at( size_type index ) { return at( index / numberColumns, index % numberColumns ); }

	DEVICE inline const_reference at( size_type rowIndex, size_type columnIndex ) const {
		//if( rowIndex >= row_size() ) throw std::out_of_range( "ecuda::matrix::at() rowIndex parameter is out of range" );	
		//if( columnIndex >= column_size() ) throw std::out_of_range( "ecuda::matrix::at() columnIndex parameter is out of range" );	
		return *allocator.address( data(), rowIndex, columnIndex, pitch ); 
	}
	DEVICE inline const_reference at( size_type index ) const { return at( index / numberColumns, index % numberColumns ); }
	*/

	HOST DEVICE inline row_type get_row( const size_type rowIndex ) { return row_type( allocator.address( data(), rowIndex, 0, pitch ), number_columns() ); }
	HOST DEVICE inline column_type get_column( const size_type columnIndex ) {
		pointer p = allocator.address( data(), 0, columnIndex, pitch );
		striding_ptr<value_type> sp( p, number_columns() );
		padded_ptr< value_type, striding_ptr<value_type>, 1 > pp( sp, 1, pitch-numberColumns*sizeof(value_type), 0 );
		return column_type( pp, number_rows() );
	}
	HOST DEVICE inline const_row_type get_row( const size_type rowIndex ) const { return const_row_type( allocator.address( data(), rowIndex, 0, pitch ), number_columns() ); }
	HOST DEVICE inline const_column_type get_column( const size_type columnIndex ) const {
		const_pointer p = allocator.address( data(), 0, columnIndex, pitch );
		striding_ptr<const value_type> sp( p, number_columns() );
		padded_ptr< const value_type, striding_ptr<const value_type>, 1 > pp( sp, 1, pitch-numberColumns*sizeof(value_type), 0 );
		return const_column_type( pp, number_rows() );
		//return const_column_type( strided_ptr<const value_type,const_pointer,1>( allocator.address( data(), 0, columnIndex, pitch ), get_pitch() ), number_rows() );
	}

	HOST DEVICE inline row_type operator[]( const size_type rowIndex ) { return get_row(rowIndex); }
	HOST DEVICE inline const_row_type operator[]( const size_type rowIndex ) const { return get_row(rowIndex); }

	DEVICE inline reference front() { return *data(); }
	DEVICE inline reference back() { return *allocator.address( data(), number_rows()-1, number_columns()-1, pitch ); }
	DEVICE inline const_reference front() const { return *data(); }
	DEVICE inline const_reference back() const { return *allocator.address( data(), number_rows()-1, number_columns()-1, pitch ); }
	HOST DEVICE inline T* data() { return deviceMemory.get(); }
	HOST DEVICE inline const T* data() const { return deviceMemory.get(); }

	HOST DEVICE inline iterator begin() { return iterator( padded_ptr<value_type,pointer,1>( data(), number_columns(), pitch-number_columns()*sizeof(value_type), 0 ) ); }
	HOST DEVICE inline iterator end() { return iterator( padded_ptr<value_type,pointer,1>( allocator.address( data(), number_rows(), 0, pitch ), number_columns(), pitch-number_columns()*sizeof(value_type), 0 ) ); }
	HOST DEVICE inline const_iterator begin() const { return const_iterator( padded_ptr<const value_type,const_pointer,1>( data(), number_columns(), pitch-number_columns()*sizeof(value_type), 0 ) ); }
	HOST DEVICE inline const_iterator end() const { return const_iterator( padded_ptr<const value_type,const_pointer,1>( allocator.address( data(), number_rows(), 0, pitch ), number_columns(), pitch-number_columns()*sizeof(value_type), 0 ) ); }

	HOST DEVICE inline reverse_iterator rbegin() { return reverse_iterator(end()); }
	HOST DEVICE inline reverse_iterator rend() { return reverse_iterator(begin()); }
	HOST DEVICE inline const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
	HOST DEVICE inline const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

	HOST DEVICE inline bool empty() const { return !number_rows() or !number_columns(); }
	HOST DEVICE inline size_type size() const { return number_rows()*number_columns(); }
	///
	/// \brief Returns the maximum number of elements the container is able to hold due to system
	/// or library implementation limitations.
	///
	/// \returns Maximum number of elements.
	///
	HOST DEVICE __CONSTEXPR__ inline size_type max_size() const { return std::numeric_limits<size_type>::max(); }
	//HOST DEVICE inline size_type number_rows() const { return numberRows; }
	//HOST DEVICE inline size_type number_columns() const { return numberColumns; }
	HOST DEVICE inline size_type number_rows() const { return numberRows; }
	HOST DEVICE inline size_type number_columns() const { return numberColumns; }

	///
	/// \brief Assigns a given value to all elements in the container.
	///
	/// \param value the value to assign to the elements
	///
	HOST DEVICE void fill( const value_type& value ) {
		#ifdef __CUDA_ARCH__
		for( iterator iter = begin(); iter != end(); ++iter ) *iter = value;
		#else
		std::vector<value_type> v( number_columns(), value );
		for( size_type i = 0; i < number_rows(); ++i )
			CUDA_CALL( cudaMemcpy<value_type>( allocator.address( data(), i, 0, pitch ), &v.front(), number_columns(), cudaMemcpyHostToDevice ) );
		#endif
	}

	HOST matrix<value_type> excise( const size_type offsetRow, const size_type offsetColumn, const size_type sizeRow, const size_type sizeColumn ) {
		if( offsetRow >= number_rows() ) throw std::out_of_range( "ecuda::matrix::excise offsetRow argument out of range" );
		if( offsetColumn >= number_columns() ) throw std::out_of_range( "ecuda::matrix::excise offsetColumn argument out of range" );
		if( (offsetRow+sizeRow) > number_rows() ) throw std::out_of_range( "ecuda::matrix::excise sizeRow argument out of range" );
		if( (offsetColumn+sizeColumn) > number_columns() ) throw std::out_of_range( "ecuda::matrix::excise sizeColumn argument out of range" );
		ecuda::matrix<value_type,allocator_type> matrix( sizeRow, sizeColumn, value_type(), get_allocator() );
		for( size_type i = 0; i < sizeRow; ++i ) {
			CUDA_CALL( cudaMemcpy<value_type>(
						matrix.allocator.address( matrix.data(), i, 0, matrix.pitch ),
						allocator.address( data(), offsetRow+i, offsetColumn, pitch ),
						sizeColumn,
						cudaMemcpyDeviceToDevice
			) );
		}
		return matrix;
	}

	// critical function used to bridge host->device code
	HOST DEVICE matrix<T>& operator=( const matrix<T>& other ) {
		numberRows = other.numberRows;
		numberColumns = other.numberColumns;
		pitch = other.pitch;
		deviceMemory = other.deviceMemory;
		return *this;
	}

	#if HAVE_ESTD_LIBRARY > 0
	template<typename U,typename V>
	HOST matrix<T,Alloc>& operator>>( estd::matrix<T,U,V>& dest ) {
		dest.resize( static_cast<U>(numberRows), static_cast<V>(numberColumns) );
		CUDA_CALL( cudaMemcpy2D<T>( dest.data(), numberColumns*sizeof(T), data(), pitch, numberColumns, numberRows, cudaMemcpyDeviceToHost ) );
		return *this;
	}
	#endif

	template<class OtherAlloc>
	HOST matrix<T,Alloc>& operator>>( std::vector<T,OtherAlloc>& other ) {
		other.resize( size() );
		CUDA_CALL( cudaMemcpy2D<T>( &other.front(), numberColumns*sizeof(T), data(), pitch, numberColumns, numberRows, cudaMemcpyDeviceToHost ) );
		return *this;
	}

	HOST void resize( const size_type numberRows, const size_type numberColumns ) {
		if( number_rows() == numberRows and number_columns() == numberColumns ) return; // no resize needed
		// allocate memory
		this->numberRows = numberRows;
		this->numberColumns = numberColumns;
		deviceMemory = device_ptr<T>( DevicePitchAllocator<T>().allocate( numberColumns, numberRows, pitch ) );
	}

	#if HAVE_ESTD_LIBRARY > 0
	template<typename U,typename V>
	HOST matrix<T,Alloc>& operator<<( const estd::matrix<T,U,V>& src ) {
		resize( src.row_size(), src.column_size() );
		CUDA_CALL( cudaMemcpy2D<T>( data(), pitch, src.data(), numberColumns*sizeof(T), numberColumns, numberRows, cudaMemcpyHostToDevice ) );
		return *this;
	}
	#endif

	template<class OtherAlloc>
	HOST matrix<T,Alloc>& operator<<( std::vector<T,OtherAlloc>& other ) {
		CUDA_CALL( cudaMemcpy2D<T>( data(), pitch, &other.front(), numberColumns*sizeof(T), numberColumns, numberRows, cudaMemcpyHostToDevice ) );
		return *this;
	}

};

template<typename T,class Alloc>
HOST matrix<T,Alloc>::matrix( const size_type numberRows, const size_type numberColumns, const_reference value, const Alloc& allocator ) : numberRows(numberRows), numberColumns(numberColumns), pitch(0), allocator(allocator) {
	if( numberRows and numberColumns ) {
		deviceMemory = device_ptr<T>( get_allocator().allocate( numberColumns, numberRows, pitch ) );
		std::vector<T> v( numberRows*numberColumns, value );
		CUDA_CALL( cudaMemcpy2D<T>( deviceMemory.get(), pitch, &v.front(), numberColumns*sizeof(T), numberColumns, numberRows, cudaMemcpyHostToDevice ) );
	}
}

template<typename T,class Alloc>
HOST DEVICE matrix<T,Alloc>::matrix( const matrix<T>& src ) : numberRows(src.numberRows), numberColumns(src.numberColumns), pitch(src.pitch), deviceMemory(src.deviceMemory), allocator(src.allocator) {}

#if HAVE_ESTD_LIBRARY > 0
template<typename T,class Alloc>
template<typename U,typename V>
HOST matrix<T,Alloc>::matrix( const estd::matrix<T,U,V>& src, const Alloc& allocator ) : numberRows(static_cast<size_type>(src.row_size())), numberColumns(static_cast<size_type>(src.column_size())), pitch(0), allocator(allocator) {
	if( numberRows and numberColumns ) {
		deviceMemory = device_ptr<T>( get_allocator().allocate( numberColumns, numberRows, pitch ) );
		CUDA_CALL( cudaMemcpy2D<T>( deviceMemory.get(), pitch, src.data(), numberColumns*sizeof(T), numberColumns, numberRows, cudaMemcpyHostToDevice ) );
	}
}
#endif

template<typename T,class Alloc>
template<class RandomAccessIterator>
HOST void matrix<T,Alloc>::assign( RandomAccessIterator begin, RandomAccessIterator end ) {
	std::size_t n = end-begin;
	if( n > size() ) n = size();
	RandomAccessIterator current = begin;
	for( std::size_t i = 0; i < n; i += number_columns(), current += number_columns() ) {
		std::size_t len = number_columns();
		if( i+len > size() ) len = size()-i;
		std::vector<T> row( current, current+len );
		CUDA_CALL( cudaMemcpy<T>( allocator.address( deviceMemory.get(), i/number_columns(), 0, pitch ), &row[0], len, cudaMemcpyHostToDevice ) );
	}
}

} // namespace ecuda

#endif

