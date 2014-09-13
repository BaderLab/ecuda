//----------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// matrix.hpp
// An STL-like structure that resides in video memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MATRIX_HPP
#define ECUDA_MATRIX_HPP

#include <cstddef>
#include <vector>
#include <estd/matrix.hpp>
#include "global.hpp"
#include "containers.hpp"
#include "memory.hpp"

namespace ecuda {

///
/// A video memory-bound matrix structure.
///
template<typename T>
class matrix {

public:
	typedef T value_type; //!< cell data type
	typedef std::size_t size_type; //!< index data type
	typedef std::ptrdiff_t difference_type; //!<
	typedef value_type& reference; //!< cell reference type
	typedef const value_type& const_reference; //!< cell const reference type
	typedef value_type* pointer; //!< cell pointer type
	typedef const value_type* const_pointer; //!< cell const pointer type

	typedef ecuda::OffsettingContainer< matrix<T> > row_type; //!< matrix row container type
	typedef ecuda::OffsettingContainer< matrix<T> > column_type; //!< matrix column container type
	typedef const ecuda::OffsettingContainer< const matrix<T> > const_row_type; //!< matrix const row container type
	typedef const ecuda::OffsettingContainer< const matrix<T> > const_column_type; //!< matrix const column container type

private:
	size_type numberRows; //!< number of matrix rows
	size_type numberColumns; //!< number of matrix columns
	size_type pitch; //!< the padded width of the 2D memory allocation in bytes
	device_ptr<T> deviceMemory; //!< smart pointer to video card memory

public:
	HOST matrix( const size_type numberRows=0, const size_type numberColumns=0, const_reference value = T() ) : numberRows(numberRows), numberColumns(numberColumns), pitch(0) {
		if( numberRows and numberColumns ) {
			CUDA_CALL( cudaMallocPitch( deviceMemory.alloc_ptr(), &pitch, numberColumns*sizeof(T), numberRows ) );
			std::vector<T> v( numberRows*numberColumns, value );
			CUDA_CALL( cudaMemcpy2D( deviceMemory.get(), pitch, &v[0], numberColumns*sizeof(T), numberColumns*sizeof(T), numberRows, cudaMemcpyHostToDevice ) );
		}
	}
	HOST matrix( const matrix<T>& src ) : numberRows(src.numberRows), numberColumns(src.numberColumns), pitch(src.pitch), deviceMemory(src.deviceMemory) {}
	template<typename U,typename V>
	HOST matrix( const estd::matrix<T,U,V>& src ) : numberRows(static_cast<size_type>(src.row_size())), numberColumns(static_cast<size_type>(src.column_size())), pitch(0) {
		if( numberRows and numberColumns ) {
			CUDA_CALL( cudaMallocPitch( deviceMemory.alloc_ptr(), &pitch, numberColumns*sizeof(T), numberRows ) );
			CUDA_CALL( cudaMemcpy2D( deviceMemory.get(), pitch, src.data(), numberColumns*sizeof(T), numberColumns*sizeof(T), numberRows, cudaMemcpyHostToDevice ) );
		}
	}

	HOST DEVICE virtual ~matrix() {}

	DEVICE inline reference at( size_type rowIndex, size_type columnIndex ) { return *(deviceMemory.get()+(rowIndex*pitch/sizeof(T)+columnIndex)); }
	DEVICE inline reference at( size_type index ) { return at( index/(pitch/sizeof(T)), index % (pitch/sizeof(T)) ); }

	DEVICE inline const_reference at( size_type rowIndex, size_type columnIndex ) const { return *(deviceMemory.get()+(rowIndex*pitch/sizeof(T)+columnIndex)); }
	DEVICE inline const_reference at( size_type index ) const { return at( index/(pitch/sizeof(T)), index % (pitch/sizeof(T)) ); }

	HOST DEVICE inline size_type size() const { return numberRows*numberColumns; }
	HOST DEVICE inline size_type row_size() const { return numberRows; }
	HOST DEVICE inline size_type column_size() const { return numberColumns; }
	HOST DEVICE inline size_type get_pitch() const { return pitch; }
	HOST DEVICE inline T* data() { return deviceMemory.get(); }
	HOST DEVICE inline const T* data() const { return deviceMemory.get(); }

	DEVICE inline row_type get_row( const size_type rowIndex ) { return row_type( *this, column_size(), rowIndex*pitch/sizeof(T) ); }
	DEVICE inline column_type get_column( const size_type columnIndex ) { return column_type( *this, row_size(), columnIndex, pitch/sizeof(T) ); }
	DEVICE inline const_row_type get_row( const size_type rowIndex ) const { return const_row_type( *this, column_size(), rowIndex*pitch/sizeof(T) ); }
	DEVICE inline const_column_type get_column( const size_type columnIndex ) const { return const_column_type( *this, row_size(), columnIndex, pitch/sizeof(T) ); }

	DEVICE inline row_type operator[]( const size_type rowIndex ) { return get_row(rowIndex); }
	DEVICE inline const_row_type operator[]( const size_type rowIndex ) const { return get_row(rowIndex); }

	DEVICE matrix<T>& operator=( const matrix<T>& other ) {
		numberRows = other.numberRows;
		numberColumns = other.numberColumns;
		pitch = other.pitch;
		deviceMemory = other.deviceMemory;
		return *this;
	}

	template<typename U,typename V>
	HOST matrix<T>& operator>>( estd::matrix<T,U,V>& other ) {
		other.resize( static_cast<U>(numberRows), static_cast<V>(numberColumns) );
		CUDA_CALL( cudaMemcpy2D( other.data(), numberColumns*sizeof(T), deviceMemory.get(), pitch, numberColumns*sizeof(T), numberRows, cudaMemcpyDeviceToHost ) );
		return *this;
	}

};


} // namespace ecuda

#endif
