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
#include <estd/containers.hpp>
#include <estd/matrix.hpp>
#include "global.hpp"
#include "memory.hpp"

namespace ecuda {

template<typename T>
class matrix {

public:
	typedef T value_type;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef value_type* pointer;
	typedef const value_type* const_pointer;

	typedef estd::OffsettingContainer< matrix<T> > row_type;
	typedef estd::OffsettingContainer< matrix<T> > column_type;
	typedef const estd::OffsettingContainer< const matrix<T> > const_row_type;
	typedef const estd::OffsettingContainer< const matrix<T> > const_column_type;

	typedef row_type subscript_type;
	typedef const_row_type const_subscript_type;

private:
	size_type numberRows;
	size_type numberColumns;
	size_type pitch;
	unique_ptr<T[]> deviceMemory; // video card memory

public:
	matrix( const size_type numberRows=0, const size_type numberColumns=0, const_reference value = T() ) : numberRows(numberRows), numberColumns(numberColumns), pitch(pitch) {
		if( numberRows and numberColumns ) {
			T* ptr = NULL;
			CUDA_CALL( cudaMallocPitch( reinterpret_cast<void**>(&ptr), &pitch, numberColumns*sizeof(T), numberRows*sizeof(T) ) );
			std::vector<T> v( numberRows*numberColumns, value );
			CUDA_CALL( cudaMemcpy2D( ptr, pitch, &v[0], numberColumns*sizeof(T), numberColumns*sizeof(T), numberRows*sizeof(T), cudaMemcpyHostToDevice ) );
			deviceMemory = unique_ptr<T[]>( ptr );
		}
	}
	matrix( const matrix<T>& src ) : numberRows(src.numberRows), numberColumns(src.numberColumns), pitch(0) {
		if( numberRows and numberColumns ) {
			T* ptr = NULL;
			CUDA_CALL( cudaMallocPitch( reinterpret_cast<void**>(&ptr), &pitch, numberColumns*sizeof(T), numberRows*sizeof(T) ) );
			CUDA_CALL( cudaMemcpy2D( ptr, pitch, src.deviceMemory.get(), src.pitch, numberRows*sizeof(T), cudaMemcpyDeviceToDevice ) );
			deviceMemory = unique_ptr<T[]>( ptr );
		}
	}
	matrix( const T* sourcePtr, const size_type numberRows, const size_type numberColumns ) : numberRows(numberRows), numberColumns(numberColumns), pitch(0) {
		if( numberRows and numberColumns ) {
			T* ptr = NULL;
			CUDA_CALL( cudaMallocPitch( reinterpret_cast<void**>(&ptr), &pitch, numberColumns*sizeof(T), numberRows*sizeof(T) ) );
			CUDA_CALL( cudaMemcpy2D( ptr, pitch, sourcePtr, numberColumns*sizeof(T), numberColumns*sizeof(T), numberRows*sizeof(T), cudaMemcpyHostToDevice ) );
			deviceMemory = unique_ptr<T[]>( ptr );
		}
	}
	template<typename X,typename Y>
	matrix( const estd::matrix<T,X,Y>& src ) : numberRows(src.row_size()), numberColumns(src.column_size()), pitch(0) {
		if( numberRows and numberColumns ) {
			T* ptr = NULL;
			CUDA_CALL( cudaMallocPitch( reinterpret_cast<void**>(&ptr), &pitch, numberColumns*sizeof(T), numberRows*sizeof(T) ) );
			CUDA_CALL( cudaMemcpy2D( ptr, pitch, &src[0][0], numberColumns*sizeof(T), numberColumns*sizeof(T), numberRows*sizeof(T), cudaMemcpyHostToDevice ) );
			deviceMemory = unique_ptr<T[]>( ptr );
		}
	}

	virtual ~matrix() {}

	__device__ reference at( size_type rowIndex, size_type columnIndex ) { return deviceMemory[rowIndex*pitch/sizeof(T)+columnIndex]; }
	__device__ reference at( size_type index ) { return at( index/numberColumns, index % numberColumns ); }

	__device__ const_reference at( size_type rowIndex, size_type columnIndex ) const { return deviceMemory[rowIndex*pitch/sizeof(T)+columnIndex]; }
	__device__ const_reference at( size_type index ) const { return at( index/numberColumns, index % numberColumns ); }

	__host__ __device__ size_type size() const { return numberRows*numberColumns; }
	__host__ __device__ size_type row_size() const { return numberRows; }
	__host__ __device__ size_type column_size() const { return numberColumns; }
	__host__ __device__ size_type get_pitch() const { return pitch; }

	__host__ __device__ row_type get_row( const size_type rowIndex ) { return row_type( *this, row_size(), rowIndex*pitch/sizeof(T) ); }
	__host__ __device__ column_type get_column( const size_type columnIndex ) { return column_type( *this, column_size(), columnIndex, pitch/sizeof(T) ); }
	__host__ __device__ const_row_type get_row( const size_type rowIndex ) const { return const_row_type( *this, row_size(), rowIndex*pitch/sizeof(T) ); }
	__host__ __device__ const_column_type get_column( const size_type columnIndex ) const { return const_column_type( *this, column_size(), columnIndex, pitch/sizeof(T) ); }

	__host__ __device__ subscript_type operator[]( const size_type rowIndex ) { return get_row(rowIndex); }
	__host__ __device__ const_subscript_type operator[]( const size_type rowIndex ) const { return get_row(rowIndex); }

};


} // namespace ecuda

#endif
