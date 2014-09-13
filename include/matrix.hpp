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

	typedef ecuda::OffsettingContainer< matrix<T> > row_type;
	typedef ecuda::OffsettingContainer< matrix<T> > column_type;
	typedef const ecuda::OffsettingContainer< const matrix<T> > const_row_type;
	typedef const ecuda::OffsettingContainer< const matrix<T> > const_column_type;

	typedef row_type subscript_type;
	typedef const_row_type const_subscript_type;

private:
public:
	size_type numberRows;
	size_type numberColumns;
	size_type pitch;
	device_ptr<T> deviceMemory; // video card memory

public:
	__host__ matrix( const size_type numberRows=0, const size_type numberColumns=0, const_reference value = T() ) : numberRows(numberRows), numberColumns(numberColumns), pitch(0) {
		if( numberRows and numberColumns ) {
			CUDA_CALL( cudaMallocPitch( deviceMemory.alloc_ptr(), &pitch, numberColumns*sizeof(T), numberRows ) );
			std::vector<T> v( numberRows*numberColumns, value );
			CUDA_CALL( cudaMemcpy2D( deviceMemory.get(), pitch, &v[0], numberColumns*sizeof(T), numberColumns*sizeof(T), numberRows, cudaMemcpyHostToDevice ) );
		}
	}
	__host__ matrix( const matrix<T>& src ) : numberRows(src.numberRows), numberColumns(src.numberColumns), pitch(src.pitch), deviceMemory(src.deviceMemory) {}
	//matrix( const T* sourcePtr, const size_type numberRows, const size_type numberColumns ) : numberRows(numberRows), numberColumns(numberColumns), pitch(0) {
	//	if( numberRows and numberColumns ) {
	//		CUDA_CALL( cudaMallocPitch( deviceMemory.alloc_ptr(), &pitch, numberColumns*sizeof(T), numberRows ) );
	//		CUDA_CALL( cudaMemcpy2D( deviceMemory.get(), pitch, sourcePtr, numberColumns*sizeof(T), numberColumns*sizeof(T), numberRows, cudaMemcpyHostToDevice ) );
	//	}
	//}
	template<typename U,typename V>
	__host__ matrix( const estd::matrix<T,U,V>& src ) : numberRows(static_cast<size_type>(src.row_size())), numberColumns(static_cast<size_type>(src.column_size())), pitch(0) {
		if( numberRows and numberColumns ) {
			CUDA_CALL( cudaMallocPitch( deviceMemory.alloc_ptr(), &pitch, numberColumns*sizeof(T), numberRows ) );
			CUDA_CALL( cudaMemcpy2D( deviceMemory.get(), pitch, &src[0][0], numberColumns*sizeof(T), numberColumns*sizeof(T), numberRows, cudaMemcpyHostToDevice ) );
		}
	}

	__host__ __device__ virtual ~matrix() {}

	__device__ reference at( size_type rowIndex, size_type columnIndex ) { return *(deviceMemory.get()+(rowIndex*pitch/sizeof(T)+columnIndex)); }
	__device__ reference at( size_type index ) { return at( index/numberColumns, index % numberColumns ); }

	__device__ const_reference at( size_type rowIndex, size_type columnIndex ) const { return *(deviceMemory.get()+(rowIndex*pitch/sizeof(T)+columnIndex)); }
	__device__ const_reference at( size_type index ) const { return at( index/numberColumns, index % numberColumns ); }

	__host__ __device__ size_type size() const { return numberRows*numberColumns; }
	__host__ __device__ size_type row_size() const { return numberRows; }
	__host__ __device__ size_type column_size() const { return numberColumns; }
	__host__ __device__ size_type get_pitch() const { return pitch; }

	__device__ row_type get_row( const size_type rowIndex ) { return row_type( *this, row_size(), rowIndex*pitch/sizeof(T) ); }
	__device__ column_type get_column( const size_type columnIndex ) { return column_type( *this, column_size(), columnIndex, pitch/sizeof(T) ); }
	__device__ const_row_type get_row( const size_type rowIndex ) const { return const_row_type( *this, row_size(), rowIndex*pitch/sizeof(T) ); }
	__device__ const_column_type get_column( const size_type columnIndex ) const { return const_column_type( *this, column_size(), columnIndex, pitch/sizeof(T) ); }

	__device__ subscript_type operator[]( const size_type rowIndex ) { return get_row(rowIndex); }
	__device__ const_subscript_type operator[]( const size_type rowIndex ) const { return get_row(rowIndex); }

	__device__ matrix<T>& operator=( const matrix<T>& other ) {
		numberRows = other.numberRows;
		numberColumns = other.numberColumns;
		pitch = other.pitch;
		deviceMemory = other.deviceMemory;
		return *this;
	}

	template<typename U,typename V>
	__host__ matrix<T>& operator>>( estd::matrix<T,U,V>& other ) {
		other.resize( static_cast<U>(numberRows), static_cast<V>(numberColumns) );
		CUDA_CALL( cudaMemcpy2D( other.data(), numberColumns*sizeof(T), deviceMemory.get(), pitch, numberColumns*sizeof(T), numberRows, cudaMemcpyDeviceToHost ) );
		return *this;
	}

};


} // namespace ecuda

#endif
