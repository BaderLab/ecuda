/*
Copyright (c) 2014, Scott Zuyderduyn
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
	typedef const ecuda::OffsettingContainer< const matrix<T>, size_type, const_pointer > const_row_type; //!< matrix const row container type
	typedef const ecuda::OffsettingContainer< const matrix<T>, size_type, const_pointer > const_column_type; //!< matrix const column container type

private:
	// REMEMBER: numberRows, numberColumns, and pitch altered on device memory won't be
	//           reflected on the host object. Don't allow the device to perform any operations that
	//           change their value.
	size_type numberRows; //!< number of matrix rows
	size_type numberColumns; //!< number of matrix columns
	size_type pitch; //!< the padded width of the 2D memory allocation in bytes
	device_ptr<T> deviceMemory; //!< smart pointer to video card memory

public:
	HOST matrix( const size_type numberRows=0, const size_type numberColumns=0, const_reference value = T() );
	HOST DEVICE matrix( const matrix<T>& src );
	template<typename U,typename V>
	HOST matrix( const estd::matrix<T,U,V>& src );
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

	HOST DEVICE inline row_type get_row( const size_type rowIndex ) { return row_type( *this, column_size(), rowIndex*pitch/sizeof(T) ); }
	HOST DEVICE inline column_type get_column( const size_type columnIndex ) { return column_type( *this, row_size(), columnIndex, pitch/sizeof(T) ); }
	HOST DEVICE inline const_row_type get_row( const size_type rowIndex ) const { return const_row_type( *this, column_size(), rowIndex*pitch/sizeof(T) ); }
	HOST DEVICE inline const_column_type get_column( const size_type columnIndex ) const { return const_column_type( *this, row_size(), columnIndex, pitch/sizeof(T) ); }

	HOST DEVICE inline row_type operator[]( const size_type rowIndex ) { return get_row(rowIndex); }
	HOST DEVICE inline const_row_type operator[]( const size_type rowIndex ) const { return get_row(rowIndex); }

	// critical function used to bridge host->device code
	HOST DEVICE matrix<T>& operator=( const matrix<T>& other ) {
		numberRows = other.numberRows;
		numberColumns = other.numberColumns;
		pitch = other.pitch;
		deviceMemory = other.deviceMemory;
		return *this;
	}

	template<typename U,typename V>
	HOST matrix<T>& operator>>( estd::matrix<T,U,V>& dest ) {
		dest.resize( static_cast<U>(numberRows), static_cast<V>(numberColumns) );
		CUDA_CALL( cudaMemcpy2D( dest.data(), numberColumns*sizeof(T), deviceMemory.get(), pitch, numberColumns*sizeof(T), numberRows, cudaMemcpyDeviceToHost ) );
		return *this;
	}

	template<typename Alloc>
	HOST matrix<T>& operator>>( std::vector<T,Alloc>& other ) {
		other.resize( size() );
		CUDA_CALL( cudaMemcpy2D( &other[0], numberColumns*sizeof(T), deviceMemory.get(), pitch, numberColumns*sizeof(T), numberRows, cudaMemcpyDeviceToHost ) );
		return *this;
	}

	HOST void resize( const size_type numberRows, const size_type numberColumns ) {
		if( row_size() == numberRows and column_size() == numberColumns ) return; // no resize needed
		// allocate memory
		this->numberRows = numberRows;
		this->numberColumns = numberColumns;
		deviceMemory = device_ptr<T>();
		CUDA_CALL( cudaMallocPitch( deviceMemory.alloc_ptr(), &pitch, numberColumns*sizeof(T), numberRows ) );
	}

	template<typename U,typename V>
	HOST matrix<T>& operator<<( const estd::matrix<T,U,V>& src ) {
		resize( src.row_size(), src.column_size() );
		CUDA_CALL( cudaMemcpy2D( deviceMemory.get(), pitch, src.data(), numberColumns*sizeof(T), numberColumns*sizeof(T), numberRows, cudaMemcpyHostToDevice ) );
		return *this;
	}

	template<typename Alloc>
	HOST matrix<T>& operator<<( std::vector<T,Alloc>& other ) {
		CUDA_CALL( cudaMemcpy2D( data(), pitch, &other[0], numberColumns*sizeof(T), numberColumns*sizeof(T), numberRows, cudaMemcpyHostToDevice ) );
		return *this;
	}

};

template<typename T>
HOST matrix<T>::matrix( const size_type numberRows, const size_type numberColumns, const_reference value ) : numberRows(numberRows), numberColumns(numberColumns), pitch(0) {
	if( numberRows and numberColumns ) {
		CUDA_CALL( cudaMallocPitch( deviceMemory.alloc_ptr(), &pitch, numberColumns*sizeof(T), numberRows ) );
		// cudaMemset2D method not general since sizeof(T) > int
		//CUDA_CALL( cudaMemset2D( deviceMemory.get(), pitch, static_cast<int>(value), numberColumns*sizeof(T), numberRows ) );
		std::vector<T> v( numberRows*numberColumns, value );
		CUDA_CALL( cudaMemcpy2D( deviceMemory.get(), pitch, &v[0], numberColumns*sizeof(T), numberColumns*sizeof(T), numberRows, cudaMemcpyHostToDevice ) );
	}
}

template<typename T>
HOST DEVICE matrix<T>::matrix( const matrix<T>& src ) : numberRows(src.numberRows), numberColumns(src.numberColumns), pitch(src.pitch), deviceMemory(src.deviceMemory) {}

template<typename T>
template<typename U,typename V>
HOST matrix<T>::matrix( const estd::matrix<T,U,V>& src ) : numberRows(static_cast<size_type>(src.row_size())), numberColumns(static_cast<size_type>(src.column_size())), pitch(0) {
	if( numberRows and numberColumns ) {
		CUDA_CALL( cudaMallocPitch( deviceMemory.alloc_ptr(), &pitch, numberColumns*sizeof(T), numberRows ) );
		CUDA_CALL( cudaMemcpy2D( deviceMemory.get(), pitch, src.data(), numberColumns*sizeof(T), numberColumns*sizeof(T), numberRows, cudaMemcpyHostToDevice ) );
	}
}


} // namespace ecuda

#endif
