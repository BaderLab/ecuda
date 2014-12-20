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
// cube.hpp
// An STL-like structure that resides in video memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------
#pragma once
#ifndef ECUDA_CUBE_HPP
#define ECUDA_CUBE_HPP

#include <limits>
#include <vector>
#include <estd/cube.hpp>
#include "global.hpp"
#include "allocators.hpp"
#include "containers.hpp"
//#include "iterators.hpp"
#include "matrix.hpp"
#include "memory.hpp"

namespace ecuda {

template< typename T, class Alloc=DevicePitchAllocator<T> >
class cube {

public:
	typedef T value_type;
	typedef Alloc allocator_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T& reference;
	typedef const T& const_reference;
	typedef std::ptrdiff_t difference_type;
	typedef std::size_t size_type;

	typedef ecuda::OffsettingContainer< cube<T> > xy_type;
	typedef ecuda::OffsettingContainer< cube<T> > xz_type;
	typedef ecuda::OffsettingContainer< cube<T> > yz_type;
	typedef const ecuda::OffsettingContainer< const cube<T>, size_type, const_pointer > const_xy_type;
	typedef const ecuda::OffsettingContainer< const cube<T>, size_type, const_pointer > const_xz_type;
	typedef const ecuda::OffsettingContainer< const cube<T>, size_type, const_pointer > const_yz_type;

	typedef ecuda::CubeSliceContainer< cube<T>, size_type, pointer > matrix_type;
	typedef const ecuda::CubeSliceContainer< const cube<T>, size_type, const_pointer > const_matrix_type;

private:
	// REMEMBER: numberRows, numberColumns, numberDepths and pitch altered on device memory won't be
	//           reflected on the host object. Don't allow the device to perform any operations that
	//           change their value.
	size_type numberRows;
	size_type numberColumns;
	size_type numberDepths;
	size_type pitch;
	device_ptr<T> deviceMemory;
	allocator_type allocator;
	//unique_ptr< matrix<T>[] > matrices;

public:
	HOST cube( const size_type numberRows=0, const size_type numberColumns=0, const size_type numberDepths=0, const T& value = T() ) : numberRows(numberRows), numberColumns(numberColumns), numberDepths(numberDepths) {
		if( numberRows and numberColumns and numberDepths ) {
			deviceMemory = device_ptr<T>( allocator.allocate( numberColumns*numberDepths, numberRows, pitch ) );
			//CUDA_CALL( cudaMallocPitch( deviceMemory.alloc_ptr(), &pitch, numberColumns*numberDepths*sizeof(T), numberRows ) );
			std::vector<T> v( numberColumns*numberDepths, value );
			for( size_t i = 0; i < numberRows; ++i )
				CUDA_CALL( cudaMemcpy<T>(
					allocator.address( deviceMemory.get(), i, 0, pitch ), // dst
					&v[0], // src
					numberColumns*numberDepths, // count
					cudaMemcpyHostToDevice // kind
				) );
				//CUDA_CALL( cudaMemcpy2D( deviceMemory.get()+(i*pitch/sizeof(T)), numberColumns*numberDepths*sizeof(T), &v[0], numberColumns*numberDepths*sizeof(T), numberColumns*numberDepths*sizeof(T), 1, cudaMemcpyHostToDevice ) );
		}
	}
	HOST DEVICE cube( const cube<T>& src ) : numberRows(src.numberRows), numberColumns(src.numberColumns), numberDepths(src.numberDepths), pitch(src.pitch), deviceMemory(src.deviceMemory) {}
	template<typename U,typename V,typename W>
	HOST cube( const estd::cube<T,U,V,W>& src ) : numberRows(src.row_size()), numberColumns(src.column_size()), numberDepths(src.depth_size()) {
		if( numberRows and numberColumns and numberDepths ) {
			deviceMemory = device_ptr<T>( allocator.allocate( numberColumns*numberDepths, numberRows, pitch ) );
			//CUDA_CALL( cudaMallocPitch( deviceMemory.alloc_ptr(), &pitch, numberColumns*numberDepths*sizeof(T), numberRows ) );
			for( size_t i = 0; i < numberRows; ++i ) {
				std::vector<T> v; v.reserve( numberColumns*numberDepths );
				for( size_t j = 0; j < numberColumns; ++j )
					for( size_t k = 0; k < numberDepths; ++k )
						v.push_back( src[i][j][k] );
				CUDA_CALL( cudaMemcpy<T>(
					allocator.address( deviceMemory.get(), i, 0, pitch ), // dst
					&v[0], // src
					numberColumns*numberDepths, // count
					cudaMemcpyHostToDevice // kind
				) );
				//CUDA_CALL( cudaMemcpy2D( deviceMemory.get()+(i*pitch/sizeof(T)), numberColumns*numberDepths*sizeof(T), &v[0], numberColumns*numberDepths*sizeof(T), numberColumns*numberDepths*sizeof(T), 1, cudaMemcpyHostToDevice ) );
			}
		}
	}

	// capacity:
	HOST DEVICE inline size_type row_size() const __NOEXCEPT__ { return numberRows; }
	HOST DEVICE inline size_type column_size() const __NOEXCEPT__ { return numberColumns; }
	HOST DEVICE inline size_type depth_size() const __NOEXCEPT__ { return numberDepths; }
	HOST DEVICE inline size_type get_pitch() const { return pitch; }
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return row_size()*column_size()*depth_size(); }
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return !size(); }

	// element access:
	DEVICE inline reference at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) { return *allocator.address( deviceMemory.get(), rowIndex, columnIndex*numberDepths+depthIndex, pitch ); }
	//DEVICE inline reference at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) { return *(deviceMemory.get()+(rowIndex*pitch/sizeof(T)+columnIndex*numberDepths+depthIndex)); }
	DEVICE inline reference at( const size_type index ) { return at( index/(numberColumns*numberDepths), (index % (numberColumns*numberDepths))/numberDepths, (index % (numberColumns*numberDepths)) % numberDepths ); }
	DEVICE inline const_reference at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) const { return *allocator.address( deviceMemory.get(), rowIndex, columnIndex*numberDepths+depthIndex, pitch ); }
	//DEVICE inline const_reference at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) const { return *(deviceMemory.get()+(rowIndex*pitch/sizeof(T)+columnIndex*numberDepths+depthIndex)); }
	DEVICE inline const_reference at( const size_type index ) const { return at( index/(numberColumns*numberDepths), (index % (numberColumns*numberDepths))/numberDepths, (index % (numberColumns*numberDepths)) % numberDepths ); }
	HOST DEVICE inline pointer data() __NOEXCEPT__ { return deviceMemory.get(); }
	HOST DEVICE inline const_pointer data() const __NOEXCEPT__ { return deviceMemory.get(); }

	HOST DEVICE inline matrix_type get_row( const size_type rowIndex ) { return matrix_type( *this, rowIndex ); }
	HOST DEVICE inline const_matrix_type get_row( const size_type rowIndex ) const { return const_matrix_type( *this, rowIndex ); }
	HOST DEVICE inline matrix_type operator[]( const size_type rowIndex ) { return get_row(rowIndex); }
	HOST DEVICE inline const_matrix_type operator[]( const size_type rowIndex ) const { return get_row(rowIndex); }

	//xy_type get_xy( const RowIndexType x, const ColumnIndexType y ) { return contents[x].get_row(y); }
	//xz_type get_xz( const RowIndexType x, const DepthIndexType z ) { return contents[x].get_column(z); }
	//yz_type get_yz( const ColumnIndexType y, const DepthIndexType z ) {	return OffsettingContainer< cube<CellType,RowIndexType,ColumnIndexType,DepthIndexType> >( *this, row_size(), y*depth_size()+z, column_size()*depth_size() ); }
	//const_xy_type get_xy( const RowIndexType x, const ColumnIndexType y ) const { return contents[x].get_row(y); }
	//const_xz_type get_xz( const RowIndexType x, const DepthIndexType z ) const { return contents[x].get_column(z); }
	//const_yz_type get_yz( const ColumnIndexType y, const DepthIndexType z ) const { return OffsettingContainer< const cube<CellType,RowIndexType,ColumnIndexType,DepthIndexType> >( *this, row_size(), y*depth_size()+z, column_size()*depth_size() ); }

	HOST inline allocator_type get_allocator() const { return allocator; }

	template<typename U,typename V,typename W>
	HOST cube<T,Alloc>& operator>>( estd::cube<T,U,V,W>& dest ) {
		//TODO: this needs to be re-implemented, it won't work as currently written
		dest.resize( static_cast<U>(numberRows), static_cast<V>(numberColumns), static_cast<W>(numberDepths) );
		CUDA_CALL( cudaMemcpy2D<T>(
			&dest[0][0][0], // dest
			numberColumns*numberDepths*sizeof(T), // dpitch
			deviceMemory.get(), // src
			pitch, // spitch
			numberColumns*numberDepths, // width
			numberRows, // height
			cudaMemcpyDeviceToHost // kind
		) );
		//CUDA_CALL( cudaMemcpy2D( &dest[0][0][0], numberColumns*numberDepths*sizeof(T), deviceMemory.get(), pitch, numberColumns*numberDepths*sizeof(T), numberRows, cudaMemcpyDeviceToHost ) );
		//for( size_type i = 0; i < numberRows; ++i ) operator[](i) >> dest[i];
		return *this;
	}

	HOST void resize( const size_type numberRows, const size_type numberColumns, const size_type numberDepths ) {
		if( row_size() == numberRows and column_size() == numberColumns and depth_size() == numberDepths ) return; // no resize needed
		// allocate memory
		this->numberRows = numberRows;
		this->numberColumns = numberColumns;
		this->numberDepths = numberDepths;
		deviceMemory = device_ptr<T>( DevicePitchAllocator<T>().allocate( numberColumns*numberDepths, numberRows, pitch ) );
	}

	template<typename U,typename V,typename W>
	HOST cube<T,Alloc>& operator<<( const estd::cube<T,U,V,W>& src ) {
		resize( src.row_size(), src.column_size(), src.depth_size() );
		CUDA_CALL( cudaMemcpy2D<T>( data(), pitch, src.data(), numberColumns*numberDepths*sizeof(T), numberColumns*numberDepths, numberRows, cudaMemcpyHostToDevice ) );
		return *this;
	}


};

} // namespace ecuda

#endif

