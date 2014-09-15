//----------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
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
#include "containers.hpp"
//#include "iterators.hpp"
#include "matrix.hpp"
#include "memory.hpp"

namespace ecuda {

template<typename T>
class cube {

public:
	typedef T value_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T& reference;
	typedef const T& const_reference;
	typedef std::ptrdiff_t difference_type;
	typedef std::size_t size_type;

	typedef ecuda::OffsettingContainer< cube<T> > xy_type;
	typedef ecuda::OffsettingContainer< cube<T> > xz_type;
	typedef ecuda::OffsettingContainer< cube<T> > yz_type;
	typedef const ecuda::OffsettingContainer< const cube<T> > const_xy_type;
	typedef const ecuda::OffsettingContainer< const cube<T> > const_xz_type;
	typedef const ecuda::OffsettingContainer< const cube<T> > const_yz_type;

	typedef matrix<T> subscript_type;
	typedef subscript_type& subscript_reference;
	typedef const subscript_type& const_subscript_reference;

private:
	size_type numberRows;
	size_type numberColumns;
	size_type numberDepths;
	//size_type pitch;
	//device_ptr<T> deviceMemory;
	unique_ptr< matrix<T>[] > matrices;

public:
	HOST cube( const size_type numberRows=0, const size_type numberColumns=0, const size_type numberDepths=0, const T& value = T() ) : numberRows(numberRows), numberColumns(numberColumns), numberDepths(numberDepths), pitch(0) {
		if( numberRows and numberColumns and numberDepths ) {
			matrices = new matrix<T>[numberRows];
			for( size_t i = 0; i < numberRows; ++i ) *matrices[i] = matrix<T>( numberColumns, numberDepths, value );
			//CUDA_CALL( cudaMallocPitch( deviceMemory.alloc_ptr(), &pitch, numberColumns*numberDepths*sizeof(T), numberRows ) );
			//std::vector<T> v( numberColumns*numberDepths, value );
			//for( size_t i = 0; i < numberRows; ++i )
			//	CUDA_CALL( cudaMemcpy2D( deviceMemory.ptr()+(pitch*i), pitch, &v[0], v.size()*sizeof(T), numberColumns*numberDepths*sizeof(T), numberRows, cudaMemcpyHostToDevice ) );
		}
	}
	//TODO: is host/device difference in how underlying allocation correct here?
	HOST DEVICE cube( const cube<T>& src ) : numberRows(src.numberRows), numberColumns(src.numberColumns), numberDepths(src.numberDepths), matrices(src.matrices) {}
	template<typename U,typename V,typename W>
	HOST cube( const estd::cube<T,U,V,W>& src ) : numberRows(src.row_size()), numberColumns(src.column_size()), numberDepths(src.depth_size()), pitch(0) {
		if( numberRows and numberColumns and numberDepths ) {
			for( size_t i = 0; i < numberRows; ++i ) (*matrices[i]) = matrix<T>( src[i] );
			//CUDA_CALL( cudaMallocPitch( deviceMemory.alloc_ptr(), &pitch, numberColumns*numberDepths*sizeof(T), numberRows ) );
			//for( size_t i = 0; i < numberRows; ++i )
			//	CUDA_CALL( cudaMemcpy2D( deviceMemory.ptr()+(pitch*i), pitch, &src[i][0][0], numberColumns*numberDepths*sizeof(T), numberColumns*numberDepths*sizeof(T), numberRows ) );
		}
	}

	// capacity:
	HOST DEVICE inline size_type row_size() const __NOEXCEPT__ { return numberRows; }
	HOST DEVICE inline size_type column_size() const __NOEXCEPT__ { return numberColumns; }
	HOST DEVICE inline size_type depth_size() const __NOEXCEPT__ { return numberDepths; }
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return row_size()*column_size()*depth_size(); }
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return !size(); }

	// element access:
	DEVICE inline reference at( const size_type index ) { return matrices[index/(numberColumns*numberDepths)].at( index % (numberColumns*numberDepths) ); }
		//return *(deviceMemory.get()+( index/(numberColumns*numberDepths)*(pitch/sizeof(value_type)) + (index % (numberColumns*numberDepths))/numberDepths + (index % (numberColumns*numberDepths)) % numberDepths )); }
	DEVICE inline reference at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) { return matrices[rowIndex][columnIndex][depthIndex]; }
		//return at( rowIndex*numberColumns*numberDepths + columnIndex*numberDepths + depthIndex ); }
	DEVICE inline const_reference at( const size_type index ) const { return matrices[index/(numberColumns*numberDepths)].at( index % (numberColumns*numberDepths) ); }
		//return *(deviceMemory.get()+( index/(numberColumns*numberDepths)*(pitch/sizeof(value_type)) + (index % (numberColumns*numberDepths))/numberDepths + (index % (numberColumns*numberDepths)) % numberDepths )); }
	DEVICE inline const_reference at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) const { return matrices[rowIndex][columnIndex][depthIndex]; }
		//return at( rowIndex*numberColumns*numberDepths + columnIndex*numberDepths + depthIndex ); }
	//HOST DEVICE inline pointer data() __NOEXCEPT__ { return deviceMemory.get(); }
	//HOST DEVICE inline const_pointer data() const __NOEXCEPT__ { return deviceMemory.get(); }

	HOST DEVICE inline subscript_reference get_row( const size_t rowIndex ) { return matrices[rowIndex]; }
	HOST DEVICE inline const_subscript_reference get_row( const size_t rowIndex ) const { return matrices[rowIndex]; }

	//xy_type get_xy( const RowIndexType x, const ColumnIndexType y ) { return contents[x].get_row(y); }
	//xz_type get_xz( const RowIndexType x, const DepthIndexType z ) { return contents[x].get_column(z); }
	//yz_type get_yz( const ColumnIndexType y, const DepthIndexType z ) {	return OffsettingContainer< cube<CellType,RowIndexType,ColumnIndexType,DepthIndexType> >( *this, row_size(), y*depth_size()+z, column_size()*depth_size() ); }
	//const_xy_type get_xy( const RowIndexType x, const ColumnIndexType y ) const { return contents[x].get_row(y); }
	//const_xz_type get_xz( const RowIndexType x, const DepthIndexType z ) const { return contents[x].get_column(z); }
	//const_yz_type get_yz( const ColumnIndexType y, const DepthIndexType z ) const { return OffsettingContainer< const cube<CellType,RowIndexType,ColumnIndexType,DepthIndexType> >( *this, row_size(), y*depth_size()+z, column_size()*depth_size() ); }

	HOST DEVICE inline subscript_reference operator[]( const size_t rowIndex ) { return get_row(rowIndex); }
	HOST DEVICE inline const_subscript_reference operator[]( const size_t rowIndex ) const { return get_row(rowIndex); }

};

} // namespace ecuda

#endif

