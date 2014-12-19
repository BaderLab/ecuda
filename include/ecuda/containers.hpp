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
// containers.hpp
// Custom STL-like container classes.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_CONTAINERS_HPP
#define ECUDA_CONTAINERS_HPP

#include <algorithm>
#include <set>
#include <vector>

#include "global.hpp"
#include "iterators.hpp"

namespace ecuda {

///
/// Helper container that encloses another container and skips elements at a defined offset and interval.
///
template<class ContainerType,class IndexType=typename ContainerType::size_type,typename PointerType=typename ContainerType::pointer>
class OffsettingContainer
{
public:
	typedef ContainerType container_type;
	typedef typename ContainerType::value_type value_type;
	typedef PointerType pointer;
	//typedef PointerType pointer_type;
	//typedef pointer_type pointer;
	typedef typename dereference<pointer>::type reference;
	typedef const pointer const_pointer;
	typedef typename dereference<const_pointer>::type const_reference;
	typedef RandomAccessIterator< OffsettingContainer<ContainerType,IndexType,PointerType>, pointer > iterator;
	typedef RandomAccessIterator< const OffsettingContainer<ContainerType,IndexType,PointerType>, const_pointer > const_iterator;
	typedef ReverseIterator< RandomAccessIterator< OffsettingContainer<ContainerType,IndexType,PointerType>, pointer > > reverse_iterator;
	typedef ReverseIterator< RandomAccessIterator< const OffsettingContainer<ContainerType,IndexType,PointerType>, const_pointer > > const_reverse_iterator;
	typedef typename ContainerType::difference_type difference_type;
	typedef IndexType size_type;
	//typedef typename ContainerType::size_type size_type;

private:
	container_type& container;
	const size_type extent;
	const size_type offset;
	const size_type increment;

public:
	///
	/// \param container The parent container.
	/// \param extent The desired number of elements to enclose (wrt. enclosing container).
	/// \param offset The offset of the starting element (wrt. parent container).
	/// \param increment The number of elements (wrt. parent container) to increment for each increase in the index (wrt. enclosing container).
	///
	/// e.g. Let a std::vector v have elements [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... ].
	///      Then OffsettingContainer( v, 3, 4, 2 ) will reflect elements [ 4, 6, 8 ].
	///
	HOST DEVICE OffsettingContainer( ContainerType& container, const IndexType extent, const IndexType offset=0, const IndexType increment=1 ) : container(container), extent(extent), offset(offset), increment(increment) {}
	HOST DEVICE OffsettingContainer( const OffsettingContainer& src ) : container(src.container), extent(src.extent), offset(src.offset), increment(src.increment) {}
	HOST DEVICE virtual ~OffsettingContainer() {}

	// iterators:
	DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(this,0); }
	DEVICE inline iterator end() __NOEXCEPT__ { return iterator(this,size()); }
	DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(this,0); }
	DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(this,size()); }
	DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(iterator(this,size())); }
	DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(iterator(this,0)); }
	DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(this,size())); }
	DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(this,0)); }

	// capacity:
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return extent; }
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return !size() ? true : false; }

	// element access:
	DEVICE inline reference operator[]( size_type index ) { return container.at( offset+index*increment ); }
	DEVICE inline reference at( size_type index ) { return operator[]( index ); }
	DEVICE inline reference front() { return operator[](0); }
	DEVICE inline reference back() { return operator[](size()-1); }
	DEVICE inline const_reference operator[]( size_type index ) const { return container.at( offset+index*increment ); }
	DEVICE inline const_reference at( size_type index ) const { return operator[]( index ); }
	DEVICE inline const_reference front() const { return operator[](0); }
	DEVICE inline const_reference back() const { return operator[](size()-1); }

};

///
/// Helper container that encloses a cube and makes a particular slice accessible as a matrix.
///
template<class CubeType,class IndexType=typename CubeType::size_type,typename PointerType=typename CubeType::pointer>
class CubeSliceContainer
{
public:
	typedef typename CubeType::value_type value_type;
	typedef PointerType pointer_type;
	typedef pointer_type pointer;
	typedef typename dereference<pointer>::type reference;
	typedef const pointer const_pointer;
	typedef typename dereference<const_pointer>::type const_reference;
	typedef typename CubeType::difference_type difference_type;
	typedef typename CubeType::size_type size_type;
	typedef OffsettingContainer< CubeSliceContainer<CubeType,IndexType,PointerType>, size_type, pointer > row_type;
	typedef OffsettingContainer< CubeSliceContainer<CubeType,IndexType,PointerType>, size_type, pointer > column_type;
	typedef const OffsettingContainer< const CubeSliceContainer<CubeType,IndexType,PointerType>, size_type, const_pointer > const_row_type;
	typedef const OffsettingContainer< const CubeSliceContainer<CubeType,IndexType,PointerType>, size_type, const_pointer > const_column_type;

private:
	CubeType& cube;
	const IndexType row;

public:
	HOST DEVICE CubeSliceContainer( CubeType& cube, const IndexType row ) : cube(cube), row(row) {}
	HOST DEVICE CubeSliceContainer( const CubeSliceContainer& src ) : cube(src.cube), row(src.row) {}
	HOST DEVICE virtual ~CubeSliceContainer() {}

	// iterators:
	DEVICE inline row_type get_row( size_type index ) { return row_type( *this, cube.column_size(), index*cube.depth_size() ); }
	DEVICE inline const_row_type get_row( size_type index ) const { return const_row_type( *this, cube.column_size(), index*cube.depth_size() ); }
	DEVICE inline column_type get_column( size_type index ) { return column_type( *this, cube.depth_size(), index, cube.column_size() ); }
	DEVICE inline const_column_type get_column( size_type index ) const { return const_column_type( *this, cube.depth_size(), index, cube.column_size() ); }

	// capacity:
	HOST DEVICE inline size_type row_size() const __NOEXCEPT__ { return cube.column_size(); }
	HOST DEVICE inline size_type column_size() const __NOEXCEPT__ { return cube.depth_size(); }

	// element access:
	DEVICE inline reference at( size_type index ) { return cube.at( row*cube.column_size()*cube.depth_size()+index ); }
	DEVICE inline reference at( size_type rowIndex, size_type columnIndex ) { return cube.at( row, rowIndex, columnIndex ); }
	DEVICE inline const_reference at( size_type index ) const { return cube.at( row*cube.column_size()*cube.depth_size()+index ); }
	DEVICE inline const_reference at( size_type rowIndex, size_type columnIndex ) const { return cube.at( row, rowIndex, columnIndex ); }

	DEVICE inline row_type operator[]( size_type index ) { return get_row(index); }
	DEVICE inline const_row_type operator[]( size_type index ) const { return get_row(index); }

	template<typename U,typename V>
	HOST CubeSliceContainer& operator>>( estd::matrix<value_type,U,V>& dest ) {
		dest.resize( static_cast<U>(row_size()), static_cast<V>(column_size()) );
		CUDA_CALL( cudaMemcpy( dest.data(), cube.data()+(row*cube.get_pitch()/sizeof(value_type)), row_size()*column_size()*sizeof(value_type), cudaMemcpyDeviceToHost ) );
		return *this;
	}

};


} // namespace estd

#endif
