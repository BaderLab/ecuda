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

#include "config.hpp"
#if HAVE_ESTD_LIBRARY > 0
#include <estd/cube.hpp>
#endif
#include "global.hpp"
#include "allocators.hpp"
#include "device_ptr.hpp"
#include "padded_ptr.hpp"
#include "striding_ptr.hpp"
#include "matrix.hpp"
#include "memory.hpp"

namespace ecuda {

///
/// \brief A video-memory bound cube container.
///
/// A cube is defined as a 3D structure of dimensions rows*columns*depths. The default implementation
/// uses pitched memory where a 2D block of video memory is allocated with width=depths and height=rows*columns.
/// Pitched memory is aligned in a device-dependent manner so that calls to individual elements can
/// be threaded more efficiently (i.e. minimizing the number of read operations required to supply data to
/// multiple threads). Consult the CUDA API documentation for a more verbose explanation.
///
/// Memory use can be conceptualized as:
/// \code
///              |- depths -|
///              |---- pitch ----|
///    _     _   +----------+----+
///   |     |    |          |xxxx|
///   |  columns |          |xxxx| x = allocated but not used, just padding to
///   |     |_   |          |xxxx|     enforce an efficient memory alignment
///  rows        |          |xxxx|
///   |          |          |xxxx|
///   |          |          |xxxx|
///   |_         +----------+----+
/// \endcode
///
/// As a result, it is highly desirable for threading to utilize a depth-wise orientation.
/// For example, a good kernel to perform an operation on the elements of a cube might be:
///
/// \code{.cpp}
/// template<typename T> __global__ void doCubeOperation( ecuda::cube<T> cube ) {
///    const int dep = blockDim.x*gridDim.x; // each thread gets a different depth value
///    const int row = blockIdx.y;
///    const int col = blockIdx.z;
///    if( row < cube.number_rows() and col < cube.number_columns() and dep < cube.number_depths() ) {
///       T& value = cube[row][col][dep];
///       // ... do work on value
///    }
/// }
/// \endcode
///
/// This could be called from host code like:
/// \code{.cpp}
/// ecuda::cube<double> cube( 10, 20, 1000 );
/// // ... fill cube with data
/// dim3 grid( 1, 10, 20 ), block( 1000, 1, 1 );
/// doCubeOperation<<<grid,block>>>( cube );
/// \endcode
///
/// Unfortunately, CUDA solutions are very problem specific, so there is no generally applicable example for
/// specifying how thread blocks should be defined.  The size of the cube, hardware limitations, CUDA API
/// limitations, etc. all play a part.  For example, the above implementation won't work in earlier versions
/// of CUDA since blockDim.x is limited to 512.
///
/// Just keep in mind that the depth dimension lies in contiguous memory, the column dimension is contiguous
/// blocks of depth blocks, and the row dimension is contiguous blocks of column blocks; thus, an implementation
/// that aims to have concurrently running threads accessing depth >>> column > row will run much more efficiently.
///
template< typename T, class Alloc=DevicePitchAllocator<T> >
class cube {

public:
	typedef T value_type; //!< cell data type
	typedef Alloc allocator_type; //!< allocator type
	typedef std::size_t size_type; //!< unsigned integral type
	typedef std::ptrdiff_t difference_type; //!< signed integral type
	#ifdef __CPP11_SUPPORTED__
	typedef value_type& reference; //!< cell reference type
	typedef const value_type& const_reference; //!< cell const reference type
	typedef typename std::allocator_traits<Alloc>::pointer pointer; //!< cell pointer type
	typedef typename std::allocator_traits<Alloc>::const_pointer const_pointer; //!< cell const pointer type
	#else
	typedef typename Alloc::reference reference; //!< cell reference type
	typedef typename Alloc::const_reference const_reference; //!< cell const reference type
	typedef typename Alloc::pointer pointer; //!< cell pointer type
	typedef typename Alloc::const_pointer const_pointer; //!< cell const pointer type
	#endif

	//typedef T* pointer;
	//typedef const T* const_pointer;
	//typedef T& reference;
	//typedef const T& const_reference;

	typedef temporary_array< value_type, striding_ptr< value_type, padded_ptr<value_type,pointer,1> > > row_type; //!< cube row type
	typedef temporary_array< value_type, striding_ptr< value_type, padded_ptr<value_type,pointer,1> > > column_type; //!< cube column type
	typedef temporary_array< value_type                                                               > depth_type; //!< cube depth type
	typedef temporary_array< const value_type, striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > > const_row_type; //!< const cube row type
	typedef temporary_array< const value_type, striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > > const_column_type; //!< const cube column type
	typedef temporary_array< const value_type                                                                                 > const_depth_type; //!< const cube depth type

	typedef temporary_matrix< value_type, striding_ptr< value_type, padded_ptr<value_type,pointer,1> > > slice_xy_type; //!< cube xy-slice type
	typedef temporary_matrix< value_type,                           padded_ptr<value_type,pointer,1>   > slice_xz_type; //!< cube xz-slice type
	typedef temporary_matrix< value_type,                           padded_ptr<value_type,pointer,1>   > slice_yz_type; //!< cube yz-slice type
	typedef temporary_matrix< const value_type, striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > > const_slice_xy_type; //!< const cube xy-slice type
	typedef temporary_matrix< const value_type,                                 padded_ptr<const value_type,const_pointer,1>   > const_slice_xz_type; //!< const cube xz-slice type
	typedef temporary_matrix< const value_type,                                 padded_ptr<const value_type,const_pointer,1>   > const_slice_yz_type; //!< const cube yz-slice type

	typedef pointer_iterator< value_type, padded_ptr<value_type,pointer,1> > iterator; //!< iterator type
	typedef pointer_iterator< const value_type, padded_ptr<const value_type,const_pointer,1> > const_iterator; //!< const iterator type
	typedef pointer_reverse_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef pointer_reverse_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

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
	HOST cube( const size_type numberRows=0, const size_type numberColumns=0, const size_type numberDepths=0, const value_type& value = value_type(), const Alloc& allocator = Alloc() ) : numberRows(numberRows), numberColumns(numberColumns), numberDepths(numberDepths), allocator(allocator) {
		if( numberRows and numberColumns and numberDepths ) {
			deviceMemory = device_ptr<value_type>( get_allocator().allocate( numberDepths, numberRows*numberColumns, pitch ) );
			std::vector<value_type> v( numberDepths, value );
			for( size_type i = 0; i < numberRows; ++i ) {
				for( size_type j = 0; j < numberColumns; ++j ) {
					CUDA_CALL( cudaMemcpy<value_type>(
						get_allocator().address( deviceMemory.get(), i*numberColumns+j, 0, pitch ), // dst
						&v.front(), // src
						numberDepths, // count
						cudaMemcpyHostToDevice
					) );
				}
			}
			/*
			std::vector<T> v( numberColumns*numberDepths, value );
			for( size_t i = 0; i < numberRows; ++i )
				CUDA_CALL( cudaMemcpy<T>(
					get_allocator().address( deviceMemory.get(), i, 0, pitch ), // dst
					&v[0], // src
					numberColumns*numberDepths, // count
					cudaMemcpyHostToDevice // kind
				) );
			*/
		}
	}

	HOST DEVICE cube( const cube<T>& src ) : numberRows(src.numberRows), numberColumns(src.numberColumns), numberDepths(src.numberDepths), pitch(src.pitch), deviceMemory(src.deviceMemory), allocator(src.allocator) {}

	#if HAVE_ESTD_LIBRARY > 0
	template<typename U,typename V,typename W>
	HOST cube( const estd::cube<T,U,V,W>& src ) : numberRows(src.row_size()), numberColumns(src.column_size()), numberDepths(src.depth_size()) {
		if( numberRows and numberColumns and numberDepths ) {
			deviceMemory = device_ptr<value_type>( get_allocator().allocate( numberDepths, numberRows*numberColumns, pitch ) );
			std::vector<value_type> v( numberDepths );
			for( size_type i = 0; i < numberRows; ++i ) {
				for( size_type j = 0; j < numberColumns; ++j ) {
					for( size_type k = 0; k < numberDepths; ++k ) v[k] = src[i][j][k];
					CUDA_CALL( cudaMemcpy<value_type>(
						get_allocator().address( deviceMemory.get(), i*numberColumns+j, 0, pitch ), // dst
						&v.front(), // src
						numberDepths, // count
						cudaMemcpyHostToDevice
					) );
				}
			}
			/*
			for( size_t i = 0; i < numberRows; ++i ) {
				std::vector<T> v; v.reserve( numberColumns*numberDepths );
				for( size_t j = 0; j < numberColumns; ++j )
					for( size_t k = 0; k < numberDepths; ++k )
						v.push_back( src[i][j][k] );
				CUDA_CALL( cudaMemcpy<T>(
					get_allocator().address( deviceMemory.get(), i, 0, pitch ), // dst
					&v[0], // src
					numberColumns*numberDepths, // count
					cudaMemcpyHostToDevice // kind
				) );
			}
			*/
		}
	}
	#endif

	// capacity:
	HOST DEVICE inline size_type row_size() const __NOEXCEPT__ { return numberRows; }
	HOST DEVICE inline size_type column_size() const __NOEXCEPT__ { return numberColumns; }
	HOST DEVICE inline size_type depth_size() const __NOEXCEPT__ { return numberDepths; }
	HOST DEVICE inline size_type get_pitch() const { return pitch; }
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return row_size()*column_size()*depth_size(); }
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return !size(); }

	// element access:
	DEVICE inline reference at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) { return *allocator.address( deviceMemory.get(), rowIndex*column_size()+columnIndex, depthIndex, pitch ); }
	//DEVICE inline reference at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) { return *(deviceMemory.get()+(rowIndex*pitch/sizeof(T)+columnIndex*numberDepths+depthIndex)); }
	DEVICE inline reference at( const size_type index ) { return at( index/(numberColumns*numberDepths), (index % (numberColumns*numberDepths))/numberDepths, (index % (numberColumns*numberDepths)) % numberDepths ); }
	DEVICE inline const_reference at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) const { return *allocator.address( deviceMemory.get(), rowIndex*column_size()+columnIndex, depthIndex, pitch ); }
	//DEVICE inline const_reference at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) const { return *(deviceMemory.get()+(rowIndex*pitch/sizeof(T)+columnIndex*numberDepths+depthIndex)); }
	DEVICE inline const_reference at( const size_type index ) const { return at( index/(numberColumns*numberDepths), (index % (numberColumns*numberDepths))/numberDepths, (index % (numberColumns*numberDepths)) % numberDepths ); }
	HOST DEVICE inline pointer data() __NOEXCEPT__ { return deviceMemory.get(); }
	HOST DEVICE inline const_pointer data() const __NOEXCEPT__ { return deviceMemory.get(); }

	HOST DEVICE inline iterator begin() { return iterator( padded_ptr<value_type,pointer,1>( data(), depth_size(), get_pitch()-depth_size()*sizeof(value_type), 0 ) ); }
	HOST DEVICE inline iterator end() { return iterator( padded_ptr<value_type,pointer,1>( allocator.address( data(), row_size()*column_size(), 0, get_pitch() ), depth_size(), get_pitch()-depth_size()*sizeof(value_type), 0 ) ); }
	HOST DEVICE inline const_iterator begin() const { return const_iterator( padded_ptr<const value_type,const_pointer,1>( data(), depth_size(), get_pitch()-depth_size()*sizeof(value_type), 0 ) ); }
	HOST DEVICE inline const_iterator end() const { return const_iterator( padded_ptr<const value_type,const_pointer,1>( allocator.address( data(), row_size()*column_size(), 0, get_pitch() ), depth_size(), get_pitch()-depth_size()*sizeof(value_type), 0 ) ); }

	HOST DEVICE inline reverse_iterator rbegin() { return reverse_iterator(end()); }
	HOST DEVICE inline reverse_iterator rend() { return reverse_iterator(begin()); }
	HOST DEVICE inline const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
	HOST DEVICE inline const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

	HOST DEVICE inline row_type get_row( const size_type columnIndex, const size_type depthIndex ) {
		pointer np = allocator.address( deviceMemory.get(), columnIndex, depthIndex, pitch );
		padded_ptr<value_type,pointer,1> pp( np, depth_size(), pitch-depth_size()*sizeof(value_type), depthIndex );
		striding_ptr< value_type, padded_ptr<value_type,pointer,1> > sp( pp, column_size()*depth_size() );
		return row_type( sp, row_size() );
		//return row_type( strided_ptr<value_type,1>( allocator.address( deviceMemory.get(), columnIndex*row_size(), depthIndex, pitch ), pitch*numberColumns ), row_size() );
	}
	HOST DEVICE inline column_type get_column( const size_type rowIndex, const size_type depthIndex ) {
		pointer np = allocator.address( deviceMemory.get(), rowIndex*column_size(), depthIndex, pitch );
		padded_ptr<value_type,pointer,1> pp( np, depth_size(), pitch-depth_size()*sizeof(value_type), depthIndex );
		striding_ptr< value_type, padded_ptr<value_type,pointer,1> > sp( pp, depth_size() );
		return column_type( sp, column_size() );
		//return column_type( strided_ptr<value_type,1>( allocator.address( deviceMemory.get(), rowIndex, depthIndex, pitch ), pitch ), column_size() );
	}
	HOST DEVICE inline depth_type get_depth( const size_type rowIndex, const size_type columnIndex ) {
		pointer np = allocator.address( deviceMemory.get(), rowIndex*column_size()+columnIndex, 0, pitch );
		return depth_type( np, depth_size() );
		//return depth_type( allocator.address( deviceMemory.get(), rowIndex*column_size()+columnIndex, 0, pitch ), depth_size() );
	}
	HOST DEVICE inline const_row_type get_row( const size_type columnIndex, const size_type depthIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), columnIndex, depthIndex, pitch );
		padded_ptr<const value_type,const_pointer,1> pp( np, depth_size(), pitch-depth_size()*sizeof(value_type), depthIndex );
		striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > sp( pp, column_size()*depth_size() );
		return const_row_type( sp, row_size() );
		//return const_row_type( strided_ptr<const value_type,1>( allocator.address( deviceMemory.get(), columnIndex*row_size(), depthIndex, pitch ), pitch*numberColumns ), row_size() );
	}
	HOST DEVICE inline const_column_type get_column( const size_type rowIndex, const size_type depthIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), rowIndex*column_size(), depthIndex, pitch );
		padded_ptr<const value_type,const_pointer,1> pp( np, depth_size(), pitch-depth_size()*sizeof(value_type), depthIndex );
		striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > sp( pp, depth_size() );
		return const_column_type( sp, column_size() );
		//return const_column_type( strided_ptr<const value_type,1>( allocator.address( deviceMemory.get(), rowIndex, depthIndex, pitch ), pitch ), column_size() );
	}
	HOST DEVICE inline const_depth_type get_depth( const size_type rowIndex, const size_type columnIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), rowIndex*column_size()+columnIndex, 0, pitch );
		return const_depth_type( np, depth_size() );
		//return const_depth_type( allocator.address( deviceMemory.get(), rowIndex*column_size()+columnIndex, 0, pitch ), depth_size() );
	}

//	HOST DEVICE inline slice_type get_row( const size_type rowIndex ) { return slice_type( allocator.address( deviceMemory.get(), rowIndex*column_size(), 0, pitch ), column_size(), depth_size(), get_pitch() ); }
//	HOST DEVICE inline const_slice_type get_row( const size_type rowIndex ) const { return const_slice_type( allocator.address( deviceMemory.get(), rowIndex*column_size(), 0, pitch ), column_size(), depth_size(), get_pitch() ); }
//	HOST DEVICE inline slice_type operator[]( const size_type rowIndex ) { return get_row(rowIndex); }
//	HOST DEVICE inline const_slice_type operator[]( const size_type rowIndex ) const { return get_row(rowIndex); }

/*
change to:
       d
  +--------+--+
  |        |  |
rc|        |  |
  |        |  |
  +--------+--+
  \__________/
      pitch

xz_type stride=c*pitch
yz_type stride=0
xy_type stride=pitch
*/
	HOST DEVICE inline slice_yz_type get_yz( const size_type rowIndex ) {
		pointer np = allocator.address( deviceMemory.get(), rowIndex*column_size(), 0, pitch );
		padded_ptr<value_type,pointer,1> pp( np, depth_size(), pitch-depth_size()*sizeof(value_type), 0 );
		return slice_yz_type( pp, column_size(), depth_size() );
	}

	HOST DEVICE inline slice_xy_type get_xy( const size_type depthIndex ) {
		pointer np = allocator.address( deviceMemory.get(), 0, depthIndex, pitch );
		padded_ptr<value_type,pointer,1> pp( np, depth_size(), pitch-depth_size()*sizeof(value_type), 0 );
		striding_ptr<value_type, padded_ptr<value_type,pointer,1> > sp( pp, depth_size() );
		return slice_xy_type( sp, row_size(), column_size() );
	}

	HOST DEVICE inline slice_xz_type get_xz( const size_type columnIndex ) {
		pointer np = allocator.address( deviceMemory.get(), columnIndex, 0, pitch );
		const size_type padding = (pitch-depth_size()*sizeof(value_type)) + (column_size()-1)*pitch;
		padded_ptr<value_type,pointer,1> pp( np, depth_size(), padding, 0 );
		return slice_xz_type( pp, row_size(), depth_size() );
	}

	HOST DEVICE inline const_slice_yz_type get_yz( const size_type rowIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), rowIndex*column_size(), 0, pitch );
		padded_ptr<const value_type,const_pointer,1> pp( np, depth_size(), pitch-depth_size()*sizeof(value_type), 0 );
		return const_slice_yz_type( pp, column_size(), depth_size() );
	}

	HOST DEVICE inline const_slice_xy_type get_xy( const size_type depthIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), 0, depthIndex, pitch );
		padded_ptr<const value_type,const_pointer,1> pp( np, depth_size(), pitch-depth_size()*sizeof(value_type), 0 );
		striding_ptr<const value_type, padded_ptr<const value_type,const_pointer,1> > sp( pp, depth_size() );
		return const_slice_xy_type( sp, row_size(), column_size() );
	}

	HOST DEVICE inline const_slice_xz_type get_xz( const size_type columnIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), columnIndex, 0, pitch );
		const size_type padding = (pitch-depth_size()*sizeof(value_type)) + (column_size()-1)*pitch;
		padded_ptr<const value_type,const_pointer,1> pp( np, depth_size(), padding, 0 );
		return const_slice_xz_type( pp, row_size(), depth_size() );
	}

	/*
	HOST DEVICE inline xy_type get_xy( const size_type x, const size_type y ) { return xy_type( allocator.address( deviceMemory.get(), x*column_size()+y, 0, pitch ), depth_size() ); }
	HOST DEVICE inline xz_type get_xz( const size_type x, const size_type z ) { return xz_type( strided_ptr<value_type,1>( allocator.address( deviceMemory.get(), x*column_size(), z, pitch ), pitch ), column_size() ); }
	HOST DEVICE inline yz_type get_yz( const size_type y, const size_type z ) { return yz_type( strided_ptr<value_type,1>( allocator.address( deviceMemory.get(), y, z, pitch ), column_size()*pitch ), row_size() ); }
	HOST DEVICE inline const_xy_type get_xy( const size_type x, const size_type y ) const { return const_xy_type( allocator.address( deviceMemory.get(), x*column_size()+y, 0, pitch ), depth_size() ); }
	HOST DEVICE inline const_xz_type get_xz( const size_type x, const size_type z ) const { return const_xz_type( strided_ptr<value_type,1>( allocator.address( deviceMemory.get(), x*column_size(), z, pitch ), pitch ), column_size() ); }
	HOST DEVICE inline const_yz_type get_yz( const size_type y, const size_type z ) const { return const_yz_type( strided_ptr<value_type,1>( allocator.address( deviceMemory.get(), y, z, pitch ), column_size()*pitch ), row_size() ); }
	*/

	HOST DEVICE inline allocator_type get_allocator() const { return allocator; }

	#if HAVE_ESTD_LIBRARY > 0
	template<typename U,typename V,typename W>
	HOST cube<T,Alloc>& operator>>( estd::cube<T,U,V,W>& dest ) {
		//TODO: this can be optimized
		dest.resize( static_cast<U>(numberRows), static_cast<V>(numberColumns), static_cast<W>(numberDepths) );
		std::vector<value_type> tmp( numberDepths );
		for( size_type i = 0; i < numberRows; ++i ) {
			for( size_type j = 0; j < numberColumns; ++j ) {
				CUDA_CALL( cudaMemcpy<value_type>( &tmp.front(), allocator.address( deviceMemory.get(), i*numberColumns+j, 0, pitch ), numberDepths, cudaMemcpyDeviceToHost ) );
				for( size_type k = 0; k < numberDepths; ++k ) dest[i][j][k] = tmp[k];
			}
		}
		/*
		CUDA_CALL( cudaMemcpy2D<T>(
			&dest[0][0][0], // dest
			numberColumns*numberDepths*sizeof(T), // dpitch
			deviceMemory.get(), // src
			pitch, // spitch
			numberColumns*numberDepths, // width
			numberRows, // height
			cudaMemcpyDeviceToHost // kind
		) );
		*/
		return *this;
	}
	#endif

	HOST void resize( const size_type r, const size_type c, const size_type d ) {
		if( row_size() == r and column_size() == c and depth_size() == d ) return; // no resize needed
		cube<value_type,allocator_type> newCube( r, c, d );
		for( size_type i = 0; i < std::min(r,row_size()); ++i ) {
			for( size_type j = 0; j < std::min(c,column_size()); ++j ) {
				CUDA_CALL(
					cudaMemcpy<value_type>(
						newCube.allocator.address( newCube.deviceMemory.get(), i*newCube.numberColumns+j, 0, newCube.pitch ),
						allocator.address( deviceMemory.get(), i*numberColumns+j, 0, pitch ),
						std::min(d,depth_size()),
						cudaMemcpyDeviceToDevice
					)
				);
			}
		}
		// take the information from the new structure
		deviceMemory = newCube.deviceMemory;
		pitch = newCube.pitch;
		numberRows = newCube.numberRows;
		numberColumns = newCube.numberColumns;
		numberDepths = newCube.numberDepths;
	}

	HOST DEVICE void fill( const value_type& value ) {
		#ifdef __CUDA_ARCH__
		for( iterator iter = begin(); iter != end(); ++iter ) *iter = value;
		#else
		std::vector<value_type> v( depth_size(), value );
		// seed the device memory
		CUDA_CALL( cudaMemcpy<value_type>( deviceMemory.get(), &v.front(), depth_size(), cudaMemcpyHostToDevice ) );
		// make additional copies within the device
		for( size_type i = 1; i < row_size()*column_size(); ++i ) {
			CUDA_CALL(
				cudaMemcpy<value_type>(
					allocator.address( data(), i, 0, pitch ), deviceMemory.get(), depth_size(), cudaMemcpyDeviceToDevice
				)
			);
		}
		#endif
	}

	#if HAVE_ESTD_LIBRARY > 0
	template<typename U,typename V,typename W>
	HOST cube<T,Alloc>& operator<<( const estd::cube<T,U,V,W>& src ) {
		//TODO: this can be optimized
		resize( src.row_size(), src.column_size(), src.depth_size() );
		std::vector<value_type> tmp( src.depth_size() );
		for( typename estd::cube<T,U,V,W>::row_index_type i = 0; i < src.row_size(); ++i ) {
			for( typename estd::cube<T,U,V,W>::column_index_type j = 0; j < src.column_size(); ++j ) {
				for( typename estd::cube<T,U,V,W>::depth_index_type k = 0; k < src.depth_size(); ++k ) tmp[k] = src[i][j][k];
				CUDA_CALL( cudaMemcpy<value_type>( allocator.address( deviceMemory.get(), i*numberColumns+j, 0, pitch ), &tmp.front(), numberDepths, cudaMemcpyHostToDevice ) );
			}
		}
		//CUDA_CALL( cudaMemcpy2D<T>( data(), pitch, src.data(), numberColumns*numberDepths*sizeof(T), numberColumns*numberDepths, numberRows, cudaMemcpyHostToDevice ) );
		return *this;
	}
	#endif

};

} // namespace ecuda

#endif

