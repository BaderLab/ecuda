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
#include <stdexcept>
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
#include "views.hpp"

namespace ecuda {

///
/// \brief A resizable cube stored in device memory.
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
template< typename T, class Alloc=device_pitch_allocator<T> >
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

	typedef sequence_view< value_type, striding_ptr< value_type, padded_ptr<value_type,pointer,1> > > row_type; //!< cube row type
	typedef sequence_view< value_type, striding_ptr< value_type, padded_ptr<value_type,pointer,1> > > column_type; //!< cube column type
	typedef contiguous_sequence_view<value_type> depth_type; //!< cube depth type
	typedef sequence_view< const value_type, striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > > const_row_type; //!< const cube row type
	typedef sequence_view< const value_type, striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > > const_column_type; //!< const cube column type
	typedef contiguous_sequence_view<const value_type> const_depth_type; //!< const cube depth type

	typedef matrix_view< value_type, striding_ptr< value_type, padded_ptr<value_type,pointer,1> > > slice_xy_type; //!< cube xy-slice type
	typedef contiguous_matrix_view< value_type,                           padded_ptr<value_type,pointer,1>   > slice_xz_type; //!< cube xz-slice type
	typedef contiguous_matrix_view< value_type,                           padded_ptr<value_type,pointer,1>   > slice_yz_type; //!< cube yz-slice type
	typedef matrix_view< const value_type, striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > > const_slice_xy_type; //!< const cube xy-slice type
	typedef contiguous_matrix_view< const value_type,                                 padded_ptr<const value_type,const_pointer,1>   > const_slice_xz_type; //!< const cube xz-slice type
	typedef contiguous_matrix_view< const value_type,                                 padded_ptr<const value_type,const_pointer,1>   > const_slice_yz_type; //!< const cube yz-slice type

	typedef device_iterator< value_type, padded_ptr<value_type,pointer,1> > iterator; //!< iterator type
	typedef device_iterator< const value_type, padded_ptr<const value_type,const_pointer,1> > const_iterator; //!< const iterator type
	typedef reverse_device_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

private:
	// REMEMBER: numberRows, numberColumns, numberDepths and pitch altered on device memory won't be
	//           reflected on the host object. Don't allow the device to perform any operations that
	//           change their value.
	size_type numberRows; //!< number of rows
	size_type numberColumns; //!< number of columns
	size_type numberDepths; //!< number of depths
	size_type pitch; //!< pitch of device memory in bytes
	device_ptr<T> deviceMemory;
	allocator_type allocator;

public:
	///
	/// \brief Constructs a cube with dimensions numberRows x numberColumns x numberDepths filled with copies of elements with value value.
	/// \param numberRows number of rows (default=0)
	/// \param numberColumns number of columns (default=0)
	/// \param numberDepths number of depths (default=0)
	/// \param value value that cube elements should initially be set to (default=value_type())
	/// \param allocator allocator to use for all memory allocations of this container
	///        (does not normally need to be specified, by default the internal ecuda pitched memory allocator)
	///
	HOST cube( const size_type numberRows=0, const size_type numberColumns=0, const size_type numberDepths=0, const value_type& value = value_type(), const Alloc& allocator = Alloc() ) : numberRows(numberRows), numberColumns(numberColumns), numberDepths(numberDepths), allocator(allocator) {
		if( numberRows and numberColumns and numberDepths ) {
			deviceMemory = device_ptr<value_type>( get_allocator().allocate( numberDepths, numberRows*numberColumns, pitch ) );
			std::vector< value_type, host_allocator<value_type> > v( numberDepths, value );
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
		}
	}

	///
	/// \brief Constructs a cube with a shallow copy of each of the elements in src.
	///
	/// Be careful to note that a shallow copy means that only the pointer to the device memory
	/// that holds the elements is copied in the newly constructed container. This allows
	/// containers to be passed-by-value to kernel functions with minimal overhead. If a deep copy
	/// of the container is required in host code, use the assignment operator. For example:
	///
	/// \code{.cpp}
	/// ecuda::cube<int> cube( 5, 10, 15 ); // create a cube of dimensions 5x10x15 filled with zeroes
	/// ecuda::cube<int> newCube( cube ); // shallow copy (changes to newCube reflected in cube)
	/// ecuda::cube<int> newCube; newCube = cube; // deep copy (new device memory allocated and contents of cube copied there)
	/// \endcode
	///
	/// \param src Another cube object of the same type, whose contents are copied.
	///
	HOST DEVICE cube( const cube<T>& src ) :
		numberRows(src.numberRows),
		numberColumns(src.numberColumns),
		numberDepths(src.numberDepths),
		pitch(src.pitch),
		deviceMemory(src.deviceMemory),
		#ifdef __CPP11_SUPPORTED__
		allocator(std::allocator_traits<allocator_type>::select_on_container_copy_construction(src.get_allocator()))
		#else
		allocator(src.allocator)
		#endif
	{
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	/// \param src another container to be used as source to initialize the elements of the container with
	///
	HOST cube( cube<T>&& src ) : numberRows(src.numberRows), numberColumns(src.numberColumns), numberDepths(src.numberDepths), pitch(src.pitch), deviceMemory(std::move(src.deviceMemory)), allocator(std::move(src.allocator)) {}
	#endif

	#if HAVE_ESTD_LIBRARY > 0
	///
	/// \brief Constructs a cube by copying the dimensions and elements of an estd library cube container.
	///
	/// This method is enabled if the HAVE_ESTD_LIBRARY flag in config.hpp is set to non-zero.
	/// The estd library needs to be visible to the compiler.
	///
	/// \param src Another cube object of the same type, whose contents are copied.
	///
	template<typename U,typename V,typename W>
	HOST cube( const estd::cube<T,U,V,W>& src ) : numberRows(src.row_size()), numberColumns(src.column_size()), numberDepths(src.depth_size()) {
		if( numberRows and numberColumns and numberDepths ) {
			deviceMemory = device_ptr<value_type>( get_allocator().allocate( numberDepths, numberRows*numberColumns, pitch ) );
			std::vector< value_type, host_allocator<value_type> > v( numberDepths );
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
		}
	}
	#endif

	HOST DEVICE virtual ~cube() {}

	HOST inline allocator_type get_allocator() const { return allocator; }

	///
	/// \brief Replaces the contents of the container with copies of those in the range [begin,end).
	///
	/// The provided iterators must have STL random access capabilities (specifically,
	/// end.operator-(begin) so that the length of the sequence can be determined). The number of
	/// elements in [begin,end) must equal the size of this cube (i.e. rows*columns*depths). In
	/// addition, the orientation of the elements is assumed to be ordered depth->column->row
	/// (the same orientation as the elements stored in this container).
	///
	/// Note that a potentially more clear way of assigning values is to use the get_depth()
	/// method, which returns a structure that also has an assign() method.  For example:
	///
	/// \code{.cpp}
	/// estd::cube<int> cube( 3, 4, 5 ); // cube of dimension 3x4x5 and filled with zeroes
	/// std::vector<int> vec( { 66, 18, 96, 49, 58 } ); // vector initialized with a C++11 initializer list
	/// for( estd::cube<int>::size_type i = 0; i < cube.number_rows(); ++i )
	///	   for( estd::cube<int>::size_type j = 0; j < cube.number_columns(); ++j )
	///       cube[i][j].assign( vec.begin(), vec.end() );
	/// \endcode
	///
	/// \param begin,end the range to copy the elements from
	///
	template<class RandomAccessIterator>
	HOST void assign( RandomAccessIterator begin, RandomAccessIterator end ) {
		if( (end-begin) != size() ) throw std::length_error( "ecuda::cube::assign(begin,end) iterator range [begin,end) does not have correct length" );
		const size_type rc = number_rows()*number_columns();
		std::vector< value_type, host_allocator<value_type> > v( number_depths() );
		for( std::size_t i = 0; i < rc; ++i, begin += number_depths() ) {
			v.assign( begin, begin+number_depths() );
			//std::vector<value_type> v( begin, begin+number_depths() );
			CUDA_CALL( cudaMemcpy<value_type>( allocator.address( deviceMemory.get(), i, 0, pitch ), &v.begin(), number_depths(), cudaMemcpyHostToDevice ) );
		}
	}


	HOST DEVICE inline size_type number_rows() const __NOEXCEPT__ { return numberRows; }
	HOST DEVICE inline size_type number_columns() const __NOEXCEPT__ { return numberColumns; }
	HOST DEVICE inline size_type number_depths() const __NOEXCEPT__ { return numberDepths; }
	HOST DEVICE inline size_type get_pitch() const { return pitch; }
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return number_rows()*number_columns()*number_depths(); }
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return !size(); }

	HOST DEVICE inline pointer data() __NOEXCEPT__ { return deviceMemory.get(); }
	HOST DEVICE inline const_pointer data() const __NOEXCEPT__ { return deviceMemory.get(); }

	HOST DEVICE inline iterator begin() { return iterator( padded_ptr<value_type,pointer,1>( data(), number_depths(), get_pitch()-number_depths()*sizeof(value_type), 0 ) ); }
	HOST DEVICE inline iterator end() { return iterator( padded_ptr<value_type,pointer,1>( allocator.address( data(), number_rows()*number_columns(), 0, get_pitch() ), number_depths(), get_pitch()-number_depths()*sizeof(value_type), 0 ) ); }
	HOST DEVICE inline const_iterator begin() const { return const_iterator( padded_ptr<const value_type,const_pointer,1>( data(), number_depths(), get_pitch()-number_depths()*sizeof(value_type), 0 ) ); }
	HOST DEVICE inline const_iterator end() const { return const_iterator( padded_ptr<const value_type,const_pointer,1>( allocator.address( data(), number_rows()*number_columns(), 0, get_pitch() ), number_depths(), get_pitch()-number_depths()*sizeof(value_type), 0 ) ); }

	HOST DEVICE inline reverse_iterator rbegin() { return reverse_iterator(end()); }
	HOST DEVICE inline reverse_iterator rend() { return reverse_iterator(begin()); }
	HOST DEVICE inline const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
	HOST DEVICE inline const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

	HOST DEVICE inline row_type get_row( const size_type columnIndex, const size_type depthIndex ) {
		pointer np = allocator.address( deviceMemory.get(), columnIndex, depthIndex, pitch );
		padded_ptr<value_type,pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), depthIndex );
		striding_ptr< value_type, padded_ptr<value_type,pointer,1> > sp( pp, number_columns()*number_depths() );
		return row_type( sp, number_rows() );
	}
	HOST DEVICE inline column_type get_column( const size_type rowIndex, const size_type depthIndex ) {
		pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns(), depthIndex, pitch );
		padded_ptr<value_type,pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), depthIndex );
		striding_ptr< value_type, padded_ptr<value_type,pointer,1> > sp( pp, number_depths() );
		return column_type( sp, number_columns() );
	}
	HOST DEVICE inline depth_type get_depth( const size_type rowIndex, const size_type columnIndex ) {
		pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns()+columnIndex, 0, pitch );
		return depth_type( np, number_depths() );
	}
	HOST DEVICE inline const_row_type get_row( const size_type columnIndex, const size_type depthIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), columnIndex, depthIndex, pitch );
		padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), depthIndex );
		striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > sp( pp, number_columns()*number_depths() );
		return const_row_type( sp, number_rows() );
	}
	HOST DEVICE inline const_column_type get_column( const size_type rowIndex, const size_type depthIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns(), depthIndex, pitch );
		padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), depthIndex );
		striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > sp( pp, number_depths() );
		return const_column_type( sp, number_columns() );
	}
	HOST DEVICE inline const_depth_type get_depth( const size_type rowIndex, const size_type columnIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns()+columnIndex, 0, pitch );
		return const_depth_type( np, number_depths() );
	}

	HOST DEVICE inline slice_yz_type get_yz( const size_type rowIndex ) {
		pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns(), 0, pitch );
		padded_ptr<value_type,pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), 0 );
		return slice_yz_type( pp, number_depths(), number_columns() );
	}

	HOST DEVICE inline slice_xy_type get_xy( const size_type depthIndex ) {
		pointer np = allocator.address( deviceMemory.get(), 0, depthIndex, pitch );
		padded_ptr<value_type,pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), 0 );
		striding_ptr<value_type, padded_ptr<value_type,pointer,1> > sp( pp, number_depths() );
		return slice_xy_type( sp, number_columns(), number_rows() );
	}

	HOST DEVICE inline slice_xz_type get_xz( const size_type columnIndex ) {
		pointer np = allocator.address( deviceMemory.get(), columnIndex, 0, pitch );
		const size_type padding = (pitch-number_depths()*sizeof(value_type)) + (number_columns()-1)*pitch;
		padded_ptr<value_type,pointer,1> pp( np, number_depths(), padding, 0 );
		return slice_xz_type( pp, number_depths(), number_rows() );
	}

	HOST DEVICE inline const_slice_yz_type get_yz( const size_type rowIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns(), 0, pitch );
		padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), 0 );
		return const_slice_yz_type( pp, number_depths(), number_columns() );
	}

	HOST DEVICE inline const_slice_xy_type get_xy( const size_type depthIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), 0, depthIndex, pitch );
		padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), 0 );
		striding_ptr<const value_type, padded_ptr<const value_type,const_pointer,1> > sp( pp, number_depths() );
		return const_slice_xy_type( sp, number_columns(), number_rows() );
	}

	HOST DEVICE inline const_slice_xz_type get_xz( const size_type columnIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), columnIndex, 0, pitch );
		const size_type padding = (pitch-number_depths()*sizeof(value_type)) + (number_columns()-1)*pitch;
		padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), padding, 0 );
		return const_slice_xz_type( pp, number_depths(), number_rows() );
	}


	#if HAVE_ESTD_LIBRARY > 0
	template<typename U,typename V,typename W>
	HOST cube<T,Alloc>& operator>>( estd::cube<T,U,V,W>& dest ) {
		//TODO: this can be optimized
		dest.resize( static_cast<U>(numberRows), static_cast<V>(numberColumns), static_cast<W>(numberDepths) );
		std::vector< value_type, host_allocator<value_type> > tmp( numberDepths );
		for( size_type i = 0; i < numberRows; ++i ) {
			for( size_type j = 0; j < numberColumns; ++j ) {
				CUDA_CALL( cudaMemcpy<value_type>( &tmp.front(), allocator.address( deviceMemory.get(), i*numberColumns+j, 0, pitch ), numberDepths, cudaMemcpyDeviceToHost ) );
				for( size_type k = 0; k < numberDepths; ++k ) dest[i][j][k] = tmp[k];
			}
		}
		return *this;
	}
	#endif

	HOST void resize( const size_type r, const size_type c, const size_type d ) {
		if( number_rows() == r and number_columns() == c and number_depths() == d ) return; // no resize needed
		cube<value_type,allocator_type> newCube( r, c, d );
		for( size_type i = 0; i < std::min(r,number_rows()); ++i ) {
			for( size_type j = 0; j < std::min(c,number_columns()); ++j ) {
				CUDA_CALL(
					cudaMemcpy<value_type>(
						newCube.allocator.address( newCube.deviceMemory.get(), i*newCube.numberColumns+j, 0, newCube.pitch ),
						allocator.address( deviceMemory.get(), i*numberColumns+j, 0, pitch ),
						std::min(d,number_depths()),
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
		std::vector< value_type, host_allocator<value_type> > v( number_depths(), value );
		// seed the device memory
		CUDA_CALL( cudaMemcpy<value_type>( deviceMemory.get(), &v.front(), number_depths(), cudaMemcpyHostToDevice ) );
		// make additional copies within the device
		for( size_type i = 1; i < number_rows()*number_columns(); ++i )
			CUDA_CALL( cudaMemcpy<value_type>( allocator.address( data(), i, 0, pitch ), deviceMemory.get(), number_depths(), cudaMemcpyDeviceToDevice ) );
		#endif
	}

	///
	/// \brief Assignment operator.
	///
	/// Copies the contents of other into this container.
	///
	/// Note that the behaviour differs depending on whether the assignment occurs on the
	/// host or the device. If called from the host, a deep copy is performed: additional
	/// memory is allocated in this container and the contents of other are copied there.
	/// If called from the device, a shallow copy is performed: the pointer to the device
	/// memory is copied only.  Therefore any changes made to this container are reflected
	/// in other as well, and vice versa.
	///
	/// \param other Container whose contents are to be assigned to this container.
	/// \return A reference to this container.
	///
	template<class Alloc2>
	HOST DEVICE cube<value_type,allocator_type>& operator=( const cube<value_type,Alloc2>& src ) {
		#ifdef __CUDA_ARCH__
		// shallow copy if called from device
		numberRows = src.numberRows;
		numberColumns = src.numberColumns;
		numberDepths = src.numberDepths;
		pitch = src.pitch;
		deviceMemory = src.deviceMemory;
		#else
		// deep copy if called from host
		numberRows = src.numberRows;
		numberColumns = src.numberColumns;
		numberDepths = src.numberDepths;
		deviceMemory = device_ptr<value_type>( allocator.allocate( numberDepths, numberRows*numberColumns, pitch ) );
		CUDA_CALL( cudaMemcpy2D<value_type>( deviceMemory.get(), pitch, src.deviceMemory.get(), src.pitch, numberDepths, numberRows*numberColumns, cudaMemcpyDeviceToDevice ) );
		#endif
		return *this;
	}

	#if HAVE_ESTD_LIBRARY > 0
	template<typename U,typename V,typename W>
	HOST cube<T,Alloc>& operator<<( const estd::cube<T,U,V,W>& src ) {
		//TODO: this can be optimized
		resize( src.row_size(), src.column_size(), src.depth_size() );
		std::vector< value_type, host_allocator<value_type> > tmp( src.depth_size() );
		for( typename estd::cube<T,U,V,W>::row_index_type i = 0; i < src.row_size(); ++i ) {
			for( typename estd::cube<T,U,V,W>::column_index_type j = 0; j < src.column_size(); ++j ) {
				for( typename estd::cube<T,U,V,W>::depth_index_type k = 0; k < src.depth_size(); ++k ) tmp[k] = src[i][j][k];
				CUDA_CALL( cudaMemcpy<value_type>( allocator.address( deviceMemory.get(), i*numberColumns+j, 0, pitch ), &tmp.front(), numberDepths, cudaMemcpyHostToDevice ) );
			}
		}
		return *this;
	}
	#endif

};

} // namespace ecuda

#endif

