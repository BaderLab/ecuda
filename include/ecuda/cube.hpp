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
/// Methods are prefaced with appropriate keywords to declare them as host and/or device capable.
/// In general: operations requiring memory allocation/deallocation are host only, operations
/// to access the values of specific elements are device only, and copy operations on ranges of data and
/// accessors of general information can be performed on both the host and device.
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
/// of CUDA when blockDim.x was limited to 512 (at the time of this writing it was 1024 in the newer versions
/// of CUDA).
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
	typedef contiguous_matrix_view<value_type> slice_xz_type; //!< cube xz-slice type
	typedef contiguous_matrix_view<value_type> slice_yz_type; //!< cube yz-slice type
	typedef matrix_view< const value_type, striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > > const_slice_xy_type; //!< const cube xy-slice type
	typedef contiguous_matrix_view<const value_type> const_slice_xz_type; //!< const cube xz-slice type
	typedef contiguous_matrix_view<const value_type> const_slice_yz_type; //!< const cube yz-slice type

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
			CUDA_CALL( cudaMemset2D<value_type>( deviceMemory.get(), pitch, value, numberDepths, numberRows*numberColumns ) );
		}
	}

	///
	/// \brief Constructs a cube with a shallow copy of each of the elements in src.
	///
	/// Be careful to note that a shallow copy means that only the pointer to the device memory
	/// that holds the elements is copied in the newly constructed container. This allows
	/// containers to be passed-by-value to kernel functions with minimal overhead. If a deep copy
	/// of the container is required in host code, use the << or >> operators, or use iterators.
	/// For example:
	///
	/// \code{.cpp}
	/// ecuda::cube<int> cube( 5, 10, 15 ); // create a cube of dimensions 5x10x15 filled with zeroes
	/// ecuda::cube<int> newCube( cube ); // shallow copy (changes to newCube reflected in cube)
	/// ecuda::cube<int> newCube( 5, 10, 15 );
	/// newCube << cube; // deep copy
	/// cube >> newCube; // deep copy
	/// newCube.assign( cube.begin(), cube.end() ); // deep copy
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
		//#ifdef __CPP11_SUPPORTED__
		//allocator(std::allocator_traits<allocator_type>::select_on_container_copy_construction(src.get_allocator()))
		//#else
		allocator(src.allocator)
		//#endif
	{
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	///
	/// This constructor is only available if the compiler is configured to allow C++11.
	///
	/// \param src another container to be used as source to initialize the elements of the container with
	///
	HOST cube( cube<T>&& src ) : numberRows(src.numberRows), numberColumns(src.numberColumns), numberDepths(src.numberDepths), pitch(src.pitch), deviceMemory(std::move(src.deviceMemory)), allocator(std::move(src.allocator)) {}
	#endif

	//HOST DEVICE virtual ~cube() {}

	///
	/// \brief Returns the allocator associated with the container.
	/// \returns The associated allocator.
	///
	HOST inline allocator_type get_allocator() const { return allocator; }

private:
	template<class Iterator>
	DEVICE inline void assign( Iterator first, Iterator last, device_iterator_tag ) {
		for( iterator iter = begin(); iter != end() and first != last; ++iter, ++first ) *iter = *first;
	}

	template<class Iterator>
	DEVICE inline void assign( Iterator first, Iterator last, contiguous_device_iterator_tag ) { assign( first, last, device_iterator_tag() ); }

	// dummy method to trick compiler, since device code will never use a non-device iterator
	template<class Iterator,class SomeOtherCategory>
	DEVICE inline void assign( Iterator first, Iterator last, SomeOtherCategory ) {}

public:

	///
	/// \brief Replaces the contents of the container with copies of those in the range [begin,end).
	///
	/// The number of elements in [begin,end) must equal the size of this cube (i.e. rows*columns*depths).
	/// In addition, the orientation of the elements is assumed to be ordered depth->column->row
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
	/// \param first,last the range to copy the elements from
	/// \throws std::length_error if number of elements doesn't match the size of the cube
	///
	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last ) {
		#ifdef __CUDA_ARCH__
		assign( first, last, typename std::iterator_traits<Iterator>::iterator_category() );
		#else
		typename std::iterator_traits<Iterator>::difference_type len = ::ecuda::distance(first,last);
		if( len < 0 or static_cast<size_type>(len) != size() )
			throw std::length_error( EXCEPTION_MSG("ecuda::cube::assign(begin,end) iterator range [begin,end) does not match the size of this cube") );
		Iterator endDepth = first;
		for( size_type i = 0; i < number_rows(); ++i ) {
			for( size_type j = 0; j < number_columns(); ++j ) {
				depth_type depth = get_depth(i,j);
				::ecuda::advance( endDepth, number_depths() );
				depth.assign( first, endDepth );
				first = endDepth;
			}
		}
		#endif
	}

	///
	/// \brief Returns the number of rows in the container.
	///
	/// \returns The number of rows in the container.
	///
	HOST DEVICE inline size_type number_rows() const __NOEXCEPT__ { return numberRows; }

	///
	/// \brief Returns the number of columns in the container.
	///
	/// \returns The number of columns in the container.
	///
	HOST DEVICE inline size_type number_columns() const __NOEXCEPT__ { return numberColumns; }

	///
	/// \brief Returns the number of depths in the container.
	///
	/// \returns The number of depths in the container.
	///
	HOST DEVICE inline size_type number_depths() const __NOEXCEPT__ { return numberDepths; }

	///
	/// \brief Returns the pitch of the underlying 2D device memory.
	///
	/// \returns The pitch (in bytes) of the underlying 2D device memory.
	///
	HOST DEVICE inline size_type get_pitch() const __NOEXCEPT__ { return pitch; }

	///
	/// \brief Returns the number of elements in the container.
	///
	/// This is the rows x columns x depths.
	///
	/// \returns The number of elements in the container.
	///
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return number_rows()*number_columns()*number_depths(); }

	///
	/// \brief Checks if the container has no elements.
	///
	/// \returns true if the container is empty, false otherwise.
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return !size(); }

	///
	/// \brief Returns pointer to the underlying 2D memory serving as element storage.
	///
	/// \returns Pointer to the underlying element storage.
	///
	HOST DEVICE inline pointer data() __NOEXCEPT__ { return deviceMemory.get(); }

	///
	/// \brief Returns pointer to the underlying 2D memory serving as element storage.
	///
	/// \returns Pointer to the underlying element storage.
	///
	HOST DEVICE inline const_pointer data() const __NOEXCEPT__ { return deviceMemory.get(); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator( padded_ptr<value_type,pointer,1>( data(), number_depths(), get_pitch()-number_depths()*sizeof(value_type), 0 ) ); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator( padded_ptr<value_type,pointer,1>( allocator.address( data(), number_rows()*number_columns(), 0, get_pitch() ), number_depths(), get_pitch()-number_depths()*sizeof(value_type), 0 ) ); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator( padded_ptr<const value_type,const_pointer,1>( data(), number_depths(), get_pitch()-number_depths()*sizeof(value_type), 0 ) ); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator( padded_ptr<const value_type,const_pointer,1>( allocator.address( data(), number_rows()*number_columns(), 0, get_pitch() ), number_depths(), get_pitch()-number_depths()*sizeof(value_type), 0 ) ); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(end()); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(begin()); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(end()); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(begin()); }

	#ifdef __CPP11_SUPPORTED__
	HOST DEVICE inline const_iterator cbegin() const __NOEXCEPT__ { return const_iterator( padded_ptr<const value_type,const_pointer,1>( data(), number_depths(), get_pitch()-number_depths()*sizeof(value_type), 0 ) ); }
	HOST DEVICE inline const_iterator cend() const __NOEXCEPT__ { return const_iterator( padded_ptr<const value_type,const_pointer,1>( allocator.address( data(), number_rows()*number_columns(), 0, get_pitch() ), number_depths(), get_pitch()-number_depths()*sizeof(value_type), 0 ) ); }
	HOST DEVICE inline const_reverse_iterator crbegin() __NOEXCEPT__ { return const_reverse_iterator(cend()); }
	HOST DEVICE inline const_reverse_iterator crend() __NOEXCEPT__ { return const_reverse_iterator(cbegin()); }
	#endif

	///
	/// \brief Gets a view of the sequence of elements forming a single row.
	///
	/// \param columnIndex the column to fix the view on
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements with the specified column and depth indices.
	///
	HOST DEVICE inline row_type get_row( const size_type columnIndex, const size_type depthIndex ) {
		pointer np = allocator.address( deviceMemory.get(), columnIndex, depthIndex, pitch );
		padded_ptr<value_type,pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), depthIndex );
		striding_ptr< value_type, padded_ptr<value_type,pointer,1> > sp( pp, number_columns()*number_depths() );
		return row_type( sp, number_rows() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single column.
	///
	/// \param rowIndex the row to fix the view on
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements with the specified row and depth indices.
	///
	HOST DEVICE inline column_type get_column( const size_type rowIndex, const size_type depthIndex ) {
		pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns(), depthIndex, pitch );
		padded_ptr<value_type,pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), depthIndex );
		striding_ptr< value_type, padded_ptr<value_type,pointer,1> > sp( pp, number_depths() );
		return column_type( sp, number_columns() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single depth.
	///
	/// \param rowIndex the row to fix the view on
	/// \param columnIndex the column to fix the view on
	/// \returns A view of the elements with the specified row and column indices.
	///
	HOST DEVICE inline depth_type get_depth( const size_type rowIndex, const size_type columnIndex ) {
		pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns()+columnIndex, 0, pitch );
		return depth_type( np, number_depths() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single row.
	///
	/// \param columnIndex the column to fix the view on
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements with the specified column and depth indices.
	///
	HOST DEVICE inline const_row_type get_row( const size_type columnIndex, const size_type depthIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), columnIndex, depthIndex, pitch );
		padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), depthIndex );
		striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > sp( pp, number_columns()*number_depths() );
		return const_row_type( sp, number_rows() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single column.
	///
	/// \param rowIndex the row to fix the view on
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements with the specified row and depth indices.
	///
	HOST DEVICE inline const_column_type get_column( const size_type rowIndex, const size_type depthIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns(), depthIndex, pitch );
		padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), depthIndex );
		striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > sp( pp, number_depths() );
		return const_column_type( sp, number_columns() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single depth.
	///
	/// \param rowIndex the row to fix the view on
	/// \param columnIndex the column to fix the view on
	/// \returns A view of the elements with the specified row and column indices.
	///
	HOST DEVICE inline const_depth_type get_depth( const size_type rowIndex, const size_type columnIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns()+columnIndex, 0, pitch );
		return const_depth_type( np, number_depths() );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single row.
	///
	/// \param rowIndex the row to fix the view on
	/// \returns A view of the elements at the specified row.
	///
	HOST DEVICE inline slice_yz_type get_yz( const size_type rowIndex ) {
		pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns(), 0, pitch );
		//padded_ptr<value_type,pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), 0 );
		const size_type padding = pitch-number_depths()*sizeof(value_type);
		return slice_yz_type( np, number_depths(), number_columns(), padding );
		//return slice_yz_type( np, number_columns(), number_depths(), padding );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single depth.
	///
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements at the specified depth.
	///
	HOST DEVICE inline slice_xy_type get_xy( const size_type depthIndex ) {
		pointer np = allocator.address( deviceMemory.get(), 0, depthIndex, pitch );
		padded_ptr<value_type,pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), 0 );
		striding_ptr<value_type, padded_ptr<value_type,pointer,1> > sp( pp, number_depths() );
		return slice_xy_type( sp, number_columns(), number_rows() );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single column.
	///
	/// \param columnIndex the column to fix the view on
	/// \returns A view of the elements at the specified column.
	///
	HOST DEVICE inline slice_xz_type get_xz( const size_type columnIndex ) {
		pointer np = allocator.address( deviceMemory.get(), columnIndex, 0, pitch );
		const size_type padding = (pitch-number_depths()*sizeof(value_type)) + (number_columns()-1)*pitch;
		//padded_ptr<value_type,pointer,1> pp( np, number_depths(), padding, 0 );
		return slice_xz_type( np, number_depths(), number_rows(), padding );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single row.
	///
	/// \param rowIndex the row to fix the view on
	/// \returns A view of the elements at the specified row.
	///
	HOST DEVICE inline const_slice_yz_type get_yz( const size_type rowIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns(), 0, pitch );
		//padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), 0 );
		const size_type padding = pitch-number_depths()*sizeof(value_type);
		return const_slice_yz_type( np, number_depths(), number_columns(), padding );
		//return const_slice_yz_type( np, number_columns(), number_depths(), padding );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single depth.
	///
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements at the specified depth.
	///
	HOST DEVICE inline const_slice_xy_type get_xy( const size_type depthIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), 0, depthIndex, pitch );
		padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), 0 );
		striding_ptr<const value_type, padded_ptr<const value_type,const_pointer,1> > sp( pp, number_depths() );
		return const_slice_xy_type( sp, number_columns(), number_rows() );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single column.
	///
	/// \param columnIndex the column to fix the view on
	/// \returns A view of the elements at the specified column.
	///
	HOST DEVICE inline const_slice_xz_type get_xz( const size_type columnIndex ) const {
		const_pointer np = allocator.address( deviceMemory.get(), columnIndex, 0, pitch );
		const size_type padding = (pitch-number_depths()*sizeof(value_type)) + (number_columns()-1)*pitch;
		//padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), padding, 0 );
		return const_slice_xz_type( np, number_depths(), number_rows(), padding );
	}

	///
	/// \brief operator[](rowIndex) alias for get_yz(rowIndex)
	/// \param rowIndex index of the YZ-slice to isolate
	/// \returns view object for the specified row
	///
	HOST DEVICE inline slice_yz_type operator[]( const size_type rowIndex ) { return get_yz( rowIndex ); }

	///
	/// \brief operator[](rowIndex) alias for get_yz(rowIndex)
	/// \param rowIndex index of the YZ-slice to isolate
	/// \returns view object for the specified row
	///
	HOST DEVICE inline const_slice_yz_type operator[]( const size_type rowIndex ) const { return get_yz( rowIndex ); }

	///
	/// \brief Returns a reference to the element at the specified cube location.
	///
	/// This method in STL containers like vector is differentiated from operator[]
	/// because it includes range checking.  In this case, no range checking is performed,
	/// but if a thread only accesses a single element, this accessor may be slightly faster.
	/// For example:
	///
	/// \code{.cpp}
	/// // host code
	/// ecuda::cube<double> deviceCube( 100, 100, 100 );
	/// // within kernel
	/// double& value = deviceCube.at( 10, 10, 10 ); // slightly faster
	/// double& value = deviceCube[10][10][10]; // slightly slower
	/// \endcode
	///
	/// This is due to the operator[] first creating a YZ-slice view, then the second
	/// operator[] creating a view of a single row within the slice, and then finally
	/// a third access to a single column within it.  Modern compilers can be pretty
	/// crafty at seeing through these these types of situations, and it may resolve to
	/// an identical set of instructions, but the direct accessor method is included here
	/// for completeness.
	///
	/// \param rowIndex index of the row to get an element reference from
	/// \param columnIndex index of the column to get an element reference from
	/// \param depthIndex index of the depth to get an element reference from
	/// \returns reference to the specified element
	///
	DEVICE inline T& at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) { return *allocator.address( deviceMemory.get(), rowIndex*number_columns()+columnIndex, depthIndex, pitch ); }

	///
	/// This method in STL containers like vector is differentiated from operator[]
	/// because it includes range checking.  In this case, no range checking is performed,
	/// but if a thread only accesses a single element, this accessor may be slightly faster.
	/// For example:
	///
	/// \code{.cpp}
	/// // host code
	/// ecuda::cube<double> deviceCube( 100, 100, 100 );
	/// // within kernel
	/// double& value = deviceCube.at( 10, 10, 10 ); // slightly faster
	/// double& value = deviceCube[10][10][10]; // slightly slower
	/// \endcode
	///
	/// This is due to the operator[] first creating a YZ-slice view, then the second
	/// operator[] creating a view of a single row within the slice, and then finally
	/// a third access to a single column within it.  Modern compilers can be pretty
	/// crafty at seeing through these these types of situations, and it may resolve to
	/// an identical set of instructions, but the direct accessor method is included here
	/// for completeness.
	///
	/// \param rowIndex index of the row to get an element reference from
	/// \param columnIndex index of the column to get an element reference from
	/// \param depthIndex index of the depth to get an element reference from
	/// \returns reference to the specified element
	///
	DEVICE inline const T& at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) const { return *allocator.address( deviceMemory.get(), rowIndex*number_columns()+columnIndex, depthIndex, pitch ); }

	///
	/// \brief Resizes the container to have dimensions newNumberRows x newNumberColumns x newNumberDepths.
	///
	/// If the current size is greater in any dimension, the existing elements are truncated.
	///
	/// \param newNumberRows new number of rows
	/// \param newNumberColumns new number of columns
	/// \param newNumberDepths new number of depths
	/// \param value the value to initialize the new elements with (default constructed if not specified)
	///
	HOST void resize( const size_type newNumberRows, const size_type newNumberColumns, const size_type newNumberDepths, const value_type& value = value_type() ) {
		if( number_rows() == newNumberRows and number_columns() == newNumberColumns and number_depths() == newNumberDepths ) return; // no resize needed
		cube<value_type,allocator_type> newCube( newNumberRows, newNumberColumns, newNumberDepths, value );
		for( size_type i = 0; i < std::min(newNumberRows,number_rows()); ++i ) {
			for( size_type j = 0; j < std::min(newNumberColumns,number_columns()); ++j ) {
				CUDA_CALL(
					cudaMemcpy<value_type>(
						newCube.allocator.address( newCube.deviceMemory.get(), i*newCube.numberColumns+j, 0, newCube.pitch ),
						allocator.address( deviceMemory.get(), i*numberColumns+j, 0, pitch ),
						std::min(newNumberDepths,number_depths()),
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

	///
	/// \brief Assigns a given value to all elements in the container.
	///
	/// \param value the value to assign to the elements
	///
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

	/*
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
	/// \param src Container whose contents are to be assigned to this container.
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
	*/

	///
	/// \brief Copies the contents of this device cube to another container.
	///
	/// The cube is converted into a row,column-major linearized form (all
	/// depths of the first column of the first row, then the second column
	/// of the first row, ...).
	///
	template<class Container>
	HOST Container& operator>>( Container& dest ) const {
		typename Container::iterator destIter = dest.begin();
		for( size_type i = 0; i < number_rows(); ++i ) {
			for( size_type j = 0; j < number_columns(); ++j ) {
				const_depth_type depth = get_depth(i,j);
				::ecuda::copy( depth.begin(), depth.end(), destIter );
				::ecuda::advance( destIter, number_depths() );
			}
		}
		return dest;
	}

	///
	/// \brief Copies the contents of another container to this device matrix.
	///
	/// The size of the container must match the number of elements in this
	/// cube (number_rows()*number_columns()*number_depths()). The source container
	/// is assumed to be in row,column-major linear form (all depths of the first
	/// column of the first row, then the second column of the first row, ...).
	///
	/// \param src container to copy data from
	/// \throws std::length_error if number of elements in src does not match the size of this matrix
	///
	template<class Container>
	HOST cube& operator<<( const Container& src ) {
		typename Container::const_iterator srcIter = src.begin();
		typename std::iterator_traits<typename Container::const_iterator>::difference_type len = ::ecuda::distance( src.begin(), src.end() );
		if( len < 0 or static_cast<size_type>(len) != size() ) throw std::length_error( EXCEPTION_MSG("ecuda::cube::operator<<() provided with a container of non-matching size") );
		for( size_type i = 0; i < number_rows(); ++i ) {
			for( size_type j = 0; j < number_columns(); ++j ) {
				depth_type depth = get_depth(i,j);
				typename Container::const_iterator srcEnd = srcIter;
				::ecuda::advance( srcEnd, number_depths() );
				depth.assign( srcIter, srcEnd );
				srcIter = srcEnd;
			}
		}
		return *this;
	}

};

} // namespace ecuda

#endif

