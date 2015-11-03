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

//#include "config.hpp"
#include "global.hpp"
#include "allocators.hpp"
#include "matrix.hpp"
#include "memory.hpp"
#include "impl/models.hpp"
#include "type_traits.hpp"

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
class cube : private impl::device_contiguous_row_matrix< T, /*padded_ptr< T,*/ shared_ptr<T> /*>*/ > {

private:
	typedef impl::device_contiguous_row_matrix< T, /*padded_ptr< T,*/ shared_ptr<T> /*>*/ > base_type;

public:
	typedef typename base_type::value_type value_type; //!< cell data type
	typedef Alloc allocator_type; //!< allocator type
	typedef typename base_type::size_type size_type; //!< unsigned integral type
	typedef typename base_type::difference_type difference_type; //!< signed integral type
	typedef typename base_type::reference reference; //!< cell reference type
	typedef typename base_type::const_reference const_reference; //!< cell const reference type
	typedef typename base_type::pointer pointer; //!< cell pointer type
	//typedef typename pointer_traits<pointer>::const_pointer const_pointer; //!< cell const pointer type
	typedef typename make_const<pointer>::type const_pointer; //!< cell const pointer type

	typedef impl::device_sequence<           value_type, striding_ptr< value_type, padded_ptr<value_type> > > row_type; //!< cube row container type
	typedef impl::device_sequence<           value_type, striding_ptr< value_type, padded_ptr<value_type> > > column_type; //!< cube column container type
	typedef impl::device_contiguous_sequence<value_type                                                     > depth_type; //!< cube depth container type
	typedef impl::device_sequence<           const value_type, striding_ptr< const value_type, padded_ptr<const value_type> > > const_row_type; //!< cube const row container type
	typedef impl::device_sequence<           const value_type, striding_ptr< const value_type, padded_ptr<const value_type> > > const_column_type; //!< cube const column container type
	typedef impl::device_contiguous_sequence<const value_type                                                                 > const_depth_type; //!< cube const depth container type

	typedef typename base_type::iterator iterator; //!< iterator type
	typedef typename base_type::const_iterator const_iterator; //!< const iterator type
	typedef typename base_type::reverse_iterator reverse_iterator; //!< reverse iterator type
	typedef typename base_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

	typedef impl::device_matrix<                value_type, striding_ptr< value_type, padded_ptr<value_type> > > slice_xy_type; //!< xy section of a cube at a fixed depth
	typedef impl::device_contiguous_row_matrix< value_type, padded_ptr< value_type,   padded_ptr<value_type> > > slice_xz_type; //!< xz section of a cube at a fixed column
	typedef impl::device_contiguous_row_matrix< value_type, padded_ptr<value_type>                             > slice_yz_type; //!< yz section of a cube at a fixed row
	typedef impl::device_matrix<                const value_type, striding_ptr< const value_type, padded_ptr<const value_type> > > const_slice_xy_type; //!< const xy section of a cube at a fixed depth
	typedef impl::device_contiguous_row_matrix< const value_type, padded_ptr< const value_type,   padded_ptr<const value_type> > > const_slice_xz_type; //!< const xz section of a cube at a fixed column
	typedef impl::device_contiguous_row_matrix< const value_type, padded_ptr<const value_type>                                   > const_slice_yz_type; //!< const yz section of a cube at a fixed row

private:
	allocator_type allocator;
	size_type numberRows; //!< number of rows

/*
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
*/

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
	__HOST__ cube( const size_type numberRows=0, const size_type numberColumns=0, const size_type numberDepths=0, const value_type& value = value_type(), const Alloc& allocator = Alloc() ) :
		base_type( pointer(), numberRows*numberColumns, numberDepths ), numberRows(numberRows), allocator(allocator)
	{
		if( numberRows and numberColumns and numberDepths ) {

			//typename Alloc::size_type pitch;
			//typename Alloc::pointer p = get_allocator().allocate( numberDepths, numberRows*numberColumns, pitch );
			typename Alloc::pointer p = get_allocator().allocate( numberDepths, numberRows*numberColumns );
			//shared_ptr<value_type> sp( p );
			//shared_ptr<value_type> sp( pointer_traits<typename Alloc::pointer>().undress(p) );
			typedef typename std::add_pointer<value_type>::type raw_pointer_type;
			shared_ptr<value_type> sp( naked_cast<raw_pointer_type>(p) );
			//padded_ptr< value_type, shared_ptr<value_type> > pp( sp, pitch, numberDepths );
			padded_ptr< value_type, shared_ptr<value_type> > pp( sp, p.get_pitch(), p.get_width(), sp );
			base_type base( pp, numberRows*numberColumns, numberDepths );
			for( size_type i = 0; i < base.number_rows(); ++i ) {
				typename base_type::row_type row = base.get_row(i);
				ecuda::fill( row.begin(), row.end(), value );
			}
			base_type::swap( base );
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
	__HOST__ __DEVICE__ cube( const cube& src ) : base_type( src ),
		numberRows(src.numberRows),
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
	__HOST__ cube( cube&& src ) : base_type(src), numberRows(std::move(src.numberRows)), allocator(std::move(src.allocator)) {}
	#endif

	//__HOST__ __DEVICE__ virtual ~cube() {}

	///
	/// \brief Returns the allocator associated with the container.
	/// \returns The associated allocator.
	///
	__HOST__ inline allocator_type get_allocator() const { return allocator; }

/*
private:
	template<class Iterator>
	__DEVICE__ inline void assign( Iterator first, Iterator last, device_iterator_tag ) {
		for( iterator iter = begin(); iter != end() and first != last; ++iter, ++first ) *iter = *first;
	}

	template<class Iterator>
	__DEVICE__ inline void assign( Iterator first, Iterator last, contiguous_device_iterator_tag ) { assign( first, last, device_iterator_tag() ); }

	// dummy method to trick compiler, since device code will never use a non-device iterator
	template<class Iterator,class SomeOtherCategory>
	__DEVICE__ inline void assign( Iterator first, Iterator last, SomeOtherCategory ) {}
*/

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
//	/// \throws std::length_error if number of elements doesn't match the size of the cube
	///
	template<class Iterator>
	__HOST__ __DEVICE__ void assign( Iterator first, Iterator last ) {
		/// @todo reimplement range-checking
		#ifdef __CUDA_ARCH__
		ecuda::copy( first, last, begin() );
		#else
		Iterator rowLast = first;
		for( typename base_type::size_type i = 0; i < base_type::number_rows(); ++i ) {
			ecuda::advance( rowLast, base_type::number_columns() );
			typename base_type::row_type row = base_type::get_row(i);
			ecuda::copy( first, rowLast, row.begin() );
			ecuda::advance( first, base_type::number_colums() );
		}
/*
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
		*/
		#endif
	}

	///
	/// \brief Returns the number of rows in the container.
	///
	/// \returns The number of rows in the container.
	///
	__HOST__ __DEVICE__ inline size_type number_rows() const __NOEXCEPT__ { return numberRows; }

	///
	/// \brief Returns the number of columns in the container.
	///
	/// \returns The number of columns in the container.
	///
	__HOST__ __DEVICE__ inline size_type number_columns() const __NOEXCEPT__ { return base_type::number_rows()/numberRows; }

	///
	/// \brief Returns the number of depths in the container.
	///
	/// \returns The number of depths in the container.
	///
	__HOST__ __DEVICE__ inline size_type number_depths() const __NOEXCEPT__ { return base_type::number_columns(); }

	///
	/// \brief Returns the pitch of the underlying 2D device memory.
	///
	/// \returns The pitch (in bytes) of the underlying 2D device memory.
	///
//	__HOST__ __DEVICE__ inline size_type get_pitch() const __NOEXCEPT__ { return pitch; }

	///
	/// \brief Returns the number of elements in the container.
	///
	/// This is the rows x columns x depths.
	///
	/// \returns The number of elements in the container.
	///
	__HOST__ __DEVICE__ inline size_type size() const __NOEXCEPT__ { return base_type::size(); }

	///
	/// \brief Checks if the container has no elements.
	///
	/// \returns true if the container is empty, false otherwise.
	__HOST__ __DEVICE__ inline bool empty() const __NOEXCEPT__ { return !size(); }

	///
	/// \brief Returns pointer to the underlying 2D memory serving as element storage.
	///
	/// \returns Pointer to the underlying element storage.
	///
//	__HOST__ __DEVICE__ inline pointer data() __NOEXCEPT__ { return deviceMemory.get(); }

	///
	/// \brief Returns pointer to the underlying 2D memory serving as element storage.
	///
	/// \returns Pointer to the underlying element storage.
	///
//	__HOST__ __DEVICE__ inline const_pointer data() const __NOEXCEPT__ { return deviceMemory.get(); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	__HOST__ __DEVICE__ inline iterator begin() __NOEXCEPT__ { return base_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline iterator end() __NOEXCEPT__ { return base_type::end(); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	__HOST__ __DEVICE__ inline const_iterator begin() const __NOEXCEPT__ { return base_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline const_iterator end() const __NOEXCEPT__ { return base_type::end(); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	__HOST__ __DEVICE__ inline reverse_iterator rbegin() __NOEXCEPT__ { return base_type::rbegin(); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline reverse_iterator rend() __NOEXCEPT__ { return base_type::rend(); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return base_type::rbegin(); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const __NOEXCEPT__ { return base_type::rend(); }

	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ inline const_iterator cbegin() const __NOEXCEPT__ { return base_type::cbegin(); }
	__HOST__ __DEVICE__ inline const_iterator cend() const __NOEXCEPT__ { return base_type::cend(); }
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin() __NOEXCEPT__ { return base_type::crbegin(); }
	__HOST__ __DEVICE__ inline const_reverse_iterator crend() __NOEXCEPT__ { return base_type::crend(); }
	#endif

	///
	/// \brief Gets a view of the sequence of elements forming a single row.
	///
	/// \param columnIndex the column to fix the view on
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements with the specified column and depth indices.
	///
	__HOST__ __DEVICE__ inline row_type get_row( const size_type columnIndex, const size_type depthIndex ) {
		//typedef typename pointer_traits<typename base_type::pointer>::unmanaged_pointer unmanaged_pointer;
		//unmanaged_pointer ptr = pointer_traits<typename base_type::pointer>::cast_unmanaged(base_type::get_pointer());
		//ptr += columnIndex*base_type::number_rows()+depthIndex; // move pointer to row start
		//typename row_type::pointer ptr2( ptr, number_columns()*number_depths() ); // give pointer correct stride
		//return row_type( ptr2, number_rows() );
		typedef typename make_unmanaged<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += columnIndex*base_type::number_rows()+depthIndex; // move pointer to row start
		typename row_type::pointer ptr2( ptr, number_columns()+number_depths() ); // give pointer correct stride
		return row_type( ptr2, number_rows() );
		//pointer np = allocator.address( deviceMemory.get(), columnIndex, depthIndex, pitch );
		//padded_ptr<value_type,pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), depthIndex );
		//striding_ptr< value_type, padded_ptr<value_type,pointer,1> > sp( pp, number_columns()*number_depths() );
		//return row_type( sp, number_rows() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single column.
	///
	/// \param rowIndex the row to fix the view on
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements with the specified row and depth indices.
	///
	__HOST__ __DEVICE__ inline column_type get_column( const size_type rowIndex, const size_type depthIndex ) {
		//typedef typename pointer_traits<typename base_type::pointer>::unmanaged_pointer unmanaged_pointer;
		//unmanaged_pointer ptr = pointer_traits<typename base_type::pointer>::cast_unmanaged(base_type::get_pointer());
		//ptr += rowIndex*number_columns()*number_depths()+depthIndex; // move pointer to column start
		//return column_type( pointer_traits<unmanaged_pointer>().undress(ptr), number_columns() );

		typedef typename make_unmanaged<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += rowIndex*number_columns()*number_depths()+depthIndex; // move pointer to column start
		return column_type( ptr, number_columns() );

		//pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns(), depthIndex, pitch );
		//padded_ptr<value_type,pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), depthIndex );
		//striding_ptr< value_type, padded_ptr<value_type,pointer,1> > sp( pp, number_depths() );
		//return column_type( sp, number_columns() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single depth.
	///
	/// \param rowIndex the row to fix the view on
	/// \param columnIndex the column to fix the view on
	/// \returns A view of the elements with the specified row and column indices.
	///
	__HOST__ __DEVICE__ inline depth_type get_depth( const size_type rowIndex, const size_type columnIndex ) {
		//typedef typename pointer_traits<typename base_type::pointer>::unmanaged_pointer unmanaged_pointer;
		typedef typename make_unmanaged<typename base_type::pointer>::type unmanaged_pointer_type;
		//unmanaged_pointer ptr = pointer_traits<typename base_type::pointer>::cast_unmanaged(base_type::get_pointer());
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += rowIndex*number_columns()*number_depths()+columnIndex*number_rows(); // move pointer to depth start
		//return depth_type( pointer_traits<unmanaged_pointer>().undress(ptr), number_depths() );
		return depth_type( naked_cast<typename std::add_pointer<value_type>::type>(ptr), number_depths() );
		//pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns()+columnIndex, 0, pitch );
		//return depth_type( np, number_depths() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single row.
	///
	/// \param columnIndex the column to fix the view on
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements with the specified column and depth indices.
	///
	__HOST__ __DEVICE__ inline const_row_type get_row( const size_type columnIndex, const size_type depthIndex ) const {
		//typedef typename pointer_traits<const typename base_type::pointer>::unmanaged_pointer unmanaged_pointer;
		typedef typename make_unmanaged_const<typename base_type::pointer>::type unmanaged_pointer_type;
		//unmanaged_pointer ptr = pointer_traits<const typename base_type::pointer>::cast_unmanaged(base_type::get_pointer());
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += columnIndex*base_type::number_rows()+depthIndex; // move pointer to row start
		//typename const_row_type::pointer ptr2( ptr, number_columns()*number_depths() ); // give pointer correct stride
		typename const_row_type::pointer ptr2( ptr, number_columns()*number_depths() ); // give pointer correct stride
		return const_row_type( ptr2, number_rows() );
		//const_pointer np = allocator.address( deviceMemory.get(), columnIndex, depthIndex, pitch );
		//padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), depthIndex );
		//striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > sp( pp, number_columns()*number_depths() );
		//return const_row_type( sp, number_rows() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single column.
	///
	/// \param rowIndex the row to fix the view on
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements with the specified row and depth indices.
	///
	__HOST__ __DEVICE__ inline const_column_type get_column( const size_type rowIndex, const size_type depthIndex ) const {
		//typedef typename pointer_traits<const typename base_type::pointer>::unmanaged_pointer unmanaged_pointer;
		typedef typename make_unmanaged_const<typename base_type::pointer>::type unmanaged_pointer_type;
		//unmanaged_pointer ptr = pointer_traits<const typename base_type::pointer>::cast_unmanaged(base_type::get_pointer());
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += rowIndex*number_columns()*number_depths()+depthIndex; // move pointer to column start
		//return const_column_type( pointer_traits<unmanaged_pointer>().undress(ptr), number_columns() );
		return const_column_type( naked_cast<typename std::add_pointer<const value_type>::type>(ptr), number_columns() );
		//const_pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns(), depthIndex, pitch );
		//padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), depthIndex );
		//striding_ptr< const value_type, padded_ptr<const value_type,const_pointer,1> > sp( pp, number_depths() );
		//return const_column_type( sp, number_columns() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single depth.
	///
	/// \param rowIndex the row to fix the view on
	/// \param columnIndex the column to fix the view on
	/// \returns A view of the elements with the specified row and column indices.
	///
	__HOST__ __DEVICE__ inline const_depth_type get_depth( const size_type rowIndex, const size_type columnIndex ) const {
		//typedef typename pointer_traits<const typename base_type::pointer>::unmanaged_pointer unmanaged_pointer;
		typedef typename make_unmanaged_const<typename base_type::pointer>::type unmanaged_pointer_type;
		//unmanaged_pointer ptr = pointer_traits<const typename base_type::pointer>::cast_unmanaged(base_type::get_pointer());
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += rowIndex*number_columns()*number_depths()+columnIndex*number_rows(); // move pointer to depth start
		//return const_depth_type( pointer_traits<unmanaged_pointer>().undress(ptr), number_depths() );
		return const_depth_type( naked_cast<typename std::add_pointer<const value_type>::type>(ptr), number_depths() );
		//const_pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns()+columnIndex, 0, pitch );
		//return const_depth_type( np, number_depths() );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single row.
	///
	/// \param rowIndex the row to fix the view on
	/// \returns A view of the elements at the specified row.
	///
	__HOST__ __DEVICE__ inline slice_yz_type get_yz( const size_type rowIndex ) {
		//typedef typename pointer_traits<typename base_type::pointer>::unmanaged_pointer unmanaged_pointer;
		typedef typename make_unmanaged<typename base_type::pointer>::type unmanaged_pointer_type;
		//unmanaged_pointer ptr = pointer_traits<typename base_type::pointer>::cast_unmanaged(base_type::get_pointer());
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += rowIndex*number_columns()*number_depths();
		return slice_yz_type( ptr, number_columns(), number_depths() );
		//pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns(), 0, pitch );
		// //padded_ptr<value_type,pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), 0 );
		//const size_type padding = pitch-number_depths()*sizeof(value_type);
		//return slice_yz_type( np, number_depths(), number_columns(), padding );
		// //return slice_yz_type( np, number_columns(), number_depths(), padding );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single depth.
	///
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements at the specified depth.
	///
	__HOST__ __DEVICE__ inline slice_xy_type get_xy( const size_type depthIndex ) {
		//typedef typename pointer_traits<typename base_type::pointer>::unmanaged_pointer unmanaged_pointer;
		typedef typename make_unmanaged<typename base_type::pointer>::type unmanaged_pointer_type;
		//unmanaged_pointer ptr = pointer_traits<typename base_type::pointer>::cast_unmanaged(base_type::get_pointer());
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += depthIndex; // move to correct depth
		//ptr += number_depths(); // move to correct depth
		typename slice_xy_type::pointer ptr2( ptr, number_depths() ); // make pointer stride over depths
		return slice_xy_type( ptr2, number_rows(), number_columns() );
		//pointer np = allocator.address( deviceMemory.get(), 0, depthIndex, pitch );
		//padded_ptr<value_type,pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), 0 );
		//striding_ptr<value_type, padded_ptr<value_type,pointer,1> > sp( pp, number_depths() );
		//return slice_xy_type( sp, number_columns(), number_rows() );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single column.
	///
	/// \param columnIndex the column to fix the view on
	/// \returns A view of the elements at the specified column.
	///
	__HOST__ __DEVICE__ inline slice_xz_type get_xz( const size_type columnIndex ) {
		//typedef typename pointer_traits<typename base_type::pointer>::unmanaged_pointer unmanaged_pointer;
		typedef typename make_unmanaged<typename base_type::pointer>::type unmanaged_pointer_type;
		//unmanaged_pointer ptr = pointer_traits<typename base_type::pointer>::cast_unmanaged(base_type::get_pointer());
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += columnIndex*number_depths(); // move to correct column
		typename slice_xz_type::pointer ptr2( ptr, number_columns()*number_depths()*sizeof(value_type), number_columns()*number_depths() ); // make pointer skip over rest of columns
		return slice_xz_type( ptr2, number_rows(), number_depths() );
		//pointer np = allocator.address( deviceMemory.get(), columnIndex, 0, pitch );
		//const size_type padding = (pitch-number_depths()*sizeof(value_type)) + (number_columns()-1)*pitch;
		// //padded_ptr<value_type,pointer,1> pp( np, number_depths(), padding, 0 );
		//return slice_xz_type( np, number_depths(), number_rows(), padding );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single row.
	///
	/// \param rowIndex the row to fix the view on
	/// \returns A view of the elements at the specified row.
	///
	__HOST__ __DEVICE__ inline const_slice_yz_type get_yz( const size_type rowIndex ) const {
		//typedef typename pointer_traits<const typename base_type::pointer>::unmanaged_pointer unmanaged_pointer;
		typedef typename make_unmanaged_const<typename base_type::pointer>::type unmanaged_pointer_type;
		//unmanaged_pointer ptr = pointer_traits<const typename base_type::pointer>::cast_unmanaged(base_type::get_pointer());
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += rowIndex*number_columns()*number_depths();
		return const_slice_yz_type( ptr, number_columns(), number_depths() );
		//const_pointer np = allocator.address( deviceMemory.get(), rowIndex*number_columns(), 0, pitch );
		// //padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), 0 );
		//const size_type padding = pitch-number_depths()*sizeof(value_type);
		//return const_slice_yz_type( np, number_depths(), number_columns(), padding );
		// //return const_slice_yz_type( np, number_columns(), number_depths(), padding );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single depth.
	///
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements at the specified depth.
	///
	__HOST__ __DEVICE__ inline const_slice_xy_type get_xy( const size_type depthIndex ) const {
		//typedef typename pointer_traits<const typename base_type::pointer>::unmanaged_pointer unmanaged_pointer;
		typedef typename make_unmanaged_const<typename base_type::pointer>::type unmanaged_pointer_type;
		//unmanaged_pointer ptr = pointer_traits<const typename base_type::pointer>::cast_unmanaged(base_type::get_pointer());
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += number_depths(); // move to correct depth
		typename const_slice_xy_type::pointer ptr2( ptr, number_depths() ); // make pointer stride over depths
		return const_slice_xy_type( ptr2, number_rows(), number_columns() );
		//const_pointer np = allocator.address( deviceMemory.get(), 0, depthIndex, pitch );
		//padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), pitch-number_depths()*sizeof(value_type), 0 );
		//striding_ptr<const value_type, padded_ptr<const value_type,const_pointer,1> > sp( pp, number_depths() );
		//return const_slice_xy_type( sp, number_columns(), number_rows() );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single column.
	///
	/// \param columnIndex the column to fix the view on
	/// \returns A view of the elements at the specified column.
	///
	__HOST__ __DEVICE__ inline const_slice_xz_type get_xz( const size_type columnIndex ) const {
		//typedef typename pointer_traits<const typename base_type::pointer>::unmanaged_pointer unmanaged_pointer;
		typedef typename make_unmanaged_const<typename base_type::pointer>::type unmanaged_pointer_type;
		//unmanaged_pointer ptr = pointer_traits<const typename base_type::pointer>::cast_unmanaged(base_type::get_pointer());
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += columnIndex*number_depths(); // move to correct column
		typename const_slice_xz_type::pointer ptr2( ptr, number_columns()*number_depths()*sizeof(value_type), number_columns()*number_depths() ); // make pointer skip over rest of columns
		return const_slice_xz_type( ptr2, number_rows(), number_depths() );
		//const_pointer np = allocator.address( deviceMemory.get(), columnIndex, 0, pitch );
		//const size_type padding = (pitch-number_depths()*sizeof(value_type)) + (number_columns()-1)*pitch;
		// //padded_ptr<const value_type,const_pointer,1> pp( np, number_depths(), padding, 0 );
		//return const_slice_xz_type( np, number_depths(), number_rows(), padding );
	}

	///
	/// \brief operator[](rowIndex) alias for get_yz(rowIndex)
	/// \param rowIndex index of the YZ-slice to isolate
	/// \returns view object for the specified row
	///
	__HOST__ __DEVICE__ inline slice_yz_type operator[]( const size_type rowIndex ) { return get_yz( rowIndex ); }

	///
	/// \brief operator[](rowIndex) alias for get_yz(rowIndex)
	/// \param rowIndex index of the YZ-slice to isolate
	/// \returns view object for the specified row
	///
	__HOST__ __DEVICE__ inline const_slice_yz_type operator[]( const size_type rowIndex ) const { return get_yz( rowIndex ); }

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
	__DEVICE__ inline T& at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) { return base_type::at( rowIndex*number_columns()+columnIndex, depthIndex ); }

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
	__DEVICE__ inline const T& at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) const { return base_type::at( rowIndex*number_columns()+columnIndex, depthIndex ); }

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
	__HOST__ void resize( const size_type newNumberRows, const size_type newNumberColumns, const size_type newNumberDepths, const value_type& value = value_type() ) {
		if( number_rows() == newNumberRows and number_columns() == newNumberColumns and number_depths() == newNumberDepths ) return; // no resize needed
		cube newCube( newNumberRows, newNumberColumns, newNumberDepths, value, get_allocator() );
		for( size_type i = 0; i < std::min(newNumberRows,number_rows()); ++i ) {
			for( size_type j = 0; j < std::min(newNumberColumns,number_columns()); ++j ) {
				depth_type inputDepth = get_depth(i,j);
				depth_type outputDepth = newCube.get_depth(i,j);
				ecuda::copy( inputDepth.begin(), inputDepth.begin()+std::min(newNumberDepths,number_depths()), outputDepth.begin() );
			}
		}
		base_type::swap( newCube );
		numberRows = newNumberRows;
	}

	///
	/// \brief Assigns a given value to all elements in the container.
	///
	/// \param value the value to assign to the elements
	///
	__HOST__ __DEVICE__ void fill( const value_type& value ) {
		#ifdef __CUDA_ARCH__
		ecuda::fill( begin(), end(), value );
		#else
		for( typename base_type::size_type i = 0; i < base_type::number_rows(); ++i ) {
			typename base_type::row_type row = base_type::get_row(i);
			ecuda::fill( row.begin(), row.end(), value );
		}
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
	__HOST__ __DEVICE__ cube<value_type,allocator_type>& operator=( const cube<value_type,Alloc2>& src ) {
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

	/*
	///
	/// \brief Copies the contents of this device cube to another container.
	///
	/// The cube is converted into a row,column-major linearized form (all
	/// depths of the first column of the first row, then the second column
	/// of the first row, ...).
	///
	template<class Container>
	__HOST__ Container& operator>>( Container& dest ) const {
		typename Container::iterator destIter = dest.begin();
		for( typename base_type::size_type i = 0; i < base_type::number_rows(); ++i ) {
			typename base_type::const_row_type row = base_type::get_row(i);
			ecuda::copy( row.begin(), row.end(), destIter );
			ecuda::advance( destIter, base_type::number_columns() );
		}
		return dest;
	}
	*/

	/*
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
	__HOST__ cube& operator<<( const Container& src ) {
		typename Container::const_iterator srcBegin = src.begin();
		typename ecuda::iterator_traits<typename Container::const_iterator>::difference_type len = ecuda::distance( src.begin(), src.end() );
		if( len < 0 or static_cast<size_type>(len) != size() ) throw std::length_error( EXCEPTION_MSG("ecuda::cube::operator<<() provided with a container of non-matching size") );
		for( typename base_type::size_type i = 0; i < base_type::number_rows(); ++i ) {
			typename base_type::row_type row = base_type::get_row(i);
			typename Container::const_iterator srcEnd = srcBegin;
			ecuda::advance( srcEnd, base_type::number_columns() );
			ecuda::copy( srcBegin, srcEnd, row.begin() );
			srcBegin = srcEnd;
		}
		return *this;
	}
	*/

};

} // namespace ecuda

#endif

