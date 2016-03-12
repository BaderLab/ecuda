/*
Copyright (c) 2014-2016, Scott Zuyderduyn
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

#include "global.hpp"
#include "allocators.hpp"
#include "matrix.hpp"
#include "memory.hpp"
#include "type_traits.hpp"

#include "model/device_sequence.hpp"
#include "model/device_contiguous_sequence.hpp"
#include "model/device_matrix.hpp"
#include "model/device_contiguous_row_matrix.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<typename T,class Alloc> class cube_kernel_argument; // forward declaration

} // namespace impl
/// \endcond

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
/// template<typename T> __global__ void doCubeOperation( typename ecuda::cube<T>::kernel_argument cube )
/// {
///    const int dep = blockDim.x*gridDim.x; // each thread gets a different depth value
///    const int row = blockIdx.y;
///    const int col = blockIdx.z;
///    if( row < cube.number_rows() && col < cube.number_columns() && dep < cube.number_depths() ) {
///       T& value = cube(row,col,dep);
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
template< typename T, class Alloc=device_pitch_allocator<T>, class P=shared_ptr<T> >
class cube : private model::device_contiguous_row_matrix< T, /*padded_ptr< T,*/ P /*>*/ > {

private:
	typedef model::device_contiguous_row_matrix< T, /*padded_ptr< T,*/ P /*>*/ > base_type;

public:
	typedef typename base_type::value_type      value_type;      //!< cell data type
	typedef Alloc                               allocator_type;  //!< allocator type
	typedef typename base_type::size_type       size_type;       //!< unsigned integral type
	typedef typename base_type::difference_type difference_type; //!< signed integral type
	typedef typename base_type::reference       reference;       //!< cell reference type
	typedef typename base_type::const_reference const_reference; //!< cell const reference type
	typedef typename base_type::pointer         pointer;         //!< cell pointer type
	typedef typename make_const<pointer>::type  const_pointer;   //!< cell const pointer type

	typedef model::device_sequence<           value_type, striding_padded_ptr<value_type,typename ecuda::add_pointer<value_type>::type> > row_type;    //!< cube row container type
	typedef model::device_sequence<           value_type, striding_padded_ptr<value_type,typename ecuda::add_pointer<value_type>::type> > column_type; //!< cube column container type
	typedef model::device_contiguous_sequence<value_type                                                                                > depth_type;  //!< cube depth container type
	typedef model::device_sequence<           const value_type, striding_padded_ptr<const value_type,typename ecuda::add_pointer<const value_type>::type> > const_row_type;    //!< cube const row container type
	typedef model::device_sequence<           const value_type, striding_padded_ptr<const value_type,typename ecuda::add_pointer<const value_type>::type> > const_column_type; //!< cube const column container type
	typedef model::device_contiguous_sequence<const value_type                                                                                            > const_depth_type;  //!< cube const depth container type

	typedef typename base_type::iterator               iterator;               //!< iterator type
	typedef typename base_type::const_iterator         const_iterator;         //!< const iterator type
	typedef typename base_type::reverse_iterator       reverse_iterator;       //!< reverse iterator type
	typedef typename base_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

	typedef model::device_matrix<                value_type,       striding_padded_ptr<value_type,typename ecuda::add_pointer<value_type>::type> > slice_xy_type; //!< xy section of a cube at a fixed depth
	typedef model::device_contiguous_row_matrix< value_type,       typename ecuda::add_pointer<value_type>::type                                 > slice_xz_type; //!< xz section of a cube at a fixed column
	typedef model::device_contiguous_row_matrix< value_type,       typename ecuda::add_pointer<value_type>::type                                 > slice_yz_type; //!< yz section of a cube at a fixed row
	typedef model::device_matrix<                const value_type, striding_padded_ptr<const value_type,typename ecuda::add_pointer<const value_type>::type> > const_slice_xy_type; //!< xy section of a cube at a fixed depth
	typedef model::device_contiguous_row_matrix< const value_type, typename ecuda::add_pointer<const value_type>::type                                       > const_slice_xz_type; //!< const xz section of a cube at a fixed row
	typedef model::device_contiguous_row_matrix< const value_type, typename ecuda::add_pointer<const value_type>::type                                       > const_slice_yz_type; //!< const yz section of a cube at a fixed row

	typedef       impl::cube_kernel_argument<T,Alloc> kernel_argument;       //!< kernel argument type
	typedef const impl::cube_kernel_argument<T,Alloc> const_kernel_argument; //!< const kernel argument type

private:
	size_type numberRows; //!< number of rows
	allocator_type allocator; //!< device memory allocator

	template<typename U,class Alloc2,class Q> friend class cube;

protected:
	///
	/// \brief Used by the kernel_argument subclass to create a shallow copy using an unmanaged pointer.
	///
	template<typename U>
	__HOST__ __DEVICE__ cube( const cube<T,Alloc,U>& src, ecuda::true_type ) : base_type( unmanaged_cast(src.get_pointer()), src.number_rows()*src.number_columns(), src.number_depths() ), numberRows(src.numberRows), allocator(src.allocator) {}

	///
	/// \brief Used by the kernel_argument subclass to create a shallow copy using an unmanaged pointer.
	///
	__HOST__ __DEVICE__ cube& shallow_assign( const cube& other )
	{
		base_type::get_pointer() = other.get_pointer();
		allocator = other.allocator;
		numberRows = other.numberRows;
		return *this;
	}

private:
	__HOST__ void init()
	{
		if( number_rows() && number_columns() && number_depths() ) {
			typename Alloc::pointer p = get_allocator().allocate( number_depths(), number_rows()*number_columns() );
			typedef typename ecuda::add_pointer<value_type>::type raw_pointer_type;
			shared_ptr<value_type> sp( naked_cast<raw_pointer_type>(p) );
			padded_ptr< value_type, shared_ptr<value_type> > pp( sp, p.get_pitch() );
			base_type base( pp, number_rows()*number_columns(), number_depths() );
			base_type::swap( base );
		}
	}

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
	__HOST__ cube(
		const size_type numberRows=0, const size_type numberColumns=0, const size_type numberDepths=0,
		const value_type& value = value_type(),
		const Alloc& allocator = Alloc()
	) :	base_type( pointer(), numberRows*numberColumns, numberDepths ), numberRows(numberRows), allocator(allocator)
	{
		init();
		if( size() ) ecuda::fill( begin(), end(), value );
	}

	///
	/// \brief Copy constructor.
	///
	/// Constructs a cube with a copy of the contents of src.
	///
	/// \param src Another cube object of the same type and dimensions, whose contents are copied.
	///
	__HOST__ cube( const cube& src ) :
		base_type( pointer(), src.number_rows()*src.number_columns(), src.number_depths() ),
		numberRows( src.numberRows ),
		allocator( src.get_allocator() )
		//allocator( std::allocator_traits<Alloc>::select_on_container_copy_construction(src.get_allocator()) )
	{
		init();
		if( size() ) ecuda::copy( src.begin(), src.end(), begin() );
	}

	///
	/// \brief Copy constructor.
	///
	/// Constructs a cube with a copy of the contents of src.
	///
	/// \param src Another cube object of the same type and dimensions, whose contents are copied.
	/// \param alloc Allocator to use for all memory allocations of this container.
	///
	__HOST__ cube( const cube& src, const allocator_type& alloc ) :
		base_type( pointer(), src.number_rows()*src.number_columns(), src.number_depths() ),
		numberRows( src.numberRows ),
		allocator(alloc)
	{
		init();
		if( size() ) ecuda::copy( src.begin(), src.end(), begin() );
	}

	__HOST__ cube& operator=( const cube& src )
	{
		if( number_rows() != src.number_rows() || number_columns() != src.number_columns() || number_depths() != src.number_depths() ) {
			resize( src.number_rows(), src.number_columns(), src.number_depths() );
		}
		if( size() ) ecuda::copy( src.begin(), src.end(), begin() );
		return *this;
	}

	#ifdef ECUDA_CPP11_AVAILABLE
	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	///
	/// This constructor is only available if the compiler is configured to allow C++11.
	///
	/// \param src another container to be used as source to initialize the elements of the container with
	///
	__HOST__ cube( cube&& src ) : base_type(std::move(src)), numberRows(std::move(src.numberRows)), allocator(std::move(src.allocator)) {}

	__HOST__ cube& operator=( cube&& src ) {
		base_type::operator=(std::move(src));
		allocator = std::move(src.allocator);
		numberRows = std::move(src.numberRows);
		return *this;
	}
	#endif

	///
	/// \brief Returns the allocator associated with the container.
	/// \returns The associated allocator.
	///
	__HOST__ inline allocator_type get_allocator() const { return allocator; }

	///
	/// \brief Returns the number of rows in the container.
	///
	/// \returns The number of rows in the container.
	///
	__HOST__ __DEVICE__ inline size_type number_rows() const ECUDA__NOEXCEPT { return numberRows; }

	///
	/// \brief Returns the number of columns in the container.
	///
	/// \returns The number of columns in the container.
	///
	__HOST__ __DEVICE__ inline size_type number_columns() const ECUDA__NOEXCEPT { return base_type::number_rows()/numberRows; } //TODO: this costs a register in kernel code

	///
	/// \brief Returns the number of depths in the container.
	///
	/// \returns The number of depths in the container.
	///
	__HOST__ __DEVICE__ inline size_type number_depths() const ECUDA__NOEXCEPT { return base_type::number_columns(); }

	///
	/// \brief Returns the number of elements in the container.
	///
	/// This is the rows x columns x depths.
	///
	/// \returns The number of elements in the container.
	///
	__HOST__ __DEVICE__ inline size_type size() const ECUDA__NOEXCEPT { return base_type::size(); }

	///
	/// \brief Checks if the container has no elements.
	///
	/// \returns true if the container is empty, false otherwise.
	__HOST__ __DEVICE__ inline bool empty() const ECUDA__NOEXCEPT { return !size(); }

	///
	/// \brief Returns pointer to the underlying 2D memory serving as element storage.
	///
	/// \returns Pointer to the underlying element storage.
	///
	__HOST__ __DEVICE__ inline pointer data() ECUDA__NOEXCEPT { return base_type::get_pointer(); }

	///
	/// \brief Returns pointer to the underlying 2D memory serving as element storage.
	///
	/// \returns Pointer to the underlying element storage.
	///
	__HOST__ __DEVICE__ inline const_pointer data() const ECUDA__NOEXCEPT { return base_type::get_pointer(); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	__HOST__ __DEVICE__ inline iterator begin() ECUDA__NOEXCEPT { return base_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline iterator end() ECUDA__NOEXCEPT { return base_type::end(); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	__HOST__ __DEVICE__ inline const_iterator begin() const ECUDA__NOEXCEPT { return base_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline const_iterator end() const ECUDA__NOEXCEPT { return base_type::end(); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	__HOST__ __DEVICE__ inline reverse_iterator rbegin() ECUDA__NOEXCEPT { return base_type::rbegin(); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline reverse_iterator rend() ECUDA__NOEXCEPT { return base_type::rend(); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const ECUDA__NOEXCEPT { return base_type::rbegin(); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const ECUDA__NOEXCEPT { return base_type::rend(); }

	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ __DEVICE__ inline const_iterator cbegin() const    ECUDA__NOEXCEPT { return base_type::cbegin();  }
	__HOST__ __DEVICE__ inline const_iterator cend() const      ECUDA__NOEXCEPT { return base_type::cend();    }
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin() ECUDA__NOEXCEPT { return base_type::crbegin(); }
	__HOST__ __DEVICE__ inline const_reverse_iterator crend()   ECUDA__NOEXCEPT { return base_type::crend();   }
	#endif

	///
	/// \brief Gets a view of the sequence of elements forming a single row.
	///
	/// \param columnIndex the column to fix the view on
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements with the specified column and depth indices.
	///
	__HOST__ __DEVICE__ row_type get_row( const size_type columnIndex, const size_type depthIndex )
	{
		typedef typename make_unmanaged<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() ); // padded_ptr
		ptr.skip_bytes( columnIndex*ptr.get_pitch() ); // move pointer to correct column
		ptr += depthIndex; // move pointer to correct depth
		typename row_type::pointer ptr2( ptr.get(), number_columns()*ptr.get_pitch() );
		return row_type( ptr2, number_rows() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single column.
	///
	/// \param rowIndex the row to fix the view on
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements with the specified row and depth indices.
	///
	__HOST__ __DEVICE__ column_type get_column( const size_type rowIndex, const size_type depthIndex )
	{
		typedef typename make_unmanaged<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() ); // padded_ptr
		ptr.skip_bytes( rowIndex*number_columns()*ptr.get_pitch() ); // move pointer to correct column
		ptr += depthIndex; // move pointer to correct depth
		typename column_type::pointer ptr2( ptr.get(), ptr.get_pitch() );
		return column_type( ptr2, number_columns() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single depth.
	///
	/// \param rowIndex the row to fix the view on
	/// \param columnIndex the column to fix the view on
	/// \returns A view of the elements with the specified row and column indices.
	///
	__HOST__ __DEVICE__ depth_type get_depth( const size_type rowIndex, const size_type columnIndex )
	{
		typedef typename make_unmanaged<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += rowIndex*number_columns()*number_depths()+columnIndex*number_depths(); // move pointer to depth start
		return depth_type( naked_cast<typename ecuda::add_pointer<value_type>::type>(ptr), number_depths() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single row.
	///
	/// \param columnIndex the column to fix the view on
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements with the specified column and depth indices.
	///
	__HOST__ __DEVICE__ const_row_type get_row( const size_type columnIndex, const size_type depthIndex ) const
	{
		typedef typename make_unmanaged_const<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() ); // padded_ptr
		ptr.skip_bytes( columnIndex*ptr.get_pitch() ); // move pointer to correct column
		ptr += depthIndex; // move pointer to correct depth
		typename const_row_type::pointer ptr2( ptr.get(), number_columns()*ptr.get_pitch() );
		return const_row_type( ptr2, number_rows() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single column.
	///
	/// \param rowIndex the row to fix the view on
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements with the specified row and depth indices.
	///
	__HOST__ __DEVICE__ const_column_type get_column( const size_type rowIndex, const size_type depthIndex ) const
	{
		typedef typename make_unmanaged_const<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() ); // padded_ptr
		ptr.skip_bytes( rowIndex*number_columns()*ptr.get_pitch() ); // move pointer to correct column
		ptr += depthIndex; // move pointer to correct depth
		typename const_column_type::pointer ptr2( ptr.get(), ptr.get_pitch() );
		return const_column_type( ptr2, number_columns() );
	}

	///
	/// \brief Gets a view of the sequence of elements forming a single depth.
	///
	/// \param rowIndex the row to fix the view on
	/// \param columnIndex the column to fix the view on
	/// \returns A view of the elements with the specified row and column indices.
	///
	__HOST__ __DEVICE__ const_depth_type get_depth( const size_type rowIndex, const size_type columnIndex ) const
	{
		typedef typename make_unmanaged_const<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += rowIndex*number_columns()*number_depths()+columnIndex*number_depths(); // move pointer to depth start
		return const_depth_type( naked_cast<typename ecuda::add_pointer<const value_type>::type>(ptr), number_depths() );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single row.
	///
	/// \param rowIndex the row to fix the view on
	/// \returns A view of the elements at the specified row.
	///
	__HOST__ __DEVICE__ slice_yz_type get_yz( const size_type rowIndex )
	{
		typedef typename make_unmanaged<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += rowIndex*number_columns()*number_depths();
		return slice_yz_type( ptr, number_columns(), number_depths() );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single depth.
	///
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements at the specified depth.
	///
	__HOST__ __DEVICE__ slice_xy_type get_xy( const size_type depthIndex )
	{
		typedef typename make_unmanaged<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() ); // padded_ptr
		ptr += depthIndex; // move to correct depth
		typename slice_xy_type::pointer ptr2( ptr.get(), ptr.get_pitch() ); // make pointer stride over depths
		return slice_xy_type( ptr2, number_rows(), number_columns() );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single column.
	///
	/// \param columnIndex the column to fix the view on
	/// \returns A view of the elements at the specified column.
	///
	__HOST__ __DEVICE__ slice_xz_type get_xz( const size_type columnIndex )
	{
		typedef typename make_unmanaged<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() ); // padded_ptr
		typename slice_xz_type::pointer ptr2( ptr.get(), ptr.get_pitch()*number_columns() ); // extend pitch
		ptr2 += columnIndex*number_depths(); // move to correct column
		return slice_xz_type( ptr2, number_rows(), number_depths() );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single row.
	///
	/// \param rowIndex the row to fix the view on
	/// \returns A view of the elements at the specified row.
	///
	__HOST__ __DEVICE__ const_slice_yz_type get_yz( const size_type rowIndex ) const
	{
		typedef typename make_unmanaged_const<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() );
		ptr += rowIndex*number_columns()*number_depths();
		return const_slice_yz_type( ptr, number_columns(), number_depths() );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single depth.
	///
	/// \param depthIndex the depth to fix the view on
	/// \returns A view of the elements at the specified depth.
	///
	__HOST__ __DEVICE__ const_slice_xy_type get_xy( const size_type depthIndex ) const
	{
		typedef typename make_unmanaged_const<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() ); // padded_ptr
		ptr += depthIndex; // move to correct depth
		typename const_slice_xy_type::pointer ptr2( ptr.get(), ptr.get_pitch() ); // make pointer stride over depths
		return const_slice_xy_type( ptr2, number_rows(), number_columns() );
	}

	///
	/// \brief Gets a view of the matrix of elements at a single column.
	///
	/// \param columnIndex the column to fix the view on
	/// \returns A view of the elements at the specified column.
	///
	__HOST__ __DEVICE__ const_slice_xz_type get_xz( const size_type columnIndex ) const
	{
		typedef typename make_unmanaged_const<typename base_type::pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type ptr = unmanaged_cast( base_type::get_pointer() ); // padded_ptr
		typename const_slice_xz_type::pointer ptr2( ptr.get(), ptr.get_pitch()*number_columns() ); // extend pitch
		ptr2 += columnIndex*number_depths(); // move to correct column
		return const_slice_xz_type( ptr2, number_rows(), number_depths() );
	}

	///
	/// \brief Returns a reference to the element at specified row, column, and depth index, with bounds checking.
	///
	/// If the row, column, and depth are not within the range of the container, the current kernel will exit and
	/// cudaGetLastError will return cudaErrorUnknown.
	///
	/// \param rowIndex position of the row to return
	/// \param columnIndex position of the column to return
	/// \param depthIndex position of the depth to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline reference at( size_type rowIndex, size_type columnIndex, size_type depthIndex ) { return base_type::at(rowIndex*number_columns()+columnIndex,depthIndex); }

	///
	/// \brief Returns a constant reference to the element at specified row, column, and depth index, with bounds checking.
	///
	/// If the row, column, and depth are not within the range of the container, the current kernel will exit and
	/// cudaGetLastError will return cudaErrorUnknown.
	///
	/// \param rowIndex position of the row to return
	/// \param columnIndex position of the column to return
	/// \param depthIndex position of the depth to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline const_reference at( size_type rowIndex, size_type columnIndex, size_type depthIndex ) const { return base_type::at(rowIndex*number_columns()+columnIndex,depthIndex); }

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// This is identical to at() but no bounds checking is performed.
	///
	/// \param rowIndex row of the element to return
	/// \param columnIndex column of the element to return
	/// \param depthIndex depth of the element to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline reference operator()( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) { return base_type::at( rowIndex*number_columns()+columnIndex, depthIndex ); }

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// This is identical to at() but no bounds checking is performed.
	///
	/// \param rowIndex row of the element to return
	/// \param columnIndex column of the element to return
	/// \param depthIndex depth of the element to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline const_reference operator()( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) const { return base_type::at( rowIndex*number_columns()+columnIndex, depthIndex ); }

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
	/// \brief Resizes the container to have dimensions newNumberRows x newNumberColumns x newNumberDepths.
	///
	/// If the current size is greater in any dimension, the existing elements are truncated.
	///
	/// \param newNumberRows new number of rows
	/// \param newNumberColumns new number of columns
	/// \param newNumberDepths new number of depths
	/// \param value the value to initialize the new elements with (default constructed if not specified)
	///
	__HOST__ void resize( const size_type newNumberRows, const size_type newNumberColumns, const size_type newNumberDepths, const value_type& value = value_type() )
	{
		if( number_rows() == newNumberRows && number_columns() == newNumberColumns && number_depths() == newNumberDepths ) return; // no resize needed
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
	__HOST__ __DEVICE__ inline void fill( const value_type& value ) { if( !empty() ) ecuda::fill( begin(), end(), value ); }

};

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

///
/// A cube subclass that should be used as the representation of a cube within kernel code.
///
/// This achieves two objectives: 1) create a new cube object that is instantiated by creating
/// a shallow copy of the contents (so that older versions of the CUDA API that don't support
/// kernel pass-by-reference can specify containers in the function arguments), and 2) strip any
/// unnecessary data that will be useless to the kernel thus reducing register usage (in this
/// case by removing the unneeded reference-counting introduced by the internal shared_ptr).
///
template< typename T, class Alloc=device_pitch_allocator<T> >
class cube_kernel_argument : public cube<T,Alloc,typename ecuda::add_pointer<T>::type>
{

private:
	typedef cube<T,Alloc,typename ecuda::add_pointer<T>::type> base_type;

public:
	template<class P>
	__HOST__ cube_kernel_argument( const cube<T,Alloc,P>& src ) : base_type( src, ecuda::true_type() ) {}

	__HOST__ __DEVICE__ cube_kernel_argument( const cube_kernel_argument& src ) : base_type( src, ecuda::true_type() ) {}

	template<class P>
	__HOST__ cube_kernel_argument& operator=( const cube<T,Alloc,P>& src )
	{
		base_type::shallow_assign( src );
		return *this;
	}

	#ifdef ECUDA_CPP11_AVAILABLE
	cube_kernel_argument( cube_kernel_argument&& src ) : base_type(std::move(src)) {}

	cube_kernel_argument& operator=( cube_kernel_argument&& src )
	{
		base_type::operator=(std::move(src));
		return *this;
	}
	#endif

};

} // namespace impl
/// \endcond

} // namespace ecuda

#endif

