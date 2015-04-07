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
//
// An STL-like 3D structure that resides in video memory.
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
#include "models.hpp"

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
/// of CUDA when blockDim.x is limited to 512 (at the time of this writing it was 1024 in the newer versions
/// of CUDA).
///
/// Just keep in mind that the depth dimension lies in contiguous memory, the column dimension is contiguous
/// blocks of depth blocks, and the row dimension is contiguous blocks of column blocks; thus, an implementation
/// that aims to have concurrently running threads accessing depth >>> column > row will run much more efficiently.
///
template< typename T, class Alloc=device_pitch_allocator<T> >
class cube : protected matrix<T,Alloc>
{

protected:
	typedef matrix<T,Alloc> base_matrix_type; //!< shortcut typedef to make invokations to the base class

public:
	typedef typename base_matrix_type::value_type value_type; //!< element data type
	typedef typename base_matrix_type::allocator_type allocator_type; //!< allocator type
	typedef typename base_matrix_type::size_type size_type; //!< unsigned integral type
	typedef typename base_matrix_type::difference_type difference_type; //!< signed integral type
	typedef typename base_matrix_type::pointer pointer; //!< element pointer type
	typedef typename base_matrix_type::reference reference; //!< element reference type
	typedef typename base_matrix_type::const_reference const_reference; //!< element const reference type

	typedef typename base_matrix_type::iterator iterator; //!< iterator type
	typedef typename base_matrix_type::const_iterator const_iterator; //!< const iterator type
	typedef typename base_matrix_type::reverse_iterator reverse_iterator; //!< reverse iterator type
	typedef typename base_matrix_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

	typedef typename base_matrix_type::column_type row_type; //!< cube row type
	typedef typename base_matrix_type::column_type column_type; //!< cube column type
	typedef typename base_matrix_type::row_type depth_type; //!< cube depth type
	typedef typename base_matrix_type::const_column_type const_row_type; //!< cube const row type
	typedef typename base_matrix_type::const_column_type const_column_type; //!< cube const column type
	typedef typename base_matrix_type::const_row_type const_depth_type; //!< cube const depth type

	typedef __device_grid< value_type, striding_ptr<value_type,pointer>, __dimension_noncontiguous_tag, __dimension_noncontiguous_tag, __container_type_derived_tag > slice_xy_type; //!< cube xy-slice type
	typedef __device_grid< value_type, /*padded_ptr<value_type,*/pointer/*>*/,   __dimension_noncontiguous_tag, __dimension_contiguous_tag,    __container_type_derived_tag > slice_xz_type; //!< cube xz-slice type
	typedef __device_grid< value_type, pointer,                          __dimension_noncontiguous_tag, __dimension_contiguous_tag,    __container_type_derived_tag > slice_yz_type; //!< cube yz-slice type

	typedef const __device_grid< const value_type, striding_ptr<const value_type,typename __pointer_traits<pointer>::const_pointer>, __dimension_noncontiguous_tag, __dimension_noncontiguous_tag, __container_type_derived_tag > const_slice_xy_type; //!< const cube xy-slice type
	typedef const __device_grid< const value_type, /*padded_ptr<const value_type,*/typename __pointer_traits<pointer>::const_pointer/*>*/,   __dimension_noncontiguous_tag, __dimension_contiguous_tag,    __container_type_derived_tag > const_slice_xz_type; //!< const cube xz-slice type
	typedef const __device_grid< const value_type, typename __pointer_traits<pointer>::const_pointer,                                __dimension_noncontiguous_tag, __dimension_contiguous_tag,    __container_type_derived_tag > const_slice_yz_type; //!< const cube yz-slice type

private:
	// REMEMBER: numberRows, numberColumns, numberDepths and pitch altered on device memory won't be
	//           reflected on the host object. Don't allow the device to perform any operations that
	//           change their value.
	size_type numberRows; //!< number of rows

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
	HOST cube( const size_type numberRows=0, const size_type numberColumns=0, const size_type numberDepths=0, const value_type& value = value_type(), const Alloc& allocator = Alloc() ) :
		base_matrix_type( numberRows*numberColumns, numberDepths, value, allocator ),
		numberRows(numberRows)
	{
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
	HOST DEVICE cube( const cube<T,Alloc>& src ) : base_matrix_type(src), numberRows(src.numberRows) {}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	///
	/// This constructor is only available if the compiler is configured to allow C++11.
	///
	/// \param src another container to be used as source to initialize the elements of the container with
	///
	HOST cube( cube<T>&& src ) : base_matrix_type(src), numberRows(std::move(src.numberRows)) {}
	#endif

	//HOST DEVICE virtual ~cube() {}

	///
	/// \brief Returns the allocator associated with the container.
	/// \returns The associated allocator.
	///
	HOST inline allocator_type get_allocator() const { return base_matrix_type::get_allocator(); }

	///
	/// \brief Replaces the contents of the container with copies of those in the range [first,last).
	///
	/// The provided iterator can be any standard STL-style iterator type, generated from standard
	/// host containers, custom containers, or other device-bound containers from within ecuda.
	///
	/// If using naked pointers (e.g. C-style arrays) see the utility class ecuda::host_array_proxy
	/// for information on how to copy them to the device.
	///
	/// The operations required to copy the memory are determined at compile-time based on the
	/// capabilities of the iterator and whether the iterator is bound to data in contiguous memory.
	/// The underlying data is treated as though it is ordered row>column-major (depths of the same
	/// row/column are together, then columns containing each set of depths are side-by-side, and finally
	/// rows containing sets of columns are side-by-side.
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
	/// If one wishes to partially transfer data without range checking see operator<<.
	///
	/// \param first,last the range to copy the elements from
	/// \throws std::length_error thrown if the range of elements provided does not match the size of this container
	/// \throws std::domain_error thrown if this cube is represented in non-contiguous memory and cannot be written
	///                           to via iterators (should not occur in the default cube implementation)
	///
	template<class Iterator>
	HOST inline void assign( Iterator first, Iterator last ) { base_matrix_type::assign( first, last ); }

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
	HOST DEVICE inline size_type number_columns() const __NOEXCEPT__ { return number_rows() == 0 ? 0 : (base_matrix_type::number_rows()/number_rows()); }

	///
	/// \brief Returns the number of depths in the container.
	///
	/// \returns The number of depths in the container.
	///
	HOST DEVICE inline size_type number_depths() const __NOEXCEPT__ { return base_matrix_type::number_columns(); }

	///
	/// \brief Returns the number of elements in the container.
	///
	/// This is the rows x columns x depths.
	///
	/// \returns The number of elements in the container.
	///
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return base_matrix_type::size(); }

	///
	/// \brief Checks if the container has no elements.
	///
	/// \returns true if the container is empty, false otherwise.
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return base_matrix_type::empty(); }

	///
	/// \brief Returns pointer to the underlying 2D memory serving as element storage.
	///
	/// \returns Pointer to the underlying element storage.
	///
	HOST DEVICE inline pointer data() __NOEXCEPT__ { return base_matrix_type::data(); }

	///
	/// \brief Returns pointer to the underlying 2D memory serving as element storage.
	///
	/// By default, the cube uses the specialized ecuda::padded_ptr pointer that automatically
	/// deals with the alignment padding associated with 2D memory allocated by CUDA.
	///
	/// \returns Pointer to the underlying element storage.
	///
	HOST DEVICE inline const pointer data() const __NOEXCEPT__ { return base_matrix_type::data(); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return base_matrix_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return base_matrix_type::end(); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return base_matrix_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return base_matrix_type::end(); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return base_matrix_type::rbegin(); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return base_matrix_type::rend(); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return base_matrix_type::rbegin(); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return base_matrix_type::rend(); }

	///
	/// \brief Gets a proxy container containing the sequence of elements forming a single row.
	///
	/// As a proxy, any changes made to the returned container are reflected in this cube.
	///
	/// \param columnIndex the column the proxy container should contain
	/// \param depthIndex the depth the proxy container should contain
	/// \returns A proxy container containing the elements with the specified column and depth indices.
	///
	HOST DEVICE inline row_type get_row( const size_type columnIndex, const size_type depthIndex ) {
		return row_type( typename row_type::pointer( data()+static_cast<int>(columnIndex*number_depths()+depthIndex), number_columns()*number_depths() ), number_rows() );
	}

	///
	/// \brief Gets a proxy container containing the sequence of elements forming a single column.
	///
	/// As a proxy, any changes made to the returned container are reflected in this cube.
	///
	/// \param rowIndex the row the proxy container should contain
	/// \param depthIndex the depth the proxy container should contain
	/// \returns A proxy container containing the elements with the specified row and depth indices.
	///
	HOST DEVICE inline column_type get_column( const size_type rowIndex, const size_type depthIndex ) {
		return column_type( typename column_type::pointer( data()+static_cast<int>(rowIndex*number_columns()*number_depths()+depthIndex), number_depths() ), number_columns() );
	}

	///
	/// \brief Gets a proxy container containing the sequence of elements forming a single depth.
	///
	/// As a proxy, any changes made to the returned container are reflected in this cube.
	///
	/// \param rowIndex the row the proxy container should contain
	/// \param columnIndex the column the proxy container should contain
	/// \returns A proxy container containing the elements with the specified row and column indices.
	///
	HOST DEVICE inline depth_type get_depth( const size_type rowIndex, const size_type columnIndex ) {
		return base_matrix_type::get_row( rowIndex*number_columns()+columnIndex );
	}

	///
	/// \brief Gets a proxy container containing the sequence of elements forming a single row.
	///
	/// As a proxy, any changes made to the returned container are reflected in this cube.
	///
	/// \param columnIndex the column the proxy container should contain
	/// \param depthIndex the depth the proxy container should contain
	/// \returns A proxy container containing the elements with the specified column and depth indices.
	///
	HOST DEVICE inline const_row_type get_row( const size_type columnIndex, const size_type depthIndex ) const {
		return const_row_type( typename const_row_type::pointer( data()+static_cast<int>(columnIndex*number_depths()+depthIndex ), number_columns()*number_depths() ), number_rows() );
	}

	///
	/// \brief Gets a proxy container containing the sequence of elements forming a single column.
	///
	/// As a proxy, any changes made to the returned container are reflected in this cube.
	///
	/// \param rowIndex the row the proxy container should contain
	/// \param depthIndex the depth the proxy container should contain
	/// \returns A proxy container containing the elements with the specified row and depth indices.
	///
	HOST DEVICE inline const_column_type get_column( const size_type rowIndex, const size_type depthIndex ) const {
		return const_column_type( typename const_column_type::pointer( data()+static_cast<int>(rowIndex*number_columns()*number_depths()+depthIndex), number_depths() ), number_columns() );
	}

	///
	/// \brief Gets a proxy container containing the sequence of elements forming a single depth.
	///
	/// As a proxy, any changes made to the returned container are reflected in this cube.
	///
	/// \param rowIndex the row the proxy container should contain
	/// \param columnIndex the column the proxy container should contain
	/// \returns A proxy container containing the elements with the specified row and column indices.
	///
	HOST DEVICE inline const_depth_type get_depth( const size_type rowIndex, const size_type columnIndex ) const {
		return base_matrix_type::get_row( rowIndex*number_columns()+columnIndex );
	}

	///
	/// \brief Gets a proxy container containing the matrix of elements at a single row.
	///
	/// As a proxy, any changes made to the returned container are reflected in this cube.
	///
	/// \param rowIndex the row the proxy container should contain
	/// \returns A proxy container containing the elements at the specified row.
	///
	HOST DEVICE inline slice_yz_type get_yz( const size_type rowIndex ) {
		return slice_yz_type( data()+static_cast<int>(rowIndex*number_columns()*number_depths()), number_columns(), number_depths() );
	}

	///
	/// \brief Gets a proxy container containing the matrix of elements at a single column.
	///
	/// As a proxy, any changes made to the returned container are reflected in this cube.
	///
	/// \param columnIndex the column the proxy container should contain
	/// \returns A proxy container containing the elements at the specified column.
	///
	HOST DEVICE inline slice_xz_type get_xz( const size_type columnIndex ) {
		//TODO: currently a hack so that padded_ptr doesn't become parent of another padded_ptr
		//      as it won't behave correctly - need to add feature in padded_ptr to combine rather
		//      than make a hierarchy
		pointer p = data()+static_cast<int>(columnIndex*number_depths());
		typename slice_xz_type::pointer pp( p, number_depths(), p.get_pitch()*number_columns()-number_depths()*sizeof(value_type) );
		return slice_xz_type( pp, number_rows(), number_depths() );
		//return slice_xz_type( typename slice_xz_type::pointer( data()+static_cast<int>(columnIndex*number_depths()), number_depths(), (number_columns()-1)*number_depths() ), number_rows(), number_depths() );
		//pointer np = allocator.address( deviceMemory.get(), columnIndex, 0, pitch );
		//const size_type padding = (pitch-number_depths()*sizeof(value_type)) + (number_columns()-1)*pitch;
		//return slice_xz_type( np, number_depths(), number_rows(), padding );
	}

	///
	/// \brief Gets a proxy container containing the matrix of elements at a single depth.
	///
	/// As a proxy, any changes made to the returned container are reflected in this cube.
	///
	/// \param depthIndex the depth the proxy container should contain
	/// \returns A proxy container containing the elements at the specified depth.
	///
	HOST DEVICE inline slice_xy_type get_xy( const size_type depthIndex ) {
		return slice_xy_type( typename slice_xy_type::pointer( data()+static_cast<int>(depthIndex), number_depths() ), number_rows(), number_columns() );
	}

	///
	/// \brief Gets a proxy container containing the matrix of elements at a single row.
	///
	/// As a proxy, any changes made to the returned container are reflected in this cube.
	///
	/// \param rowIndex the row the proxy container should contain
	/// \returns A proxy container containing the elements at the specified row.
	///
	HOST DEVICE inline const_slice_yz_type get_yz( const size_type rowIndex ) const {
		return const_slice_yz_type( data()+static_cast<int>(rowIndex*number_columns()*number_depths()), number_columns(), number_depths() );
	}

	///
	/// \brief Gets a proxy container containing the matrix of elements at a single column.
	///
	/// As a proxy, any changes made to the returned container are reflected in this cube.
	///
	/// \param columnIndex the column the proxy container should contain
	/// \returns A proxy container containing the elements at the specified column.
	///
	HOST DEVICE inline const_slice_xz_type get_xz( const size_type columnIndex ) const {
		//TODO: currently a hack so that padded_ptr doesn't become parent of another padded_ptr
		//      as it won't behave correctly - need to add feature in padded_ptr to combine rather
		//      than make a hierarchy
		pointer p = data()+static_cast<int>(columnIndex*number_depths());
		typename slice_xz_type::pointer pp( p, number_depths(), p.get_pitch()*number_columns()-number_depths()*sizeof(value_type) );
		return slice_xz_type( pp, number_rows(), number_depths() );
		//return const_slice_xz_type( typename const_slice_xy_type::pointer( data()+static_cast<int>(columnIndex*number_depths()), number_depths(), (number_columns()-1)*number_depths() ), number_rows(), number_depths() );
		//const_pointer np = allocator.address( deviceMemory.get(), columnIndex, 0, pitch );
		//const size_type padding = (pitch-number_depths()*sizeof(value_type)) + (number_columns()-1)*pitch;
		//return const_slice_xz_type( np, number_depths(), number_rows(), padding );
	}

	///
	/// \brief Gets a proxy container containing the matrix of elements at a single depth.
	///
	/// As a proxy, any changes made to the returned container are reflected in this cube.
	///
	/// \param depthIndex the depth the proxy container should contain
	/// \returns A proxy container containing the elements at the specified depth.
	///
	HOST DEVICE inline const_slice_xy_type get_xy( const size_type depthIndex ) const {
		return const_slice_xy_type( typename const_slice_xy_type::pointer( data()+static_cast<int>(depthIndex), number_depths() ), number_rows(), number_columns() );
	}

	///
	/// \brief operator[](rowIndex) alias for get_yz(rowIndex)
	/// \param rowIndex index of the YZ-slice to isolate
	/// \returns a proxy container containing the elements of the specified YZ-slice
	///
	HOST DEVICE inline slice_yz_type operator[]( const size_type rowIndex ) { return get_yz( rowIndex ); }

	///
	/// \brief operator[](rowIndex) alias for get_yz(rowIndex)
	/// \param rowIndex index of the YZ-slice to isolate
	/// \returns a proxy container containing the elements of the specified YZ-slice
	///
	HOST DEVICE inline const_slice_yz_type operator[]( const size_type rowIndex ) const { return get_yz( rowIndex ); }

	///
	/// \brief Returns a reference to the element at the specified cube location.
	///
	/// This method in STL containers like vector is differentiated from operator[]
	/// because the former includes range checking.  In this case, no range checking is performed,
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
	/// crafty at seeing through these these types of situations (as the logic produced
	/// as the [][][] operation is unrolled should resolve to essentially this method),
	/// but the direct accessor method is included here for completeness.
	///
	/// However, remember that if a thread is accessing some related range of elements
	/// it will be faster to use operator[] or one of the other methods such as
	/// get_column(), get_yz(), etc. in whatever fashion makes sense for the task.
	///
	/// \param rowIndex index of the row to get an element reference from
	/// \param columnIndex index of the column to get an element reference from
	/// \param depthIndex index of the depth to get an element reference from
	/// \returns reference to the specified element
	///
	DEVICE inline reference at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) { return base_matrix_type::at( rowIndex*number_columns()+columnIndex, depthIndex ); }

	///
	/// \brief Returns a reference to the element at the specified cube location.
	///
	/// This method in STL containers like vector is differentiated from operator[]
	/// because the former includes range checking.  In this case, no range checking is performed,
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
	/// crafty at seeing through these these types of situations (as the logic produced
	/// as the [][][] operation is unrolled should resolve to essentially this method),
	/// but the direct accessor method is included here for completeness.
	///
	/// However, remember that if a thread is accessing some related range of elements
	/// it will be faster to use operator[] or one of the other methods such as
	/// get_column(), get_yz(), etc. in whatever fashion makes sense for the task.
	///
	/// \param rowIndex index of the row to get an element reference from
	/// \param columnIndex index of the column to get an element reference from
	/// \param depthIndex index of the depth to get an element reference from
	/// \returns reference to the specified element
	///
	DEVICE inline const_reference at( const size_type rowIndex, const size_type columnIndex, const size_type depthIndex ) const { return base_matrix_type::at( rowIndex*number_columns()+columnIndex, depthIndex ); }

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
		base_matrix_type::resize( newNumberRows*newNumberColumns, newNumberDepths, value );
		numberRows = newNumberRows;
	}

	///
	/// \brief Assigns a given value to all elements in the container.
	///
	/// \param value the value to assign to the elements
	///
	HOST DEVICE inline void fill( const value_type& value ) { base_matrix_type::fill( value ); }

	///
	/// \brief Copies the contents of this device array to another container.
	///
	/// This method is designed to be as flexible as possible so that any container
	/// that has the standard begin() and end() methods implemented should work.
	///
	/// The underlying architecture attempts to identify if the target container
	/// is in contiguous memory or not, or if the target is another ecuda device-bound
	/// container and decides at compile-time how the transfer should be optimally
	/// performed.
	///
	/// The method will attempt to assign all elements to the target container.
	/// It is the responsibility of the caller to make sure there is enough
	/// space allocated to hold them.  Failure to do so will result in undefined
	/// behaviour.
	///
	template<class Container>
	HOST const cube& operator>>( Container& container ) const {
		base_matrix_type::operator>>( container );
		return *this;
	}

	///
	/// \brief Copies the contents of another container into this device cube.
	///
	/// This method is designed to be as flexible as possible so that any container
	/// that has the standard begin() and end() methods implemented should work.
	///
	/// The underlying architecture attempts to identify if the source container
	/// is in contiguous memory or not, or if the target is another ecuda device-bound
	/// container and decides at compile-time how the transfer should be optimally
	/// performed.
	///
	/// Note that the number of elements in the other container do not have to match
	/// the size of this cube.  If there are too few elements, the cube will only
	/// be partially filled, if there are too many elements only as many elements as
	/// needed to fill this cube will be used.
	///
	template<class Container>
	HOST cube<T,N>& operator<<( const Container& container ) {
		base_matrix_type::operator<<( container );
		return *this;
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
	/// \param src Container whose contents are to be assigned to this container.
	/// \return A reference to this container.
	///
	template<class Alloc2>
	HOST DEVICE cube<value_type,allocator_type>& operator=( const cube<value_type,Alloc2>& src ) {
		base_matrix_type::operator=( src );
		numberRows = src.number_rows();
		return *this;
	}

};

} // namespace ecuda

#endif

