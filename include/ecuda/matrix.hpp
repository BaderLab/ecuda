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
// matrix.hpp
// An STL-like structure that resides in video memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MATRIX_HPP
#define ECUDA_MATRIX_HPP

#ifdef __CPP11_SUPPORTED__
#include <type_traits>
#endif
#include <vector>

#include "config.hpp"
#include "global.hpp"
#include "allocators.hpp"
#include "apiwrappers.hpp"
#include "device_ptr.hpp"
#include "models.hpp"
#include "padded_ptr.hpp"
#include "striding_ptr.hpp"
#include "vector.hpp"


namespace ecuda {

///
/// \brief A resizable matrix stored in device memory.
///
/// A matrix is defined as a 2D structure of dimensions rows*columns.  The default implementation
/// uses pitched memory where a 2D block of video memory is allocated with width=columns and height=rows.
/// Pitched memory is aligned in a device-dependent manner so that calls to individual elements can be
/// threaded more efficiently (i.e. minimizing the number of read operations required to supply data to
/// multiple threads). Consult the CUDA API documentation for a more verbose explanation.
///
/// Memory use can be conceptualized as:
/// \code
///       |- columns -|
///       |---- pitch -----|
///    _  +-----------+----+
///   |   |           |xxxx| x = allocated but not used, just padding to
///   |   |           |xxxx|     enforce an efficient memory alignment
///  rows |           |xxxx|
///   |   |           |xxxx|
///   |_  +-----------+----+
/// \endcode
///
/// As a result, it is highly desirable for threading to utilize a column-wise orientation.
/// For example, a good kernel to perform an operation on the elements of a matrix might be:
///
/// \code{.cpp}
/// template<typename T> __global__ void doMatrixOperation( ecuda::matrix<T> matrix ) {
///    const int row = blockIdx.x;
///    const int col = blockDim.y*gridDim.y; // each thread gets a different column value
///    if( row < matrix.number_rows() and col < matrix.number_columns() ) {
///       T& value = matrix[row][col];
///       // ... do work on value
///    }
/// }
/// \endcode
///
/// This could be called from host code like:
/// \code{.cpp}
/// ecuda::matrix<double> matrix( 100, 1000 );
/// // ... fill matrix with data
/// dim3 grid( 100, 1 ), block( 1, 1000 );
/// doMatrixOperation<<<grid,block>>>( matrix );
/// \endcode
///
/// Unfortunately, CUDA solutions are very problem specific, so there is no generally applicable example for
/// specifying how thread blocks should be defined.  The size of the matrix, hardware limitations, CUDA API
/// limitations, etc. all play a part.  For example, the above implementation won't work in earlier versions
/// of CUDA since blockDim.y is limited to 512.
///
/// Just keep in mind that the column dimension lies in contiguous memory, and the row dimension is contiguous
/// blocks of columns; thus, an implementation that aims to have concurrently running threads accessing
/// column >>>> row will run much more efficiently.
///
/// Matrix iterators (via begin(),end(),rbegin(),rend()) and lexicographical comparisons traverse the matrix
/// linearly in row-major fashion (i.e. each column of the first row is traversed, then each column of the
/// next row, and so on...).
///
template< typename T, class Alloc=device_pitch_allocator<T> >
class matrix :
	protected __device_grid<
		T,
		device_ptr<
			T,
			#ifdef __CPP11_SUPPORTED__
			typename std::allocator_traits<Alloc>::pointer
			#else
			typename Alloc::pointer
			#endif
		>,
		__dimension_noncontiguous_tag,
		__dimension_contiguous_tag,
		__container_type_base_tag
	>
{

protected:
	typedef __device_grid<
		T,
		device_ptr<
			T,
			#ifdef __CPP11_SUPPORTED__
			typename std::allocator_traits<Alloc>::pointer
			#else
			typename Alloc::pointer
			#endif
		>,
		__dimension_noncontiguous_tag,
		__dimension_contiguous_tag,
		__container_type_base_tag
	> base_container_type;

	typedef device_ptr<
		T,
		#ifdef __CPP11_SUPPORTED__
		typename std::allocator_traits<Alloc>::pointer
		#else
		typename Alloc::pointer
		#endif
	> managed_pointer;

public:
	typedef typename base_container_type::value_type value_type; //!< element data type
	typedef Alloc allocator_type; //!< allocator type
	typedef typename base_container_type::size_type size_type; //!< unsigned integral type
	typedef typename base_container_type::difference_type difference_type; //!< signed integral type
	typedef typename base_container_type::pointer pointer; //!< element pointer type
	typedef typename base_container_type::reference reference; //!< element reference type
	typedef typename base_container_type::const_reference const_reference; //!< element const reference type

	typedef typename base_container_type::iterator iterator; //!< iterator type
	typedef typename base_container_type::const_iterator const_iterator; //!< const iterator type
	typedef typename base_container_type::reverse_iterator reverse_iterator; //!< reverse iterator type
	typedef typename base_container_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

	typedef typename base_container_type::row_type row_type;
	typedef typename base_container_type::column_type column_type;
	typedef typename base_container_type::const_row_type const_row_type;
	typedef typename base_container_type::const_column_type const_column_type;

private:
	allocator_type allocator;

public:
	///
	/// \brief Constructs a matrix with dimensions numberRows x numberColumns filled with copies of elements with value value.
	/// \param numberRows number of rows (default: 0)
	/// \param numberColumns number of columns (default: 0)
	/// \param value the value to initialize elements of the matrix with (default: T())
	/// \param allocator allocator to use for all memory allocations of this container
	///        (does not normally need to be specified, by default the internal ecuda pitched memory allocator)
	///
	HOST matrix( const size_type numberRows=0, const size_type numberColumns=0, const T& value = T(), const Alloc& allocator = Alloc() ) :
		base_container_type( managed_pointer( allocator.allocate(numberColumns,numberRows) ), numberRows, numberColumns ),
		allocator(allocator)
	{
		base_container_type::fill( value );
	}

	///
	/// \brief Constructs a matrix with a shallow copy of each of the elements in src.
	///
	/// Be careful to note that a shallow copy means that only the pointer to the device memory
	/// that holds the elements is copied in the newly constructed container. This allows
	/// containers to be passed-by-value to kernel functions with minimal overhead. If a deep copy
	/// of the container is required in host code, use the assignment operator. For example:
	///
	/// \code{.cpp}
	/// ecuda::matrix<int> matrix( 5, 10 ); // create a matrix of dimensions 5x10 filled with zeroes
	/// ecuda::matrix<int> newMatrix( matrix ); // shallow copy (changes to newCube reflected in cube)
	/// ecuda::matrix<int> newMatrix; newMatrix = matrix; // deep copy (new device memory allocated and contents of cube copied there)
	/// \endcode
	///
	/// \param src Another matrix object of the same type, whose contents are copied.
	///
	HOST DEVICE matrix( const matrix<T,Alloc>& src ) : base_container_type(src), allocator(src.allocator) {}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	///
	/// This operator is only available if the compiler is configured to allow C++11.
	///
	/// \param src another container to be used as source to initialize the elements of the container with
	///
	HOST matrix( matrix<T>&& src ) : base_container_type(src), allocator(std::move(src.allocator)) {}
	#endif

	//HOST DEVICE virtual ~matrix() {}

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return base_container_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return base_container_type::end(); }

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return base_container_type::begin(); }

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return base_container_type::end(); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return base_container_type::rbegin(); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return base_container_type::rend(); }

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return base_container_type::rbegin(); }

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return base_container_type::rend(); }

	///
	/// \brief Returns the number of elements in the container (numberRows*numberColumns).
	///
	/// \returns The number of elements in the container.
	///
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return base_container_type::size(); }

	///
	/// \brief Returns the maximum number of elements the container is able to hold due to system
	///        or library implementation limitations.
	///
	/// \returns Maximum number of elements.
	///
	HOST DEVICE __CONSTEXPR__ inline size_type max_size() const __NOEXCEPT__ { return std::numeric_limits<size_type>::max(); }

	///
	/// \brief Returns the number of rows in the container.
	///
	/// \returns The number of rows in the container.
	///
	HOST DEVICE inline size_type number_rows() const __NOEXCEPT__ { return base_container_type::number_rows(); }

	///
	/// \brief Returns the number of columns in the container.
	///
	/// \returns The number of columns in the container.
	///
	HOST DEVICE inline size_type number_columns() const __NOEXCEPT__ { return base_container_type::number_columns(); }

	/*
	///
	/// \brief Returns the pitch of the underlying 2D device memory.
	///
	/// \returns THe pitch of the underlying 2D device memory (in bytes).
	///
	HOST DEVICE inline size_type get_pitch() const __NOEXCEPT__ { return pitch; }
	*/

	///
	/// \brief Resizes the container to have dimensions newNumberRows x newNumberColumns.
	///
	/// If the current size is greater in either or both dimensions, the existing elements are truncated.
	///
	/// \param newNumberRows new number of rows
	/// \param newNumberColumns new number of columns
	/// \param value the value to initialize the new elements with (default constructed if not specified)
	///
	HOST void resize( const size_type newNumberRows, const size_type newNumberColumns, const value_type& value = value_type() ) {
		if( number_rows() == newNumberRows and number_columns() == newNumberColumns ) return; // no resize needed
		managed_pointer newMemory( allocator.allocate(newNumberColumns,newNumberRows) );
		base_container_type newContainer( newMemory, newNumberRows, newNumberColumns );
		newContainer.fill( value );
		for( size_type i = 0; i < std::min(number_rows(),newContainer.number_rows()); ++i ) {
			typename base_container_type::row_type row = newContainer.get_row(i);
			row.copy_range_from( begin(), begin()+(std::min(number_columns(),newContainer.number_columns())), row.begin() );
		}
		base_container_type::operator=( newContainer );
	}

	///
	/// \brief Checks if the container has no elements.
	///
	/// \returns true if the container is empty, false otherwise.
	///
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return !number_rows() or !number_columns(); }

	///
	/// \brief Gets a proxy container containing the sequence of elements forming a single row.
	///
	/// As a proxy, any changes made to the returned container are reflected in this matrix.
	///
	/// \param rowIndex the row the proxy container should contain
	/// \returns A proxy container containing the elements with the specified row index.
	///
	HOST DEVICE inline row_type get_row( const size_type rowIndex ) { return base_container_type::get_row(rowIndex); }

	///
	/// \brief Gets a proxy container containing the sequence of elements forming a single row.
	///
	/// As a proxy, any changes made to the returned container are reflected in this matrix.
	///
	/// \param rowIndex the row the proxy container should contain
	/// \returns A proxy container containing the elements with the specified row index.
	///
	HOST DEVICE inline const_row_type get_row( const size_type rowIndex ) const { return base_container_type::get_row(rowIndex); }

	///
	/// \brief Gets a proxy container containing the sequence of elements forming a single column.
	///
	/// As a proxy, any changes made to the returned container are reflected in this matrix.
	///
	/// \param columnIndex the row the proxy container should contain
	/// \returns A proxy container containing the elements with the specified column index.
	///
	HOST DEVICE inline column_type get_column( const size_type columnIndex ) { return base_container_type::get_column(columnIndex); }

	///
	/// \brief Gets a proxy container containing the sequence of elements forming a single column.
	///
	/// As a proxy, any changes made to the returned container are reflected in this matrix.
	///
	/// \param columnIndex the row the proxy container should contain
	/// \returns A proxy container containing the elements with the specified column index.
	///
	HOST DEVICE inline const_column_type get_column( const size_type columnIndex ) const { return base_container_type::get_column(columnIndex); }

	///
	/// \brief operator[](rowIndex) alias for get_row(rowIndex)
	/// \param rowIndex index of the row to isolate
	/// \returns a proxy container containing the elements of the specified row
	///
	HOST DEVICE inline row_type operator[]( const size_type rowIndex ) { return base_container_type::operator[](rowIndex); }

	///
	/// \brief operator[](rowIndex) alias for get_row(rowIndex)
	/// \param rowIndex index of the row to isolate
	/// \returns a proxy container containing the elements of the specified row
	///
	HOST DEVICE inline const_row_type operator[]( const size_type rowIndex ) const { return base_container_type::operator[](rowIndex); }

	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// This is effectively the element at position (0,0). Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
	DEVICE inline reference front() { return base_container_type::at(0,0); }

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// This is effectively the element at position (numberRows-1,numberColumns-1). Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
	DEVICE inline reference back() { return base_container_type::at(number_rows()-1,number_columns()-1); }

	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// This is effectively the element at position (0,0). Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
	DEVICE inline const_reference front() const { return base_container_type::at(0,0); }

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// This is effectively the element at position (numberRows-1,numberColumns-1). Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
	DEVICE inline const_reference back() const { return base_container_type::at(number_rows()-1,number_columns()-1); }

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(),data()+size()) is always a valid
	/// range, even if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
	HOST DEVICE inline pointer data() __NOEXCEPT__ { return base_container_type::data(); }

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(),data()+size()) is always a valid
	/// range, even if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
	HOST DEVICE inline const pointer data() const __NOEXCEPT__ { return base_container_type::data(); }

	///
	/// \brief Replaces the contents of the container.
	/// \param newNumberRows new number of rows
	/// \param newNumberColumns new number of columns
	/// \param value the value to initialize elements of the container with
	///
	HOST void assign( size_type newNumberRows, size_type newNumberColumns, const value_type& value = value_type() ) {
		managed_pointer newMemory( allocator.allocate(newNumberColumns,newNumberRows) );
		base_container_type newContainer( newMemory, newNumberRows, newNumberColumns );
		newContainer.fill( value );
		base_container_type::operator=( newContainer );
	}

private:
	template<class Iterator>
	HOST void assign( Iterator first, Iterator last, contiguous_device_iterator_tag ) {
		const typename std::iterator_traits<Iterator>::difference_type n = last-first;
		if( n < 0 or static_cast<size_type>(n) != size() ) throw std::length_error( EXCEPTION_MSG("ecuda::matrix::assign range of first,last does not match matrix size" ) );
		for( size_type i = 0; i < number_rows(); ++i, first += number_columns() ) {
			row_type row = get_row(i);
			row.copy_range_from( first, first+number_columns(), row.begin() );
		}
	}

	template<class Iterator> HOST inline void assign( Iterator first, Iterator last, std::random_access_iterator_tag ) {
		assign( first, last, contiguous_device_iterator_tag() );
	}

	template<class Iterator> HOST void assign( Iterator first, Iterator last, std::bidirectional_iterator_tag ) {
		std::vector< value_type, host_allocator<value_type> > v( first, last );
		if( v.size() != size() ) throw std::length_error( EXCEPTION_MSG("ecuda::matrix::assign range of first,last does not match matrix size" ) );
		typename std::vector< value_type, host_allocator<value_type> >::const_iterator src = v.begin();
		for( size_type i = 0; i < number_rows(); ++i, src += number_columns() ) {
			row_type row = get_row(i);
			row.copy_range_from( src, src+number_columns(), row.begin() );
		}
	}

	template<class Iterator> HOST inline void assign( Iterator first, Iterator last, device_iterator_tag ) {
		throw cuda_error( cudaErrorInvalidDevicePointer, EXCEPTION_MSG("ecuda::matrix::assign() cannot assign from non-contiguous device elements") );
	}

	template<class Iterator> HOST inline void assign( Iterator first, Iterator last, std::forward_iterator_tag ) { assign( first, last, std::bidirectional_iterator_tag() ); }
	template<class Iterator> HOST inline void assign( Iterator first, Iterator last, std::input_iterator_tag ) { assign( first, last, std::bidirectional_iterator_tag() ); }

public:

	///
	/// \brief Replaces the contents of the container with copies of those in the range [begin,end).
	/// \throws std::length_error if the number of elements in the range [begin,end) does not match the number of elements in this container
	/// \param first,last the range to copy the elements from
	///

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
	HOST inline void assign( Iterator first, Iterator last ) { assign( first, last, typename std::iterator_traits<Iterator>::iterator_category() );	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Replaces the contents of the container with copies of those in the initializer list.
	/// \throws std::length_error if the number of elements in the initializer list does not match the number of elements in this container
	/// \param il initializer list to initialize the elements of the container with
	///
	HOST inline void assign( std::initializer_list<T> il ) { assign( il.begin(), il.end() ); }
	#endif

	///
	/// \brief Assigns a given value to all elements in the container.
	///
	/// \param value the value to assign to the elements
	///
	HOST DEVICE inline void fill( const value_type& value ) { base_container_type::fill( value ); }

	///
	/// \brief Exchanges the contents of the container with those of the other.
	///
	/// Does not invoke any move, copy, or swap operations on individual elements. All iterators
	/// and references remain valid. The past-the-end iterator is invalidated.
	///
	/// Although this can be called from both the host and device, a call from the
	/// device only swaps the contents of the containers in the calling thread only.
	///
	/// \param other container to exchange the contents with
	///
	HOST DEVICE inline void swap( matrix& other ) { base_container_type::swap( other ); }

	///
	/// \brief Returns the allocator associated with the container.
	/// \returns The associated allocator.
	///
	HOST inline allocator_type get_allocator() const { return allocator; }

	///
	/// \brief Checks if the contents of two matrices are equal.
	///
	/// That is, whether number_rows() == other.number_rows(), number_columns() == other.number_columns()
	/// and each element in the this matrix compares equal to the other matrix at the same position.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents are equal, false otherwise
	///
	template<class Alloc2>
	HOST DEVICE inline bool operator==( const matrix<value_type,Alloc2>& other ) const { return base_container_type::operator==( other ); }

	///
	/// \brief Checks if the contents of two matrices are not equal.
	///
	/// That is, whether number_rows() != other.number_rows(), number_columns() != other.number_columns(),
	/// or whether any element in the this matrix does not compare equal to the element in the other matrix
	/// at the same position.
	///
	/// \param other container to compare contents with
	/// \returns true if the contents are not equal, false otherwise
	///
	template<class Alloc2>
	HOST DEVICE inline bool operator!=( const matrix<value_type,Alloc2>& other ) const { return base_container_type::operator!=( other ); }

	///
	/// \brief Compares the contents of two matrices lexicographically.
	///
	/// The order that elements are compared corresponds to their linearized layout (i.e. each column of the first row
	/// is compared, then each column of the next row, and so on...).
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this matrix are lexicographically less than the other matrix, false otherwise
	///
	template<class Alloc2>
	HOST DEVICE inline bool operator<( const matrix<value_type,Alloc2>& other ) const { return base_container_type::operator<( other ); }

	///
	/// \brief Compares the contents of two matrices lexicographically.
	///
	/// The order that elements are compared corresponds to their linearized layout (i.e. each column of the first row
	/// is compared, then each column of the next row, and so on...).
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this matrix are lexicographically greater than the other matrix, false otherwise
	///
	template<class Alloc2>
	HOST DEVICE bool operator>( const matrix<value_type,Alloc2>& other ) const { return base_container_type::operator>( other ); }

	///
	/// \brief Compares the contents of two matrices lexicographically.
	///
	/// The order that elements are compared corresponds to their linearized layout (i.e. each column of the first row
	/// is compared, then each column of the next row, and so on...).
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this matrix are lexicographically less than or equal to the other matrix, false otherwise
	///
	template<class Alloc2>
	HOST DEVICE inline bool operator<=( const matrix<value_type,Alloc2>& other ) const { return base_container_type::operator<=( other ); }

	///
	/// \brief Compares the contents of two matrices lexicographically.
	///
	/// The order that elements are compared corresponds to their linearized layout (i.e. each column of the first row
	/// is compared, then each column of the next row, and so on...).
	///
	/// \param other container to compare contents with
	/// \returns true if the contents of this matrix are lexicographically greater than or equal to the other matrix, false otherwise
	///
	template<class Alloc2>
	HOST DEVICE inline bool operator>=( const matrix<value_type,Alloc2>& other ) const { return base_container_type::operator>=( other ); }

	///
	/// \brief Returns a reference to the element at the specified matrix location.
	///
	/// This method in STL containers like vector is differentiated from operator[]
	/// because it includes range checking.  In this case, no range checking is performed,
	/// but if a thread only accesses a single element, this accessor may be slightly faster.
	/// For example:
	///
	/// \code{.cpp}
	/// // host code
	/// ecuda::matrix<double> deviceMatrix( 100, 100 );
	/// // within kernel
	/// double& value = deviceMatrix.at( 10, 10 ); // slightly faster
	/// double& value = deviceMatrix[10][10]; // slightly slower
	/// \endcode
	///
	/// This is due to the operator[] first creating a row view, and then performing an
	/// additional access to a column within it.  Modern compilers can be pretty crafty
	/// at seeing through these these types of situations, and it may resolve to an
	/// identical set of instructions, but the direct accessor method is included here
	/// for completeness.
	///
	/// \param rowIndex index of the row to get an element reference from
	/// \param columnIndex index of the column to get an element reference from
	/// \returns reference to the specified element
	///
	DEVICE inline reference at( const size_type rowIndex, const size_type columnIndex ) { return base_container_type::at(rowIndex,columnIndex); }

	///
	/// \brief Returns a reference to the element at the specified matrix location.
	///
	/// This method in STL containers like vector is differentiated from operator[]
	/// because it includes range checking.  In this case, no range checking is performed,
	/// but if a thread only accesses a single element, this accessor may be slightly faster.
	/// For example:
	///
	/// \code{.cpp}
	/// // host code
	/// ecuda::matrix<double> deviceMatrix( 100, 100 );
	/// // within kernel
	/// double& value = deviceMatrix.at( 10, 10 ); // slightly faster
	/// double& value = deviceMatrix[10][10]; // slightly slower
	/// \endcode
	///
	/// This is due to the operator[] first creating a row view, and then performing an
	/// additional access to a column within it.  Modern compilers can be pretty crafty
	/// at seeing through these these types of situations, and it may resolve to an
	/// identical set of instructions, but the direct accessor method is included here
	/// for completeness.
	///
	/// \param rowIndex index of the row to get an element reference from
	/// \param columnIndex index of the column to get an element reference from
	/// \returns reference to the specified element
	///
	DEVICE inline const_reference at( const size_type rowIndex, const size_type columnIndex ) const { return base_container_type::at(rowIndex,columnIndex); }

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
	HOST DEVICE matrix<value_type,allocator_type>& operator=( const matrix<value_type,Alloc2>& src ) {
		base_container_type::operator=( src );
		return *this;
	}

	///
	/// \brief Copies the contents of this device matrix to another container.
	///
	/// The matrix is converted into a row-major linearized form (all columns
	/// of the first row, then all columns of the second row, ...).
	///
	template<class Container>
	HOST const matrix& operator>>( Container& container ) const {
		base_container_type::operator>>( container );
		return *this;
	}

	///
	/// \brief Copies the contents of another container to this device matrix.
	///
	/// The size of the other container must match the number of elements in this
	/// matrix (number_rows()*number_columns()). The container is assumed to
	/// be in row-major lineared form (all columns of the first row, then all
	/// columns of the second row, ...).
	///
	template<class Container>
	HOST matrix<T,Alloc>& operator<<( const Container& container ) {
		assign( container.begin(), container.end() );
		return *this;
	}

};


///
/// \brief Copies some or all of a source matrix to a destination matrix.
///
/// The subset of the source matrix can be specified by the offsetRow and offsetColumn parameters
/// as well as the size of the destination matrix. If the destination matrix is larger than needed
/// in either dimension the extra elements remain unaltered.
///
/// For example, to copy a subset of a matrix:
/// \code{.cpp}
/// ecuda::matrix<int> src( 100, 100, 99 ); // fill 100x100 matrix with the number 99
/// ecuda::matrix<int> dest( 20, 10 ); // fill 20x10 matrix with zeros
/// // copies the elements of the src matrix lying within the top-left coordinate (30,30)
/// // and bottom-right coordinate (50,40) to the dest matrix
/// ecuda::matrix_copy( dest, src, 30, 30 );
/// \endcode
///
/// \param dest the destination matrix
/// \param src the source matrix
/// \param offsetRow offset in the starting row of the source matrix (default: 0)
/// \param offsetColumn offset in the starting column of the destination matrix (default: 0)
/// \return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
///
/*
template<typename T,class Alloc1,class Alloc2>
HOST cudaError_t matrix_copy( matrix<T,Alloc1>& dest, const matrix<T,Alloc2>& src, typename matrix<T,Alloc2>::size_type offsetRow=0, typename matrix<T,Alloc2>::size_type offsetColumn=0 ) {
	typedef typename matrix<T,Alloc2>::size_type size_type;
	const size_type nr = std::min( dest.number_rows()   , src.number_rows()-offsetRow       );
	const size_type nc = std::min( dest.number_columns(), src.number_columns()-offsetColumn );
	for( size_type i = 0; i < nr; ++i ) {
		cudaError_t rc = cudaMemcpy<T>( dest[i].data(), src[i+offsetRow].data(), nc, cudaMemcpyDeviceToDevice );
		if( rc != cudaSuccess ) return rc;
	}
	return cudaSuccess;
}
*/

///
/// \brief Swaps some or all of a source matrix with a destination matrix.
///
/// The subset of the two matrices can be specified with the offsetRow1, offsetColumn1, offsetRow2,
/// offsetColumn2 parameters along with the numberRows and numberColumns parameters which are applied
/// to both of the matrices.
///
/// If either of the subsets exceed the bounds of their matrix in either dimension a std::out_of_range
/// exception is thrown.
///
/// \param mat1 a matrix whose contents are to be swapped
/// \param mat2 the other matrix whose contents are to be swapped
/// \param numberRows the number of rows to swap
/// \param numberColumns the number of columns to swap
/// \param offsetRow1 the starting row in mat1 that will be swapped (default:0)
/// \param offsetColumn1 the starting column in mat1 that will be swapped (default:0)
/// \param offsetRow2 the starting row in mat2 that will be swapped (default:0)
/// \param offsetColumn2 the starting column in mat2 that will be swapped (default:0)
/// \return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
/// \throws std::out_of_range thrown if the specified bounds of either matrix exceeds its actual dimensions
///
/*
template<typename T,class Alloc1,class Alloc2>
HOST cudaError_t matrix_swap(
	matrix<T,Alloc1>& mat1,
	matrix<T,Alloc2>& mat2,
	typename matrix<T,Alloc1>::size_type numberRows=0, typename matrix<T,Alloc1>::size_type numberColumns=0,
	typename matrix<T,Alloc1>::size_type offsetRow1=0, typename matrix<T,Alloc1>::size_type offsetColumn1=0,
	typename matrix<T,Alloc2>::size_type offsetRow2=0, typename matrix<T,Alloc2>::size_type offsetColumn2=0
)
{
	if( (offsetRow1+numberRows) > mat1.number_rows() ) throw std::out_of_range( "ecuda::matrix_swap() specified row subset of mat1 is out of bounds" );
	if( (offsetRow2+numberRows) > mat2.number_rows() ) throw std::out_of_range( "ecuda::matrix_swap() specified row subset of mat2 is out of bounds" );
	if( (offsetColumn1+numberColumns) > mat1.number_columns() ) throw std::out_of_range( "ecuda::matrix_swap() specified column subset of mat1 is out of bounds" );
	if( (offsetColumn2+numberColumns) > mat2.number_columns() ) throw std::out_of_range( "ecuda::matrix_swap() specified column subset of mat2 is out of bounds" );
	ecuda::vector<T> stagingMemory( numberColumns );
	typedef typename matrix<T,Alloc1>::size_type size_type;
	for( size_type i = 0; i < numberRows; ++i ) {
		typename matrix<T,Alloc1>::row_type row1 = mat1[offsetRow1+i];
		typename matrix<T,Alloc2>::row_type row2 = mat1[offsetRow2+i];
		try {
			stagingMemory.assign_from_device( row1+offsetColumn1, row1+(offsetColumn1+numberColumns) );
			row1.assign_from_device( row2+offsetColumn2, row2+(offsetColumn2+numberColumns) );
			row2.assign_from_device( stagingMemory.begin(), stagingMemory.end() );
		} catch( cuda_error& ex ) {
			return ex.get_cuda_error_type();
		}
	}
	return cudaSuccess;
}
*/

} // namespace ecuda

#endif
