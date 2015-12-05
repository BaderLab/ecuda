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

#include <vector>

#include "global.hpp"
#include "algorithm.hpp"
#include "allocators.hpp"
#include "apiwrappers.hpp"
#include "memory.hpp"
#include "impl/models.hpp"
#include "type_traits.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<typename T,class Alloc> class matrix_kernel_argument; // forward declaration

} // namespace impl
/// \endcond

///
/// \brief A resizable matrix stored in device memory.
///
/// A matrix is defined as a 2D structure of dimensions rows*columns.  The default implementation
/// uses pitched memory where a 2D block of video memory is allocated with width=columns and height=rows.
/// Pitched memory is aligned in a device-dependent manner so that calls to individual elements can be
/// threaded more efficiently (i.e. minimizing the number of read operations required to supply data to
/// multiple threads). Consult the CUDA API documentation for a more verbose explanation.
///
/// Methods are prefaced with appropriate keywords to declare them as host and/or device capable.
/// In general: operations requiring memory allocation/deallocation are host only, operations
/// to access the values of specific elements are device only, and copy operations on ranges of data and
/// accessors of general information can be performed on both the host and device.
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
/// of CUDA since blockDim.y was limited to 512 (at the time of this writing it was 1024 in the newer versions
/// of CUDA).
///
/// Just keep in mind that the column dimension lies in contiguous memory, and the row dimension is contiguous
/// blocks of columns; thus, an implementation that aims to have concurrently running threads accessing
/// column >>>> row will run much more efficiently.
///
/// Matrix iterators (via begin(),end(),rbegin(),rend()) and lexicographical comparisons traverse the matrix
/// linearly in row-major fashion (i.e. each column of the first row is traversed, then each column of the
/// next row, and so on...).
///
template< typename T, class Alloc=device_pitch_allocator<T>, class P=shared_ptr<T> >
class matrix : private impl::device_contiguous_row_matrix< T, /*padded_ptr< T,*/P/* >*/ >
{

private:
	typedef impl::device_contiguous_row_matrix< T, /*padded_ptr< T,*/P/* >*/ > base_type;

public:
	typedef typename base_type::value_type      value_type;      //!< cell data type
	typedef Alloc                               allocator_type;  //!< allocator type
	typedef typename base_type::size_type       size_type;       //!< unsigned integral type
	typedef typename base_type::difference_type difference_type; //!< signed integral type
	typedef typename base_type::reference       reference;       //!< cell reference type
	typedef typename base_type::const_reference const_reference; //!< cell const reference type
	typedef typename base_type::pointer         pointer;         //!< cell pointer type
	typedef typename make_const<pointer>::type  const_pointer;   //!< cell const pointer type

	typedef typename base_type::row_type          row_type;          //!< matrix row container type
	typedef typename base_type::column_type       column_type;       //!< matrix column container type
	typedef typename base_type::const_row_type    const_row_type;    //!< matrix const row container type
	typedef typename base_type::const_column_type const_column_type; //!< matrix const column container type

	typedef typename base_type::iterator               iterator;               //!< iterator type
	typedef typename base_type::const_iterator         const_iterator;         //!< const iterator type
	typedef typename base_type::reverse_iterator       reverse_iterator;       //!< reverse iterator type
	typedef typename base_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

	typedef impl::matrix_kernel_argument<T,Alloc> kernel_argument; //!< kernel argument type

private:
	allocator_type allocator;
	//template<typename U,class Alloc2> class device_matrix;
	template<typename U,class Alloc2,class Q> friend class matrix;

protected:
	template<typename U>
	__HOST__ __DEVICE__ matrix( const matrix<T,Alloc,U>& src, ecuda::true_type ) : base_type( unmanaged_cast(src.get_pointer()), src.number_rows(), src.number_columns() ), allocator(src.allocator) {}
	__HOST__ __DEVICE__ matrix( const matrix& src, ecuda::true_type ) : base_type(src), allocator(src.allocator) {}

	template<typename U>
	__HOST__ __DEVICE__ matrix& shallow_assign( const matrix<T,Alloc,U>& other )
	{
		base_type::get_pointer() = unmanaged_cast(other.get_pointer());
		allocator = other.allocator;
		return *this;
	}

private:
	__HOST__ void init()
	{
		if( number_rows() and number_columns() ) {
			// TODO: this is unfortunate - have to get a padded_ptr from the allocator, unwrap it and
			//       give it to shared_ptr, and then rewrap it in a padded_ptr with the same attributes
			//       as the original - the device_contiguous_row_matrix second template parameter which
			//       enforces a padded_ptr of some type is the reason
			typename Alloc::pointer p = get_allocator().allocate( number_columns(), number_rows() );
			shared_ptr<value_type> sp( naked_cast<typename ecuda::add_pointer<value_type>::type>(p) );
			padded_ptr< value_type, shared_ptr<value_type> > pp( sp, p.get_pitch() ); //, p.get_width(), sp );
			base_type base( pp, number_rows(), number_columns() );
			base_type::swap( base );
		}
	}

public:
	///
	/// \brief Constructs a matrix with dimensions numberRows x numberColumns filled with copies of elements with value value.
	/// \param numberRows number of rows (default: 0)
	/// \param numberColumns number of columns (default: 0)
	/// \param value the value to initialize elements of the matrix with (default: T())
	/// \param allocator allocator to use for all memory allocations of this container
	///        (does not normally need to be specified, by default the internal ecuda pitched memory allocator)
	///
	__HOST__ matrix( const size_type numberRows=0, const size_type numberColumns=0, const value_type& value = value_type(), const allocator_type& allocator = allocator_type() ) :
		base_type( pointer(), numberRows, numberColumns ),
		allocator(allocator)
	{
		init();
		if( size() ) ecuda::fill( begin(), end(), value );
	}

	///
	/// \brief Copy constructor.
	///
	/// Constructs the matrix with a copy of the contents of src.
	///
	/// \param src Another matrix object of the same type, whose contents are copied.
	///
	__HOST__ matrix( const matrix& src ) :
		base_type( pointer(), src.number_rows(), src.number_columns() ),
		allocator(std::allocator_traits<allocator_type>::select_on_container_copy_construction(src.get_allocator()))
	{
		init();
		if( size() ) ecuda::copy( src.begin(), src.end(), begin() );
	}

	///
	/// \brief Copy constructor.
	///
	/// Constructs the matrix with a copy of the contents of src.
	///
	/// \param src Another matrix object of the same type, whose contents are copied.
	/// \param alloc Allocator to use for all memory allocations of this container.
	///
	__HOST__ matrix( const matrix& src, const allocator_type& alloc ) :
		base_type( pointer(), src.number_rows(), src.number_columns() ),
		allocator(alloc)
	{
		init();
		if( size() ) ecuda::copy( src.begin(), src.end(), begin() );
	}

	__HOST__ matrix& operator=( const matrix& other )
	{
		allocator = other.allocator;
		if( number_rows() != other.number_rows() or number_columns() != other.number_columns() )
			resize( other.number_rows(), other.number_columns() );
		if( size() ) ecuda::copy( other.begin(), other.end(), begin() );
		return *this;
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	///
	/// This operator is only available if the compiler is configured to allow C++11.
	///
	/// \param src another container to be used as source to initialize the elements of the container with
	///
	__HOST__ matrix( matrix&& src ) : base_type(std::move(src)), allocator(std::move(src.allocator)) {}

	__HOST__ matrix& operator=( matrix&& src )
	{
		base_type::operator=(std::move(src));
		allocator = std::move(src.allocator);
		return *this;
	}
	#endif

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
	__HOST__ __DEVICE__ inline const_iterator         cbegin()  const __NOEXCEPT__ { return base_type::cbegin();  }
	__HOST__ __DEVICE__ inline const_iterator         cend()    const __NOEXCEPT__ { return base_type::cend();    }
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin()       __NOEXCEPT__ { return base_type::crbegin(); }
	__HOST__ __DEVICE__ inline const_reverse_iterator crend()         __NOEXCEPT__ { return base_type::crend();   }
	#endif

	///
	/// \brief Returns the number of elements in the container (numberRows*numberColumns).
	///
	/// \returns The number of elements in the container.
	///
	__HOST__ __DEVICE__ inline size_type size() const __NOEXCEPT__ { return base_type::size(); }

	///
	/// \brief Returns the maximum number of elements the container is able to hold due to system
	///        or library implementation limitations.
	///
	/// \returns Maximum number of elements.
	///
	__HOST__ __DEVICE__ __CONSTEXPR__ inline size_type max_size() const __NOEXCEPT__ { return std::numeric_limits<size_type>::max(); }

	///
	/// \brief Returns the number of rows in the container.
	///
	/// \returns The number of rows in the container.
	///
	__HOST__ __DEVICE__ inline size_type number_rows() const __NOEXCEPT__ { return base_type::number_rows(); }

	///
	/// \brief Returns the number of columns in the container.
	///
	/// \returns The number of columns in the container.
	///
	__HOST__ __DEVICE__ inline size_type number_columns() const __NOEXCEPT__ { return base_type::number_columns(); }

	///
	/// \brief Resizes the container to have dimensions newNumberRows x newNumberColumns.
	///
	/// If the current size is greater in either or both dimensions, the existing elements are truncated.
	///
	/// \param newNumberRows new number of rows
	/// \param newNumberColumns new number of columns
	/// \param value the value to initialize the new elements with (default constructed if not specified)
	///
	__HOST__ void resize( const size_type newNumberRows, const size_type newNumberColumns, const value_type& value = value_type() )
	{
		if( number_rows() == newNumberRows and number_columns() == newNumberColumns ) return; // no resize needed
		// create new model
		matrix newMatrix( newNumberRows, newNumberColumns, value, get_allocator() );
		for( size_type i = 0; i < std::min(number_rows(),newNumberRows); ++i ) {
			const_row_type oldRow = get_row(i);
			row_type newRow = newMatrix.get_row(i);
			ecuda::copy( oldRow.begin(), oldRow.begin()+std::min(number_columns(),newNumberColumns), newRow.begin() );
		}
		swap( newMatrix );
	}

	///
	/// \brief Checks if the container has no elements.
	///
	/// \returns true if the container is empty, false otherwise.
	///
	__HOST__ __DEVICE__ inline bool empty() const __NOEXCEPT__ { return !number_rows() or !number_columns(); }

	///
	/// \brief Gets a view object of a single row of the matrix.
	///
	/// The view object is guaranteed to perform no memory allocations or deallocation, and merely
	/// holds a pointer to the start of the row and provides methods to traverse, access, and alter
	/// the underlying data.
	///
	/// \param rowIndex of the row to isolate
	/// \returns view object for the specified row
	///
	__HOST__ __DEVICE__ inline row_type get_row( const size_type rowIndex ) { return base_type::get_row(rowIndex); }

	///
	/// \brief Gets a view object of a single row of the matrix.
	///
	/// The view object is guaranteed to perform no memory allocations or deallocation, and merely
	/// holds a pointer to the start of the row and provides methods to traverse and access the
	/// underlying data.  In addition, the constness of this matrix is enforced so the view will not
	/// allow any alterations to the underlying data.
	///
	/// \param rowIndex of the row to isolate
	/// \returns view object for the specified row
	///
	__HOST__ __DEVICE__ inline const_row_type get_row( const size_type rowIndex ) const { return base_type::get_row(rowIndex); }

	///
	/// \brief Gets a view object of a single column of the matrix.
	///
	/// The view object is guaranteed to perform no memory allocations or deallocation, and merely
	/// holds a pointer to the start of the row and provides methods to traverse, access, and alter
	/// the underlying data.
	///
	/// \param columnIndex index of the column to isolate
	/// \returns view object for the specified column
	///
	__HOST__ __DEVICE__ inline column_type get_column( const size_type columnIndex ) { return base_type::get_column(columnIndex); }

	///
	/// \brief Gets a view object of a single column of the matrix.
	///
	/// The view object is guaranteed to perform no memory allocations or deallocation, and merely
	/// holds a pointer to the start of the row and provides methods to traverse and access the
	/// underlying data.  In addition, the constness of this matrix is enforced so the view will not
	/// allow any alterations to the underlying data.
	///
	/// \param columnIndex index of the column to isolate
	/// \returns view object for the specified column
	///
	__HOST__ __DEVICE__ inline const_column_type get_column( const size_type columnIndex ) const { return base_type::get_column(columnIndex); }

	///
	/// \brief Returns a reference to the element at specified row and column index, with bounds checking.
	///
	/// If the row and column are not within the range of the container, the current kernel will exit and
	/// cudaGetLastError will return cudaErrorUnknown.
	///
	/// \param rowIndex position of the row to return
	/// \param columnIndex position of the column to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline reference at( size_type rowIndex, size_type columnIndex )
	{
		if( rowIndex >= number_rows() or columnIndex >= number_columns() ) {
			#ifndef __CUDACC__
			throw std::out_of_range( EXCEPTION_MSG("ecuda::matrix::at() row and/or column index parameter is out of range") );
			#else
			// this strategy is taken from:
			// http://stackoverflow.com/questions/12521721/crashing-a-kernel-gracefully
			__threadfence();
			asm("trap;");
			#endif
		}
		return base_type::at( rowIndex, columnIndex );
	}

	///
	/// \brief Returns a constant reference to the element at specified row and column index, with bounds checking.
	///
	/// If the row and column are not within the range of the container, the current kernel will exit and
	/// cudaGetLastError will return cudaErrorUnknown.
	///
	/// \param rowIndex position of the row to return
	/// \param columnIndex position of the column to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline const_reference at( size_type rowIndex, size_type columnIndex ) const
	{
		if( rowIndex >= number_rows() or columnIndex >= number_columns() ) {
			#ifndef __CUDACC__
			throw std::out_of_range( EXCEPTION_MSG("ecuda::matrix::at() row and/or column index parameter is out of range") );
			#else
			// this strategy is taken from:
			// http://stackoverflow.com/questions/12521721/crashing-a-kernel-gracefully
			__threadfence();
			asm("trap;");
			#endif
		}
		return base_type::at( rowIndex, columnIndex );
	}

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// This is identical to at() but no bounds checking is performed.
	///
	/// \param rowIndex row of the element to return
	/// \param columnIndex column of the element to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline reference operator()( const size_type rowIndex, const size_type columnIndex ) { return base_type::at(rowIndex,columnIndex); }

	///
	/// \brief Returns a reference to the element at specified location index. No bounds checking is performed.
	///
	/// This is identical to at() but no bounds checking is performed.
	///
	/// \param rowIndex row of the element to return
	/// \param columnIndex column of the element to return
	/// \returns Reference to the requested element.
	///
	__DEVICE__ inline const_reference operator()( const size_type rowIndex, const size_type columnIndex ) const { return base_type::at(rowIndex,columnIndex); }

	///
	/// \brief operator[](rowIndex) alias for get_row(rowIndex)
	/// \param rowIndex index of the row to isolate
	/// \returns view object for the specified row
	///
	__HOST__ __DEVICE__ inline row_type operator[]( const size_type rowIndex ) { return get_row(rowIndex); }

	///
	/// \brief operator[](rowIndex) alias for get_row(rowIndex)
	/// \param rowIndex index of the row to isolate
	/// \returns view object for the specified row
	///
	__HOST__ __DEVICE__ inline const_row_type operator[]( const size_type rowIndex ) const { return get_row(rowIndex); }

	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// This is effectively the element at position (0,0). Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
	__DEVICE__ inline reference front() { return base_type::at(0,0); }

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// This is effectively the element at position (numberRows-1,numberColumns-1). Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
	__DEVICE__ inline reference back() { return base_type::at(number_rows()-1,number_columns()-1); }

	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// This is effectively the element at position (0,0). Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
	__DEVICE__ inline const_reference front() const { return base_type::at(0,0); }

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// This is effectively the element at position (numberRows-1,numberColumns-1). Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
	__DEVICE__ inline const_reference back() const { return base_type::at(number_rows()-1,number_columns()-1); }

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(),data()+size()) is always a valid
	/// range, even if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
	__HOST__ __DEVICE__ inline pointer data() __NOEXCEPT__ { return base_type::get_pointer(); }

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(),data()+size()) is always a valid
	/// range, even if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
	__HOST__ __DEVICE__ inline const_pointer data() const __NOEXCEPT__ { return base_type::get_pointer(); }

	///
	/// \brief Assigns a given value to all elements in the container.
	///
	/// \param value the value to assign to the elements
	///
	__HOST__ __DEVICE__ inline void fill( const value_type& value ) { if( !empty() ) ecuda::fill( begin(), end(), value ); }

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
	__HOST__ __DEVICE__ inline void swap( matrix& other ) { base_type::swap( other ); }

	///
	/// \brief Returns the allocator associated with the container.
	/// \returns The associated allocator.
	///
	__HOST__ inline allocator_type get_allocator() const { return allocator; }

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
	__HOST__ __DEVICE__ inline bool operator==( const matrix<value_type,Alloc2>& other ) const
	{
		return ecuda::equal( begin(), end(), other.begin() );
	}

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
	__HOST__ __DEVICE__ inline bool operator!=( const matrix<value_type,Alloc2>& other ) const { return !operator==(other); }

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
	__HOST__ __DEVICE__ inline bool operator<( const matrix<value_type,Alloc2>& other ) const
	{
		return ecuda::lexicographical_compare( begin(), end(), other.begin(), other.end() );
	}

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
	__HOST__ __DEVICE__ inline bool operator>( const matrix<value_type,Alloc2>& other ) const
	{
		return ecuda::lexicographical_compare( other.begin(), other.end(), begin(), end() );
	}

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
	__HOST__ __DEVICE__ inline bool operator<=( const matrix<value_type,Alloc2>& other ) const { return !operator>(other); }

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
	__HOST__ __DEVICE__ inline bool operator>=( const matrix<value_type,Alloc2>& other ) const { return !operator<(other); }

};

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template< typename T, class Alloc=device_pitch_allocator<T> >
class matrix_kernel_argument : public matrix<T,Alloc,typename ecuda::add_pointer<T>::type>
{
private:
	typedef matrix<T,Alloc,typename ecuda::add_pointer<T>::type> base_type;

public:
	template<class P>
	__HOST__ matrix_kernel_argument( const matrix<T,Alloc,P>& src ) : base_type( src, ecuda::true_type() ) {}

	__HOST__ __DEVICE__ matrix_kernel_argument( const matrix_kernel_argument& src ) : base_type( src, ecuda::true_type() ) {}

	template<class P>
	__HOST__ matrix_kernel_argument& operator=( const matrix<T,Alloc,P>& src )
	{
		base_type::shallow_assign( src );
		return *this;
	}

	#ifdef __CPP11_SUPPORTED__
	matrix_kernel_argument( matrix_kernel_argument&& src ) : base_type(std::move(src)) {}

	matrix_kernel_argument& operator=( matrix_kernel_argument&& src )
	{
		base_type::operator=(std::move(src));
		return *this;
	}
	#endif

};

} // namespace impl
/// \endcond

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
///
template<typename T,class Alloc1,class Alloc2>
__HOST__ void matrix_copy( matrix<T,Alloc1>& dest, const matrix<T,Alloc2>& src, typename matrix<T,Alloc2>::size_type offsetRow=0, typename matrix<T,Alloc2>::size_type offsetColumn=0 ) {
	typedef typename matrix<T,Alloc2>::size_type size_type;
	const size_type nr = std::min( dest.number_rows()   , src.number_rows()-offsetRow       );
	const size_type nc = std::min( dest.number_columns(), src.number_columns()-offsetColumn );
	for( size_type i = 0; i < nr; ++i ) {
		typename matrix<T,Alloc1>::row_type destRow = dest[i];
		typename matrix<T,Alloc2>::const_row_type srcRow = src[i+offsetRow];
		::ecuda::copy( src.begin(), src.end(), dest.begin() );
	}
}

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
/// \throws std::out_of_range thrown if the specified bounds of either matrix exceeds its actual dimensions
///
template<typename T,class Alloc1,class Alloc2>
__HOST__ void matrix_swap(
	matrix<T,Alloc1>& mat1,
	matrix<T,Alloc2>& mat2,
	typename matrix<T,Alloc1>::size_type numberRows=0, typename matrix<T,Alloc1>::size_type numberColumns=0,
	typename matrix<T,Alloc1>::size_type offsetRow1=0, typename matrix<T,Alloc1>::size_type offsetColumn1=0,
	typename matrix<T,Alloc2>::size_type offsetRow2=0, typename matrix<T,Alloc2>::size_type offsetColumn2=0
)
{
	if( (offsetRow1+numberRows) > mat1.number_rows() ) throw std::out_of_range( EXCEPTION_MSG("ecuda::matrix_swap() specified row subset of mat1 is out of bounds") );
	if( (offsetRow2+numberRows) > mat2.number_rows() ) throw std::out_of_range( EXCEPTION_MSG("ecuda::matrix_swap() specified row subset of mat2 is out of bounds" ) );
	if( (offsetColumn1+numberColumns) > mat1.number_columns() ) throw std::out_of_range( EXCEPTION_MSG("ecuda::matrix_swap() specified column subset of mat1 is out of bounds") );
	if( (offsetColumn2+numberColumns) > mat2.number_columns() ) throw std::out_of_range( EXCEPTION_MSG("ecuda::matrix_swap() specified column subset of mat2 is out of bounds") );
	std::vector< T, host_allocator<T> > stagingMemory( numberColumns );
	typedef typename matrix<T,Alloc1>::size_type size_type;
	for( size_type i = 0; i < numberRows; ++i ) {
		typename matrix<T,Alloc1>::row_type row1 = mat1[offsetRow1+i];
		typename matrix<T,Alloc2>::row_type row2 = mat1[offsetRow2+i];
		stagingMemory.assign( row1.begin()+offsetColumn1, row1.begin()+(offsetColumn1+numberColumns) );
		ecuda::copy( row2.begin()+offsetColumn2, row2.begin()+(offsetColumn2+numberColumns), row1.begin()+offsetColumn1 );
		ecuda::copy( stagingMemory.begin(), stagingMemory.end(), row2.begin()+offsetColumn2 );
	}
}

template<typename T,class Alloc>
__HOST__ void matrix_transpose(
	matrix<T,Alloc>& src
)
{
	if( src.empty() ) return;
	std::vector< T, host_allocator<T> > stagingMemory( src.number_columns() ); // stage a single row
	std::vector<T> hostMatrix( src.size() );
	for( typename matrix<T,Alloc>::size_type i = 0; i < src.number_rows(); ++i ) {
		ecuda::copy( src[i].begin(), src[i].end(), stagingMemory.begin() ); // copy row
		typename std::vector< T, host_allocator<T> >::const_iterator srcElement = stagingMemory.begin();
		for( std::size_t j = 0; j < src.number_columns(); ++j, ++srcElement ) hostMatrix[j*src.number_rows()+i] = *srcElement; // transpose
	}
	src.resize( src.number_columns(), src.number_rows() ); // resize destination matrix
	ecuda::copy( hostMatrix.begin(), hostMatrix.end(), src.begin() );
	//typename std::vector<T>::const_iterator srcRow = hostMatrix.begin();
	//for( typename matrix<T,Alloc>::size_type i = 0; i < src.number_rows(); ++i, srcRow += src.number_columns() ) {
	//	ecuda::copy( srcRow, srcRow+src.number_columns(), src[i].begin() );
	//}
}

} // namespace ecuda

#endif
