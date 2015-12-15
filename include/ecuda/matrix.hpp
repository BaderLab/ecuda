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

<<<<<<< HEAD
#ifdef __CPP11_SUPPORTED__
#include <type_traits>
#endif
#include <vector>

#include "config.hpp"
#include "global.hpp"
#include "allocators.hpp"
#include "apiwrappers.hpp"
#include "device_ptr.hpp"
#include "padded_ptr.hpp"
#include "striding_ptr.hpp"
#include "vector.hpp"
#include "views.hpp"


namespace ecuda {

=======
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

>>>>>>> ecuda2/master
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
<<<<<<< HEAD
/// template<typename T> __global__ void doMatrixOperation( ecuda::matrix<T> matrix ) {
///    const int row = blockIdx.x;
///    const int col = blockDim.y*gridDim.y; // each thread gets a different column value
///    if( row < matrix.number_rows() and col < matrix.number_columns() ) {
///       T& value = matrix[row][col];
=======
/// template<typename T> __global__ void doMatrixOperation( typename ecuda::matrix<T>::kernel_argument matrix )
/// {
///    const int row = blockIdx.x;
///    const int col = blockDim.y*gridDim.y; // each thread gets a different column value
///    if( row < matrix.number_rows() && col < matrix.number_columns() ) {
///       T& value = matrix(row,col);
>>>>>>> ecuda2/master
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
<<<<<<< HEAD
template< typename T, class Alloc=device_pitch_allocator<T> >
class matrix {

public:
	typedef T value_type; //!< cell data type
	typedef Alloc allocator_type; //!< allocator type
	typedef std::size_t size_type; //!< unsigned integral type
	typedef std::ptrdiff_t difference_type; //!< signed integral type
	typedef value_type& reference; //!< cell reference type
	typedef const value_type& const_reference; //!< cell const reference type
	typedef value_type* pointer; //!< cell pointer type
	typedef const value_type* const_pointer; //!< cell const pointer type

	typedef contiguous_sequence_view<value_type> row_type; //!< matrix row container type
	typedef sequence_view< value_type, padded_ptr<value_type,striding_ptr<value_type>,1> > column_type; //!< matrix column container type
	typedef const contiguous_sequence_view<const value_type> const_row_type; //!< matrix const row container type
	typedef const sequence_view< const value_type, padded_ptr<const value_type,striding_ptr<const value_type>,1> > const_column_type; //!< matrix const column container type

	typedef device_iterator< value_type, padded_ptr<value_type,pointer,1> > iterator; //!< iterator type
	typedef device_iterator< const value_type, padded_ptr<const value_type,const_pointer,1> > const_iterator; //!< const iterator type
	typedef reverse_device_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type


private:
	// REMEMBER: numberRows, numberColumns, and pitch altered on device memory won't be
	//           reflected on the host object. Don't allow the device to perform any operations that
	//           change their value.
	size_type numberRows; //!< number of matrix rows
	size_type numberColumns; //!< number of matrix columns
	size_type pitch; //!< the padded width of the 2D memory allocation in bytes
	device_ptr<value_type> deviceMemory; //!< smart pointer to video card memory
	allocator_type allocator;
=======
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

	///
	/// \brief Used by the kernel_argument subclass to create a shallow copy using an unmanaged pointer.
	///
	template<typename U>
	__HOST__ __DEVICE__ matrix( const matrix<T,Alloc,U>& src, ecuda::true_type ) : base_type( unmanaged_cast(src.get_pointer()), src.number_rows(), src.number_columns() ), allocator(src.allocator) {}

	///
	/// \brief Used by the kernel_argument subclass to create a shallow copy using an unmanaged pointer.
	///
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
>>>>>>> ecuda2/master

public:
	///
	/// \brief Constructs a matrix with dimensions numberRows x numberColumns filled with copies of elements with value value.
	/// \param numberRows number of rows (default: 0)
	/// \param numberColumns number of columns (default: 0)
	/// \param value the value to initialize elements of the matrix with (default: T())
	/// \param allocator allocator to use for all memory allocations of this container
	///        (does not normally need to be specified, by default the internal ecuda pitched memory allocator)
	///
<<<<<<< HEAD
	HOST matrix( const size_type numberRows=0, const size_type numberColumns=0, const T& value = T(), const Alloc& allocator = Alloc() ) : numberRows(numberRows), numberColumns(numberColumns), allocator(allocator) {
		if( numberRows and numberColumns ) {
			deviceMemory = device_ptr<value_type>( this->allocator.allocate( numberColumns, numberRows, pitch ) );
			std::vector< value_type, host_allocator<value_type> > v( numberRows*numberColumns, value );
			CUDA_CALL( cudaMemcpy2D<value_type>( deviceMemory.get(), pitch, &v.front(), numberColumns*sizeof(value_type), numberColumns, numberRows, cudaMemcpyHostToDevice ) );
		}
	}

	///
	/// \brief Constructs a matrix with a shallow copy of each of the elements in src.
	///
	/// Be careful to note that a shallow copy means that only the pointer to the device memory
	/// that holds the elements is copied in the newly constructed container. This allows
	/// containers to be passed-by-value to kernel functions with minimal overhead. If a deep copy
	/// of the container is required in host code, use the << or >> operators, or use iterators.
	/// For example:
	///
	/// \code{.cpp}
	/// ecuda::matrix<int> matrix( 5, 10, 99 ); // create a matrix of dimensions 5x10 filled with 99
	/// ecuda::matrix<int> newMatrix( matrix ); // shallow copy (changes to newMatrix reflected in matrix)
	/// ecuda::matrix<int> newMatrix( 5, 10 );
	/// newMatrix << matrix; // deep copy
	/// matrix >> newMatrix; // deep copy
	/// newMatrix.assign( matrix.begin(), matrix.end() ); // deep copy
	/// \endcode
	///
	/// \param src Another matrix object of the same type, whose contents are copied.
	///
	HOST DEVICE matrix( const matrix& src ) :
		numberRows(src.numberRows),
		numberColumns(src.numberColumns),
		pitch(src.pitch),
		deviceMemory(src.deviceMemory),
		//#ifdef __CPP11_SUPPORTED__
		//allocator(std::allocator_traits<allocator_type>::select_on_container_copy_construction(src.get_allocator()))
		//#else
		allocator(src.allocator)
		//#endif
	{
=======
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
		allocator(src.get_allocator())
// TODO: this is broken due to some complaints from stdlib about making an iterator to a void pointer
//		allocator(std::allocator_traits<allocator_type>::select_on_container_copy_construction(src.get_allocator()))
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
>>>>>>> ecuda2/master
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Move constructor. Constructs the container with the contents of the other using move semantics.
	///
	/// This operator is only available if the compiler is configured to allow C++11.
	///
	/// \param src another container to be used as source to initialize the elements of the container with
	///
<<<<<<< HEAD
	HOST matrix( matrix<T>&& src ) : numberRows(src.numberRows), numberColumns(src.numberColumns), pitch(src.pitch), deviceMemory(std::move(src.deviceMemory)), allocator(std::move(src.allocator)) {}
	#endif

	//HOST DEVICE virtual ~matrix() {}
=======
	__HOST__ matrix( matrix&& src ) : base_type() { swap(src); }

	__HOST__ matrix& operator=( matrix&& src )
	{
		swap(src);
		return *this;
	}
	#endif
>>>>>>> ecuda2/master

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
<<<<<<< HEAD
	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator( padded_ptr<value_type,pointer,1>( data(), number_columns(), pitch-number_columns()*sizeof(value_type), 0 ) ); }
=======
	__HOST__ __DEVICE__ inline iterator begin() __NOEXCEPT__ { return base_type::begin(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
<<<<<<< HEAD
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator( padded_ptr<value_type,pointer,1>( allocator.address( data(), number_rows(), 0, pitch ), number_columns(), pitch-number_columns()*sizeof(value_type), 0 ) ); }
=======
	__HOST__ __DEVICE__ inline iterator end() __NOEXCEPT__ { return base_type::end(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns an iterator to the first element of the container.
	///
	/// If the container is empty, the returned iterator will be equal to end().
	///
	/// \returns Iterator to the first element.
	///
<<<<<<< HEAD
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator( padded_ptr<const value_type,const_pointer,1>( data(), number_columns(), pitch-number_columns()*sizeof(value_type), 0 ) ); }
=======
	__HOST__ __DEVICE__ inline const_iterator begin() const __NOEXCEPT__ { return base_type::begin(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns an iterator to the element following the last element of the container.
	///
	/// The element acts as a placeholder; attempting to access it results in undefined behaviour.
	///
	/// \returns Iterator to the element following the last element.
	///
<<<<<<< HEAD
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator( padded_ptr<const value_type,const_pointer,1>( allocator.address( data(), number_rows(), 0, pitch ), number_columns(), pitch-number_columns()*sizeof(value_type), 0 ) ); }
=======
	__HOST__ __DEVICE__ inline const_iterator end() const __NOEXCEPT__ { return base_type::end(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
<<<<<<< HEAD
	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(end()); }
=======
	__HOST__ __DEVICE__ inline reverse_iterator rbegin() __NOEXCEPT__ { return base_type::rbegin(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
<<<<<<< HEAD
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(begin()); }
=======
	__HOST__ __DEVICE__ inline reverse_iterator rend() __NOEXCEPT__ { return base_type::rend(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns a reverse iterator to the first element of the reversed container.
	///
	/// It corresponds to the last element of the non-reversed container.
	///
	/// \returns Reverse iterator to the first element.
	///
<<<<<<< HEAD
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(end()); }
=======
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return base_type::rbegin(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns a reverse iterator to the element following the last element of the reversed container.
	///
	/// It corresponds to the element preceding the first element of the non-reversed container. This element
	/// acts as a placeholder, attempting to access it results in undefined behaviour.
	///
	/// \returns Reverse iterator to the element following the last element.
	///
<<<<<<< HEAD
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(begin()); }

	#ifdef __CPP11_SUPPORTED__
	HOST DEVICE inline const_iterator cbegin() const __NOEXCEPT__ { return const_iterator( padded_ptr<const value_type,const_pointer,1>( data(), number_columns(), pitch-number_columns()*sizeof(value_type), 0 ) ); }
	HOST DEVICE inline const_iterator cend() const __NOEXCEPT__ { return const_iterator( padded_ptr<const value_type,const_pointer,1>( allocator.address( data(), number_rows(), 0, pitch ), number_columns(), pitch-number_columns()*sizeof(value_type), 0 ) ); }
	HOST DEVICE inline const_reverse_iterator crbegin() __NOEXCEPT__ { return const_reverse_iterator(cend()); }
	HOST DEVICE inline const_reverse_iterator crend() __NOEXCEPT__ { return const_reverse_iterator(cbegin()); }
=======
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const __NOEXCEPT__ { return base_type::rend(); }

	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ inline const_iterator         cbegin()  const __NOEXCEPT__ { return base_type::cbegin();  }
	__HOST__ __DEVICE__ inline const_iterator         cend()    const __NOEXCEPT__ { return base_type::cend();    }
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin()       __NOEXCEPT__ { return base_type::crbegin(); }
	__HOST__ __DEVICE__ inline const_reverse_iterator crend()         __NOEXCEPT__ { return base_type::crend();   }
>>>>>>> ecuda2/master
	#endif

	///
	/// \brief Returns the number of elements in the container (numberRows*numberColumns).
	///
	/// \returns The number of elements in the container.
	///
<<<<<<< HEAD
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return number_rows()*number_columns(); }
=======
	__HOST__ __DEVICE__ inline size_type size() const __NOEXCEPT__ { return base_type::size(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns the maximum number of elements the container is able to hold due to system
	///        or library implementation limitations.
	///
	/// \returns Maximum number of elements.
	///
<<<<<<< HEAD
	HOST DEVICE __CONSTEXPR__ inline size_type max_size() const __NOEXCEPT__ { return std::numeric_limits<size_type>::max(); }
=======
	__HOST__ __DEVICE__ __CONSTEXPR__ inline size_type max_size() const __NOEXCEPT__ { return std::numeric_limits<size_type>::max(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns the number of rows in the container.
	///
	/// \returns The number of rows in the container.
	///
<<<<<<< HEAD
	HOST DEVICE inline size_type number_rows() const __NOEXCEPT__ { return numberRows; }
=======
	__HOST__ __DEVICE__ inline size_type number_rows() const __NOEXCEPT__ { return base_type::number_rows(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns the number of columns in the container.
	///
	/// \returns The number of columns in the container.
	///
<<<<<<< HEAD
	HOST DEVICE inline size_type number_columns() const __NOEXCEPT__ { return numberColumns; }

	///
	/// \brief Returns the pitch of the underlying 2D device memory.
	///
	/// \returns THe pitch of the underlying 2D device memory (in bytes).
	///
	HOST DEVICE inline size_type get_pitch() const __NOEXCEPT__ { return pitch; }
=======
	__HOST__ __DEVICE__ inline size_type number_columns() const __NOEXCEPT__ { return base_type::number_columns(); }
>>>>>>> ecuda2/master

	///
	/// \brief Resizes the container to have dimensions newNumberRows x newNumberColumns.
	///
	/// If the current size is greater in either or both dimensions, the existing elements are truncated.
	///
	/// \param newNumberRows new number of rows
	/// \param newNumberColumns new number of columns
	/// \param value the value to initialize the new elements with (default constructed if not specified)
	///
<<<<<<< HEAD
	HOST void resize( const size_type newNumberRows, const size_type newNumberColumns, const value_type& value = value_type() ) {
		if( number_rows() == newNumberRows and number_columns() == newNumberColumns ) return; // no resize needed
		// allocate memory
		size_type newPitch;
		device_ptr<value_type> newDeviceMemory( allocator.allocate( newNumberColumns, newNumberRows, newPitch ) );
		CUDA_CALL( cudaMemset2D<value_type>( newDeviceMemory.get(), newPitch, value, newNumberColumns, newNumberRows ) );
		for( size_type i = 0; i < std::min(numberRows,newNumberRows); ++i ) {
			CUDA_CALL(
				cudaMemcpy<value_type>(
					allocator.address( newDeviceMemory.get(), i, 0, newPitch ),
					allocator.address( deviceMemory.get(), i, 0, pitch ),
					std::min(numberColumns,newNumberColumns),
					cudaMemcpyDeviceToDevice
				)
			);
		}
		numberRows = newNumberRows;
		numberColumns = newNumberColumns;
		pitch = newPitch;
		deviceMemory = newDeviceMemory;
=======
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
>>>>>>> ecuda2/master
	}

	///
	/// \brief Checks if the container has no elements.
	///
	/// \returns true if the container is empty, false otherwise.
	///
<<<<<<< HEAD
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return !number_rows() or !number_columns(); }
=======
	__HOST__ __DEVICE__ inline bool empty() const __NOEXCEPT__ { return !number_rows() or !number_columns(); }
>>>>>>> ecuda2/master

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
<<<<<<< HEAD
	HOST DEVICE inline row_type get_row( const size_type rowIndex ) { return row_type( allocator.address( data(), rowIndex, 0, pitch ), number_columns() ); }
=======
	__HOST__ __DEVICE__ inline row_type get_row( const size_type rowIndex ) { return base_type::get_row(rowIndex); }
>>>>>>> ecuda2/master

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
<<<<<<< HEAD
	HOST DEVICE inline const_row_type get_row( const size_type rowIndex ) const { return const_row_type( allocator.address( data(), rowIndex, 0, pitch ), number_columns() ); }
=======
	__HOST__ __DEVICE__ inline const_row_type get_row( const size_type rowIndex ) const { return base_type::get_row(rowIndex); }
>>>>>>> ecuda2/master

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
<<<<<<< HEAD
	HOST DEVICE inline column_type get_column( const size_type columnIndex ) {
		pointer p = allocator.address( data(), 0, columnIndex, pitch );
		striding_ptr<value_type> sp( p, number_columns() );
		padded_ptr< value_type, striding_ptr<value_type>, 1 > pp( sp, 1, pitch-numberColumns*sizeof(value_type), 0 );
		return column_type( pp, number_rows() );
	}
=======
	__HOST__ __DEVICE__ inline column_type get_column( const size_type columnIndex ) { return base_type::get_column(columnIndex); }
>>>>>>> ecuda2/master

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
<<<<<<< HEAD
	HOST DEVICE inline const_column_type get_column( const size_type columnIndex ) const {
		const_pointer p = allocator.address( data(), 0, columnIndex, pitch );
		striding_ptr<const value_type> sp( p, number_columns() );
		padded_ptr< const value_type, striding_ptr<const value_type>, 1 > pp( sp, 1, pitch-numberColumns*sizeof(value_type), 0 );
		return const_column_type( pp, number_rows() );
	}

	///
=======
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
>>>>>>> ecuda2/master
	/// \brief operator[](rowIndex) alias for get_row(rowIndex)
	/// \param rowIndex index of the row to isolate
	/// \returns view object for the specified row
	///
<<<<<<< HEAD
	HOST DEVICE inline row_type operator[]( const size_type rowIndex ) { return get_row(rowIndex); }
=======
	__HOST__ __DEVICE__ inline row_type operator[]( const size_type rowIndex ) { return get_row(rowIndex); }
>>>>>>> ecuda2/master

	///
	/// \brief operator[](rowIndex) alias for get_row(rowIndex)
	/// \param rowIndex index of the row to isolate
	/// \returns view object for the specified row
	///
<<<<<<< HEAD
	HOST DEVICE inline const_row_type operator[]( const size_type rowIndex ) const { return get_row(rowIndex); }
=======
	__HOST__ __DEVICE__ inline const_row_type operator[]( const size_type rowIndex ) const { return get_row(rowIndex); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// This is effectively the element at position (0,0). Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
<<<<<<< HEAD
	DEVICE inline reference front() { return *data(); }
=======
	__DEVICE__ inline reference front() { return base_type::at(0,0); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// This is effectively the element at position (numberRows-1,numberColumns-1). Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
<<<<<<< HEAD
	DEVICE inline reference back() { return *allocator.address( data(), number_rows()-1, number_columns()-1, pitch ); }
=======
	__DEVICE__ inline reference back() { return base_type::at(number_rows()-1,number_columns()-1); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns a reference to the first element in the container.
	///
	/// This is effectively the element at position (0,0). Calling front on an empty container is undefined.
	///
	/// \returns Reference to the first element.
	///
<<<<<<< HEAD
	DEVICE inline const_reference front() const { return *data(); }
=======
	__DEVICE__ inline const_reference front() const { return base_type::at(0,0); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns a reference to the last element in the container.
	///
	/// This is effectively the element at position (numberRows-1,numberColumns-1). Calling back on an empty container is undefined.
	///
	/// \returns Reference to the last element.
	///
<<<<<<< HEAD
	DEVICE inline const_reference back() const { return *allocator.address( data(), number_rows()-1, number_columns()-1, pitch ); }
=======
	__DEVICE__ inline const_reference back() const { return base_type::at(number_rows()-1,number_columns()-1); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(),data()+size()) is always a valid
	/// range, even if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
<<<<<<< HEAD
	HOST DEVICE inline pointer data() __NOEXCEPT__ { return deviceMemory.get(); }
=======
	__HOST__ __DEVICE__ inline pointer data() __NOEXCEPT__ { return base_type::get_pointer(); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns pointer to the underlying array serving as element storage.
	///
	/// The pointer is such that range [data(),data()+size()) is always a valid
	/// range, even if the container is empty.
	///
	/// \returns Pointer to the underlying element storage.
	///
<<<<<<< HEAD
	HOST DEVICE inline const_pointer data() const __NOEXCEPT__ { return deviceMemory.get(); }

	///
	/// \brief Replaces the contents of the container.
	/// \param newNumberRows new number of rows
	/// \param newNumberColumns new number of columns
	/// \param value the value to initialize elements of the container with
	///
	HOST inline void assign( size_type newNumberRows, size_type newNumberColumns, const value_type& value = value_type() ) { resize( newNumberRows, newNumberColumns, value ); }

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
	/// \throws std::length_error if the number of elements in the range [begin,end) does not match the number of elements in this container
	/// \param first,last the range to copy the elements from
	///
	template<class Iterator>
	HOST DEVICE void assign( Iterator first, Iterator last ) {
		#ifdef __CUDA_ARCH__
		assign( first, last, typename std::iterator_traits<Iterator>::iterator_category() );
		#else
		const typename std::iterator_traits<Iterator>::difference_type len = ::ecuda::distance(first,last);
		if( len < 0 or static_cast<size_type>(len) != size() ) throw std::length_error( EXCEPTION_MSG("ecuda::matrix::assign(first,last) the number of elements to assign does not match the size of this matrix") );
		for( size_type i = 0; i < number_rows(); ++i ) {
			row_type row = get_row(i);
			Iterator rowEnd = first;
			::ecuda::advance( rowEnd, number_columns() );
			row.assign( first, rowEnd );
			first = rowEnd;
		}
		#endif
	}

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Replaces the contents of the container with copies of those in the initializer list.
	/// \throws std::length_error if the number of elements in the initializer list does not match the number of elements in this container
	/// \param il initializer list to initialize the elements of the container with
	///
	HOST inline void assign( std::initializer_list<T> il ) {
		host_array_proxy<const T> proxy( il.begin(), il.size() );
		assign( proxy.begin(), proxy.end() );
		//assign( il.begin(), il.end() );
	}
	#endif
=======
	__HOST__ __DEVICE__ inline const_pointer data() const __NOEXCEPT__ { return base_type::get_pointer(); }
>>>>>>> ecuda2/master

	///
	/// \brief Assigns a given value to all elements in the container.
	///
	/// \param value the value to assign to the elements
	///
<<<<<<< HEAD
	HOST DEVICE void fill( const value_type& value ) {
		#ifdef __CUDA_ARCH__
		for( iterator iter = begin(); iter != end(); ++iter ) *iter = value;
		#else
		std::vector< value_type, host_allocator<value_type> > v( number_columns(), value );
		for( size_type i = 0; i < number_rows(); ++i ) get_row(i).assign( v.begin(), v.end() );
		#endif
	}
=======
	__HOST__ __DEVICE__ inline void fill( const value_type& value ) { if( !empty() ) ecuda::fill( begin(), end(), value ); }
>>>>>>> ecuda2/master

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
<<<<<<< HEAD
	HOST DEVICE void swap( matrix& other ) {
		// just swap all members
		#ifdef __CUDA_ARCH__
		ecuda::swap( numberRows, other.numberRows );
		ecuda::swap( numberColumns, other.numberColumns );
		ecuda::swap( pitch, other.pitch );
		ecuda::swap( deviceMemory, other.deviceMemory );
		#else
		std::swap( numberRows, other.numberRows );
		std::swap( numberColumns, other.numberColumns );
		std::swap( pitch, other.pitch );
		std::swap( deviceMemory, other.deviceMemory );
		#endif
	}
=======
	__HOST__ __DEVICE__ inline void swap( matrix& other ) { base_type::swap( other ); }
>>>>>>> ecuda2/master

	///
	/// \brief Returns the allocator associated with the container.
	/// \returns The associated allocator.
	///
<<<<<<< HEAD
	HOST inline allocator_type get_allocator() const { return allocator; }
=======
	__HOST__ inline allocator_type get_allocator() const { return allocator; }
>>>>>>> ecuda2/master

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
<<<<<<< HEAD
	HOST DEVICE bool operator==( const matrix<value_type,Alloc2>& other ) const {
		if( number_rows() != other.number_rows() ) return false;
		if( number_columns() != other.number_columns() ) return false;
		#ifdef __CUDA_ARCH__
		const_iterator iter1 = begin();
		const_iterator iter2 = other.begin();
		for( ; iter1 != end(); ++iter1, ++iter2 ) if( !( *iter1 == *iter2 ) ) return false;
		return true;
		#else
		std::vector< value_type, host_allocator<value_type> > v1( number_columns() );
		std::vector< value_type, host_allocator<value_type> > v2( number_columns() );
		for( size_type i = 0; i < number_rows(); ++i ) {
			const_row_type row1 = get_row(i);
			typename matrix<value_type,Alloc2>::const_row_type row2 = other.get_row(i);
			::ecuda::copy( row1.begin(), row1.end(), v1.begin() );
			::ecuda::copy( row2.begin(), row2.end(), v2.begin() );
			if( v1 == v2 ) continue;
			return false;
		}
		return true;
		#endif
=======
	__HOST__ __DEVICE__ inline bool operator==( const matrix<value_type,Alloc2>& other ) const
	{
		return ecuda::equal( begin(), end(), other.begin() );
>>>>>>> ecuda2/master
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
<<<<<<< HEAD
	HOST DEVICE inline bool operator!=( const matrix<value_type,Alloc2>& other ) const { return !operator==(other); }
=======
	__HOST__ __DEVICE__ inline bool operator!=( const matrix<value_type,Alloc2>& other ) const { return !operator==(other); }
>>>>>>> ecuda2/master

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
<<<<<<< HEAD
	HOST DEVICE bool operator<( const matrix<value_type,Alloc2>& other ) const {
		#ifdef __CUDA_ARCH__
		return ecuda::lexicographical_compare( begin(), end(), other.begin(), other.end() );
		#else
		std::vector< value_type, host_allocator<value_type> > v1( number_columns() );
		std::vector< value_type, host_allocator<value_type> > v2( number_columns() );
		for( size_type i = 0; i < number_rows(); ++i ) {
			const_row_type row1 = get_row(i);
			typename matrix<value_type,Alloc2>::const_row_type row2 = other.get_row(i);
			::ecuda::copy( row1.begin(), row1.end(), v1.begin() );
			::ecuda::copy( row2.begin(), row2.end(), v2.begin() );
			if( v1 < v2 ) return true;
		}
		return false;
		#endif
=======
	__HOST__ __DEVICE__ inline bool operator<( const matrix<value_type,Alloc2>& other ) const
	{
		return ecuda::lexicographical_compare( begin(), end(), other.begin(), other.end() );
>>>>>>> ecuda2/master
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
<<<<<<< HEAD
	HOST DEVICE bool operator>( const matrix<value_type,Alloc2>& other ) const {
		#ifdef __CUDA_ARCH__
		return ecuda::lexicographical_compare( other.begin(), other.end(), begin(), end() );
		#else
		std::vector< value_type, host_allocator<value_type> > v1( number_columns() );
		std::vector< value_type, host_allocator<value_type> > v2( number_columns() );
		for( size_type i = 0; i < number_rows(); ++i ) {
			const_row_type row1 = get_row(i);
			typename matrix<value_type,Alloc2>::const_row_type row2 = other.get_row(i);
			::ecuda::copy( row1.begin(), row1.end(), v1.begin() );
			::ecuda::copy( row2.begin(), row2.end(), v2.begin() );
			if( v1 > v2 ) return true;
		}
		return false;
		#endif
=======
	__HOST__ __DEVICE__ inline bool operator>( const matrix<value_type,Alloc2>& other ) const
	{
		return ecuda::lexicographical_compare( other.begin(), other.end(), begin(), end() );
>>>>>>> ecuda2/master
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
<<<<<<< HEAD
	HOST DEVICE inline bool operator<=( const matrix<value_type,Alloc2>& other ) const { return !operator>(other); }
=======
	__HOST__ __DEVICE__ inline bool operator<=( const matrix<value_type,Alloc2>& other ) const { return !operator>(other); }
>>>>>>> ecuda2/master

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
<<<<<<< HEAD
	HOST DEVICE inline bool operator>=( const matrix<value_type,Alloc2>& other ) const { return !operator<(other); }

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
	DEVICE inline T& at( const size_type rowIndex, const size_type columnIndex ) { return *allocator.address( deviceMemory.get(), rowIndex, columnIndex, pitch ); }

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
	DEVICE inline const T& at( const size_type rowIndex, const size_type columnIndex ) const { return *allocator.address( deviceMemory.get(), rowIndex, columnIndex, pitch ); }

	/*
	 * Deprecating this function since the STL standard seems to specify that the at() accessor
	 * must implement range checking that throws an exception on failure.  Since exceptions are
	 * not supported within a CUDA kernel, this cannot be satisfied.
	 *
	DEVICE inline reference at( size_type rowIndex, size_type columnIndex ) {
		//if( rowIndex >= row_size() ) throw std::out_of_range( "ecuda::matrix::at() rowIndex parameter is out of range" );	
		//if( columnIndex >= column_size() ) throw std::out_of_range( "ecuda::matrix::at() columnIndex parameter is out of range" );	
		return *allocator.address( data(), rowIndex, columnIndex, pitch ); 
	}
	DEVICE inline reference at( size_type index ) { return at( index / numberColumns, index % numberColumns ); }

	DEVICE inline const_reference at( size_type rowIndex, size_type columnIndex ) const {
		//if( rowIndex >= row_size() ) throw std::out_of_range( "ecuda::matrix::at() rowIndex parameter is out of range" );	
		//if( columnIndex >= column_size() ) throw std::out_of_range( "ecuda::matrix::at() columnIndex parameter is out of range" );	
		return *allocator.address( data(), rowIndex, columnIndex, pitch ); 
	}
	DEVICE inline const_reference at( size_type index ) const { return at( index / numberColumns, index % numberColumns ); }
	*/

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
	HOST DEVICE matrix<value_type,allocator_type>& operator=( const matrix<value_type,Alloc2>& src ) {
		#ifdef __CUDA_ARCH__
		// shallow copy if called from device
		numberRows = src.numberRows;
		numberColumns = src.numberColumns;
		pitch = src.pitch;
		deviceMemory = src.deviceMemory;
		#else
		// deep copy if called from host
		numberRows = src.numberRows;
		numberColumns = src.numberColumns;
		deviceMemory = device_ptr<value_type>( allocator.allocate( numberColumns, numberRows, pitch ) );
		CUDA_CALL( cudaMemcpy2D<value_type>( deviceMemory.get(), pitch, src.deviceMemory.get(), src.pitch, numberColumns, numberRows, cudaMemcpyDeviceToDevice ) );
		#endif
		return *this;
	}
	*/

	///
	/// \brief Copies the contents of this device matrix to another container.
	///
	/// The matrix is converted into a row-major linearized form (all columns
	/// of the first row, then all columns of the second row, ...).
	///
	template<class Container>
	HOST Container& operator>>( Container& dest ) const {
		typename Container::iterator destIter = dest.begin();
		for( size_type i = 0; i < number_rows(); ++i ) {
			const_row_type row = get_row(i);
			::ecuda::copy( row.begin(), row.end(), destIter );
			::ecuda::advance( destIter, number_columns() );
		}
		return dest;
	}

	///
	/// \brief Copies the contents of another container to this device matrix.
	///
	/// The size of the container must match the number of elements in this
	/// matrix (number_rows()*number_columns()). The source container is assumed to
	/// be in row-major linear form (all columns of the first row, then all
	/// columns of the second row, ...).
	///
	/// \param src container to copy data from
	/// \throws std::length_error if number of elements in src does not match the size of this matrix
	///
	template<class Container>
	HOST matrix& operator<<( const Container& src ) {
		typename Container::const_iterator srcIter = src.begin();
		typename std::iterator_traits<typename Container::const_iterator>::difference_type len = ::ecuda::distance( src.begin(), src.end() );
		if( len < 0 or static_cast<size_type>(len) != size() ) throw std::length_error( EXCEPTION_MSG("ecuda::matrix::operator<<() provided with a container of non-matching size") );
		for( size_type i = 0; i < number_rows(); ++i ) {
			row_type row = get_row(i);
			typename Container::const_iterator srcEnd = srcIter;
			::ecuda::advance( srcEnd, number_columns() );
			row.assign( srcIter, srcEnd );
			srcIter = srcEnd;
		}
		return *this;
	}

};

=======
	__HOST__ __DEVICE__ inline bool operator>=( const matrix<value_type,Alloc2>& other ) const { return !operator<(other); }

};

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

///
/// A matrix subclass that should be used as the representation of a matrix within kernel code.
///
/// This achieves two objectives: 1) create a new cube object that is instantiated by creating
/// a shallow copy of the contents (so that older versions of the CUDA API that don't support
/// kernel pass-by-reference can specify containers in the function arguments), and 2) strip any
/// unnecessary data that will be useless to the kernel thus reducing register usage (in this
/// case by removing the unneeded reference-counting introduced by the internal shared_ptr).
///
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
>>>>>>> ecuda2/master

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
<<<<<<< HEAD
HOST void matrix_copy( matrix<T,Alloc1>& dest, const matrix<T,Alloc2>& src, typename matrix<T,Alloc2>::size_type offsetRow=0, typename matrix<T,Alloc2>::size_type offsetColumn=0 ) {
=======
__HOST__ void matrix_copy( matrix<T,Alloc1>& dest, const matrix<T,Alloc2>& src, typename matrix<T,Alloc2>::size_type offsetRow=0, typename matrix<T,Alloc2>::size_type offsetColumn=0 ) {
>>>>>>> ecuda2/master
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
<<<<<<< HEAD
HOST void matrix_swap(
=======
__HOST__ void matrix_swap(
>>>>>>> ecuda2/master
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
<<<<<<< HEAD
	ecuda::vector<T> stagingMemory( numberColumns );
=======
	std::vector< T, host_allocator<T> > stagingMemory( numberColumns );
>>>>>>> ecuda2/master
	typedef typename matrix<T,Alloc1>::size_type size_type;
	for( size_type i = 0; i < numberRows; ++i ) {
		typename matrix<T,Alloc1>::row_type row1 = mat1[offsetRow1+i];
		typename matrix<T,Alloc2>::row_type row2 = mat1[offsetRow2+i];
		stagingMemory.assign( row1.begin()+offsetColumn1, row1.begin()+(offsetColumn1+numberColumns) );
<<<<<<< HEAD
		::ecuda::copy( row2.begin()+offsetColumn2, row2.begin()+(offsetColumn2+numberColumns), row1.begin()+offsetColumn1 );
		::ecuda::copy( stagingMemory.begin(), stagingMemory.end(), row2.begin()+offsetColumn2 );
	}
=======
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
>>>>>>> ecuda2/master
}

} // namespace ecuda

#endif
