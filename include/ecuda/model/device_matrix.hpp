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
// model/device_matrix.hpp
//
// Lowest-level representation of a matrix in device memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MODEL_DEVICE_MATRIX_HPP
#define ECUDA_MODEL_DEVICE_MATRIX_HPP

#include "../global.hpp"
#include "../memory.hpp"
#include "../iterator.hpp"

#include "device_sequence.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace model {

///
/// \brief Base representation of a device-bound matrix.
///
/// This class makes no assumptions about the contiguity of the allocated memory.
/// The pointer specialization is fully responsible for traversing the matrix.
///
template<typename T,class P>
class device_matrix : public device_sequence<T,P>
{
private:
	typedef device_sequence<T,P> base_type;

public:
	typedef typename base_type::value_type      value_type;
	typedef typename base_type::pointer         pointer;
	typedef typename base_type::reference       reference;
	typedef typename base_type::const_reference const_reference;
	typedef typename base_type::size_type       size_type;
	typedef typename base_type::difference_type difference_type;

	typedef typename base_type::iterator               iterator;
	typedef typename base_type::const_iterator         const_iterator;
	typedef typename base_type::reverse_iterator       reverse_iterator;
	typedef typename base_type::const_reverse_iterator const_reverse_iterator;

	typedef device_sequence< value_type,       typename make_unmanaged<pointer>::type                                      > row_type;
	typedef device_sequence< const value_type, typename make_unmanaged_const<pointer>::type                                > const_row_type;
	typedef device_sequence< value_type,       striding_ptr<value_type,typename make_unmanaged<pointer>::type>             > column_type;
	typedef device_sequence< const value_type, striding_ptr<const value_type,typename make_unmanaged_const<pointer>::type> > const_column_type;

	template<typename U,typename Q> friend class device_matrix;

private:
	size_type rows;

public:
	__HOST__ __DEVICE__ device_matrix( pointer ptr = pointer(), size_type rows = 0, size_type columns = 0 ) : base_type(ptr,rows*columns), rows(rows) {}

	__HOST__ __DEVICE__ device_matrix( const device_matrix& src ) : base_type(src), rows(src.rows) {}

	template<typename U,typename Q> __HOST__ __DEVICE__ device_matrix( const device_matrix<U,Q>& src ) : base_type(src), rows(src.rows) {}

	__HOST__ device_matrix& operator=( const device_matrix& src ) {
		base_type::operator=(src);
		rows = src.rows;
		return *this;
	}

#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ device_matrix( device_matrix&& src ) : base_type(src), rows(std::move(src.rows)) {}
	__HOST__ device_matrix& operator=( device_matrix&& src )
	{
		base_type::operator=( src );
		rows = std::move(src.rows);
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline size_type number_rows() const    ECUDA__NOEXCEPT { return rows; }
	__HOST__ __DEVICE__ inline size_type number_columns() const ECUDA__NOEXCEPT { return base_type::size()/rows; }

	__HOST__ __DEVICE__ inline row_type get_row( const size_type row )
	{
		return row_type(
					unmanaged_cast(base_type::get_pointer())+(row*number_columns()),
					number_columns()
				);
	}

	__HOST__ __DEVICE__ inline const_row_type get_row( const size_type row ) const
	{
		return const_row_type(
					unmanaged_cast(base_type::get_pointer())+(row*number_columns()),
					number_columns()
				);
	}

	__HOST__ __DEVICE__ inline column_type get_column( const size_type column )
	{
		return column_type(
					striding_ptr<value_type,typename make_unmanaged<pointer>::type>(
						unmanaged_cast(base_type::get_pointer())+column, // move to top of column
						number_columns() // stride by number of columns
					),
					number_rows()
				);
	}

	__HOST__ __DEVICE__ inline const_column_type get_column( const size_type column ) const {
		return const_column_type(
					striding_ptr<const value_type,typename make_unmanaged_const<pointer>::type>(
						unmanaged_cast(base_type::get_pointer())+column, // move to top of column
						number_columns() // stride by number of columns
					),
					number_rows()
				);
	}

	__HOST__ __DEVICE__ inline row_type       operator[]( const size_type row )       { return get_row(row); }
	__HOST__ __DEVICE__ inline const_row_type operator[]( const size_type row ) const { return get_row(row); }

	__DEVICE__ inline reference operator()( const size_type row, const size_type column )
	{
		typename make_unmanaged<P>::type p = unmanaged_cast(base_type::get_pointer());
		p += row*number_columns();
		p += column;
		return *p;
	}

	__DEVICE__ inline const_reference operator()( const size_type row, const size_type column ) const
	{
		typename make_unmanaged_const<P>::type p = unmanaged_cast(base_type::get_pointer());
		p += row*number_columns();
		p += column;
		return *p;
	}

	__DEVICE__ inline reference at( const size_type rowIndex, const size_type columnIndex )
	{
		if( rowIndex >= number_rows() || columnIndex >= number_columns() ) {
			#ifndef __CUDACC__
			throw std::out_of_range( ECUDA_EXCEPTION_MSG("ecuda::model::device_matrix::at() row and/or column index parameter is out of range") );
			#else
			// this strategy is taken from:
			// http://stackoverflow.com/questions/12521721/crashing-a-kernel-gracefully
			ecuda::threadfence();
			asm("trap;");
			#endif
		}
		return operator()(rowIndex,columnIndex);
	}

	__DEVICE__ inline const_reference at( const size_type rowIndex, const size_type columnIndex ) const
	{
		if( rowIndex >= number_rows() || columnIndex >= number_columns() ) {
			#ifndef __CUDACC__
			throw std::out_of_range( ECUDA_EXCEPTION_MSG("ecuda::model::device_matrix::at() row and/or column index parameter is out of range") );
			#else
			// this strategy is taken from:
			// http://stackoverflow.com/questions/12521721/crashing-a-kernel-gracefully
			ecuda::threadfence();
			asm("trap;");
			#endif
		}
		return operator()(rowIndex,columnIndex);
	}

	__HOST__ __DEVICE__ void swap( device_matrix& other )
	{
		base_type::swap( other );
		::ecuda::swap( rows, other.rows );
	}

};

} // namespace model
/// \endcond

} // namespace ecuda

#endif
