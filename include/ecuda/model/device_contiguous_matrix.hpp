/*
Copyright (c) 2016, Scott Zuyderduyn
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
// model/device_contiguous_matrix.hpp
//
// Lowest-level representation of a contiguous matrix in device memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MODEL_DEVICE_CONTIGUOUS_MATRIX_HPP
#define ECUDA_MODEL_DEVICE_CONTIGUOUS_MATRIX_HPP

#include "../global.hpp"
#include "../memory.hpp"
#include "../iterator.hpp"

#include "device_sequence.hpp"
#include "device_contiguous_sequence.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace model {

///
/// \brief Base representation of a contiguous device-bound matrix.
///
/// This class assumes the allocated memory is contiguous in order to function
/// properly, otherwise any operations will be undefined. The caller is
/// responsible for ensuring this.
///
template<typename T,class P=typename ecuda::add_pointer<T>::type>
class device_contiguous_matrix : public device_contiguous_sequence<T,P>
{
private:
	typedef device_contiguous_sequence<T,P> base_type;

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

	typedef device_contiguous_sequence<      value_type> row_type;
	typedef device_contiguous_sequence<const value_type> const_row_type;
	typedef device_sequence<       value_type, striding_ptr<      value_type,typename ecuda::add_pointer<      value_type>::type> > column_type;
	typedef device_sequence< const value_type, striding_ptr<const value_type,typename ecuda::add_pointer<const value_type>::type> > const_column_type;

private:
	size_type rows;

public:
	__HOST__ __DEVICE__ device_contiguous_matrix( pointer ptr = pointer(), size_type rows = 0, size_type columns = 0 ) : base_type(ptr,rows*columns), rows(rows) {}
	__HOST__ __DEVICE__ device_contiguous_matrix( const device_contiguous_matrix& src ) : base_type(src), rows(src.rows) {}
	template<typename U,class Q>	__HOST__ __DEVICE__ device_contiguous_matrix( const device_contiguous_matrix<U,Q>& src ) : base_type(src), rows(src.rows) {}
	__HOST__ device_contiguous_matrix& operator=( const device_contiguous_matrix& src )
	{
		base_type::operator=(src);
		rows = src.rows;
		return *this;
	}
	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ device_contiguous_matrix( device_contiguous_matrix&& src ) : base_type(std::move(src)), rows(std::move(src.rows)) {}
	__HOST__ device_contiguous_matrix& operator=( device_contiguous_matrix&& src )
	{
		base_type::operator=(std::move(src));
		rows = std::move(src.rows);
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline iterator       begin()        { return base_type::begin(); }
	__HOST__ __DEVICE__ inline iterator       end()          { return base_type::end(); }
	__HOST__ __DEVICE__ inline const_iterator begin() const  { return base_type::begin(); }
	__HOST__ __DEVICE__ inline const_iterator end() const    { return base_type::end(); }
	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ __DEVICE__ inline const_iterator cbegin() const { return base_type::cbegin(); }
	__HOST__ __DEVICE__ inline const_iterator cend() const   { return base_type::cend(); }
	#endif

	__HOST__ __DEVICE__ inline reverse_iterator       rbegin()        { return base_type::rbegin(); }
	__HOST__ __DEVICE__ inline reverse_iterator       rend()          { return base_type::rend(); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const  { return base_type::rbegin(); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const    { return base_type::rend(); }
	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin() const { return base_type::crbegin(); }
	__HOST__ __DEVICE__ inline const_reverse_iterator crend() const   { return base_type::crend(); }
	#endif

	__HOST__ __DEVICE__ inline size_type number_rows()    const { return rows; }
	__HOST__ __DEVICE__ inline size_type number_columns() const { return base_type::size()/rows; }

	__HOST__ __DEVICE__ row_type       get_row( const size_type row )       { return row_type(unmanaged_cast(base_type::get_pointer())+(row*number_columns()),number_columns()); }
	__HOST__ __DEVICE__ const_row_type get_row( const size_type row ) const { return const_row_type(unmanaged_cast(base_type::get_pointer())+(row*number_columns()),number_columns()); }

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

	__HOST__ __DEVICE__ inline const_column_type get_column( const size_type column ) const
	{
		return const_column_type(
					striding_ptr<const value_type,typename make_unmanaged_const<pointer>::type>(
						unmanaged_cast(base_type::get_pointer())+column, // move to top of column
						number_columns() // stride by number of columns
					),
					number_rows()
				);
	}

	__DEVICE__ inline reference operator()( const size_type row, const size_type column )
	{
		return *naked_cast<typename ecuda::add_pointer<value_type>::type>(base_type::get_pointer()+(row*number_columns())+column);
	}

	__DEVICE__ inline const_reference operator()( const size_type row, const size_type column ) const
	{
		return *naked_cast<typename ecuda::add_pointer<const value_type>::type>(base_type::get_pointer()+(row*number_columns())+column);
	}

	__DEVICE__ inline reference at( const size_type rowIndex, const size_type columnIndex )
	{
		if( rowIndex >= number_rows() || columnIndex >= number_columns() ) {
			#ifndef __CUDACC__
			throw std::out_of_range( EXCEPTION_MSG("ecuda::model::device_contiguous_matrix::at() row and/or column index parameter is out of range") );
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
			throw std::out_of_range( EXCEPTION_MSG("ecuda::model::device_contiguous_matrix::at() row and/or column index parameter is out of range") );
			#else
			// this strategy is taken from:
			// http://stackoverflow.com/questions/12521721/crashing-a-kernel-gracefully
			ecuda::threadfence();
			asm("trap;");
			#endif
		}
		return operator()(rowIndex,columnIndex);
	}

};

} // namespace model
/// \endcond

} // namespace ecuda

#endif
