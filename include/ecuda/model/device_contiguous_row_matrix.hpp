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
// model/device_contiguous_row_matrix.hpp
//
// Lowest-level representation of a contiguous row matrix in device memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MODEL_DEVICE_CONTIGUOUS_ROW_MATRIX_HPP
#define ECUDA_MODEL_DEVICE_CONTIGUOUS_ROW_MATRIX_HPP

#include "../global.hpp"
#include "../memory.hpp"
#include "../iterator.hpp"

#include "device_sequence.hpp"
#include "device_contiguous_sequence.hpp"
#include "device_matrix.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace model {

///
/// \brief Base representation of a device-bound matrix where each row is contiguous.
///
/// This class enforces a pointer type of padded_ptr, which ensures the underlying
/// memory is contiguous in repeating blocks, where each block is followed by some
/// fixed padding.  This provides seamless support for device-aligned memory.
///
template<typename T,class P>
class device_contiguous_row_matrix : public device_matrix< T, padded_ptr<T,P> >
{
private:
	typedef device_matrix< T, padded_ptr<T,P> > base_type;

public:
	typedef typename base_type::value_type      value_type;
	typedef typename base_type::pointer         pointer;
	typedef typename base_type::reference       reference;
	typedef typename base_type::const_reference const_reference;
	typedef typename base_type::size_type       size_type;
	typedef typename base_type::difference_type difference_type;

	typedef device_contiguous_block_iterator<value_type,      typename make_unmanaged<P>::type      > iterator; // this iterator's 2nd template param is also padded_ptr<T,P>
	typedef device_contiguous_block_iterator<const value_type,typename make_unmanaged_const<P>::type> const_iterator;

	typedef reverse_device_iterator<iterator      > reverse_iterator;
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator;

	typedef device_contiguous_sequence<value_type      > row_type;
	typedef device_contiguous_sequence<const value_type> const_row_type;
	typedef device_sequence< value_type,       striding_padded_ptr< value_type,       typename ecuda::add_pointer<value_type>::type >       > column_type;
	typedef device_sequence< const value_type, striding_padded_ptr< const value_type, typename ecuda::add_pointer<const value_type>::type > > const_column_type;

public:
	__HOST__ __DEVICE__ device_contiguous_row_matrix( pointer ptr = pointer(), size_type rows = 0, size_type columns = 0 ) : base_type(ptr,rows,columns) {}

	__HOST__ __DEVICE__ device_contiguous_row_matrix( const device_contiguous_row_matrix& src ) : base_type(src) {}

	template<typename U,class PointerType2>	__HOST__ __DEVICE__ device_contiguous_row_matrix( const device_contiguous_row_matrix<U,PointerType2>& src ) : base_type(src) {}

	__HOST__ device_contiguous_row_matrix& operator=( const device_contiguous_row_matrix& src )
	{
		base_type::operator=(src);
		return *this;
	}

	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ device_contiguous_row_matrix( device_contiguous_row_matrix&& src ) : base_type(src) {}
	__HOST__ device_contiguous_row_matrix& operator=( device_contiguous_row_matrix&& src )
	{
		base_type::operator=(src);
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline iterator begin() ECUDA__NOEXCEPT { return iterator( unmanaged_cast(base_type::get_pointer()), base_type::number_columns() ); }
	__HOST__ __DEVICE__ iterator end() ECUDA__NOEXCEPT
	{
		typedef typename ecuda::make_unmanaged<pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type p = unmanaged_cast(base_type::get_pointer());
		p.skip_bytes( p.get_pitch()*base_type::number_rows() );
		return iterator( p, base_type::number_columns() );
	}
	__HOST__ __DEVICE__ inline const_iterator begin() const ECUDA__NOEXCEPT { return const_iterator( unmanaged_cast(base_type::get_pointer()), base_type::number_columns() ); }
	__HOST__ __DEVICE__ const_iterator end() const ECUDA__NOEXCEPT
	{
		typedef typename ecuda::make_unmanaged_const<pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type p = unmanaged_cast(base_type::get_pointer());
		p.skip_bytes( p.get_pitch()*base_type::number_rows() );
		return const_iterator( p, base_type::number_columns() );
	}
	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ __DEVICE__ inline const_iterator cbegin() const ECUDA__NOEXCEPT { return const_iterator( unmanaged_cast(base_type::get_pointer()) ); }
	__HOST__ __DEVICE__ const_iterator cend() const   ECUDA__NOEXCEPT
	{
		typedef typename ecuda::make_unmanaged_const<pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type p = unmanaged_cast(base_type::get_pointer());
		p.skip_bytes( p.get_pitch()*base_type::number_rows() );
		return const_iterator( p, base_type::number_columns() );
	}
	// return const_iterator( unmanaged_cast(base_type::get_pointer())+base_type::size() ); }
	#endif

	__HOST__ __DEVICE__ inline reverse_iterator       rbegin()        ECUDA__NOEXCEPT { return reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline reverse_iterator       rend()          ECUDA__NOEXCEPT { return reverse_iterator(begin()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const  ECUDA__NOEXCEPT { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const    ECUDA__NOEXCEPT { return const_reverse_iterator(begin()); }
	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin() const ECUDA__NOEXCEPT { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator crend() const   ECUDA__NOEXCEPT { return const_reverse_iterator(begin()); }
	#endif

	__HOST__ __DEVICE__ inline row_type get_row( const size_type row )
	{
		typedef typename make_unmanaged<pointer>::type unmanaged_pointer;
		unmanaged_pointer up = unmanaged_cast( base_type::get_pointer() ); // strip any mgmt by smart pointer
		up.skip_bytes( row*up.get_pitch() ); // advance to row start
		return row_type( naked_cast<typename ecuda::add_pointer<value_type>::type>( up ), base_type::number_columns() ); // provide naked pointer since row is contiguous
	}

	__HOST__ __DEVICE__ inline const_row_type get_row( const size_type row ) const
	{
		typedef typename make_unmanaged_const<pointer>::type unmanaged_pointer;
		unmanaged_pointer up = unmanaged_cast( base_type::get_pointer() ); // strip any mgmt by smart pointer
		up.skip_bytes( row*up.get_pitch() );
		return const_row_type( naked_cast<typename ecuda::add_pointer<const value_type>::type>( up ), base_type::number_columns() ); // provide naked pointer since row is contiguous
	}

	__HOST__ __DEVICE__ inline column_type get_column( const size_type column )
	{
		typedef typename ecuda::add_pointer<value_type>::type naked_pointer_type;
		naked_pointer_type np = naked_cast<naked_pointer_type>( unmanaged_cast(base_type::get_pointer()) + column );
		return column_type( striding_padded_ptr<value_type,naked_pointer_type>( np, base_type::get_pointer().get_pitch() ), base_type::number_rows() );
	}

	__HOST__ __DEVICE__ inline const_column_type get_column( const size_type column ) const
	{
		typedef typename ecuda::add_pointer<const value_type>::type naked_pointer_type;
		naked_pointer_type np = naked_cast<naked_pointer_type>( unmanaged_cast(base_type::get_pointer()) + column );
		return const_column_type( striding_padded_ptr<const value_type,naked_pointer_type>( np, base_type::get_pointer().get_pitch() ), base_type::number_rows() );
	}

	__HOST__ __DEVICE__ inline row_type       operator[]( const size_type row )       { return get_row(row); }
	__HOST__ __DEVICE__ inline const_row_type operator[]( const size_type row ) const { return get_row(row); }

	__DEVICE__ inline reference operator()( const size_type row, const size_type column )
	{
		typename make_unmanaged<pointer>::type up = unmanaged_cast(base_type::get_pointer());
		up.skip_bytes( up.get_pitch()*row );
		up.operator+=( column );
		//return *up;
		return *naked_cast<typename ecuda::add_pointer<value_type>::type>(up);
	}

	__DEVICE__ inline const_reference operator()( const size_type row, const size_type column ) const
	{
		typename make_unmanaged_const<pointer>::type up = unmanaged_cast(base_type::get_pointer());
		up.skip_bytes( up.get_pitch()*row );
		up.operator+=( column );
		//return *up;
		return *naked_cast<typename ecuda::add_pointer<const value_type>::type>(up);
	}

	__DEVICE__ inline reference at( const size_type rowIndex, const size_type columnIndex )
	{
		if( rowIndex >= base_type::number_rows() || columnIndex >= base_type::number_columns() ) {
			#ifndef __CUDACC__
			throw std::out_of_range( ECUDA_EXCEPTION_MSG("ecuda::model::device_contiguous_row_matrix::at() row and/or column index parameter is out of range") );
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
		if( rowIndex >= base_type::number_rows() || columnIndex >= base_type::number_columns() ) {
			#ifndef __CUDACC__
			throw std::out_of_range( ECUDA_EXCEPTION_MSG("ecuda::model::device_contiguous_row_matrix::at() row and/or column index parameter is out of range") );
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
