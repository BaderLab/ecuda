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
// impl/models.hpp
//
// Lowest-level representations of data structures stored in video memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MODELS_HPP
#define ECUDA_MODELS_HPP

#include "../global.hpp"
#include "../memory.hpp"
#include "../iterator.hpp"

///
/// ecuda models contain the lowest-level representation of data structures stored
/// in video memory.
///

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

///
/// \brief Base representation of a sequence in device memory.
///
/// The class stores a pointer (raw or specialized) to the beginning of the sequence
/// and the length of the sequence.
///
/// This class makes no assumptions about the contiguity of the allocated memory.
/// I.e. ( stored pointer + length ) doesn't necessarily refer to an
///      address length*size(T) away.
///
/// Responsibility for the logic required to traverse the sequence element-by-element
/// is delegated to the pointer specialization. This allows higher-level classes to
/// re-use this structure to represent arrays, matrix rows and columns, and so on.
///
template<typename T,class P>
class device_sequence
{

public:
	typedef T                                             value_type;
	typedef P                                             pointer;
	typedef typename ecuda::add_lvalue_reference<T>::type reference;
	typedef typename ecuda::add_const<reference>::type    const_reference;
	typedef std::size_t                                   size_type;
	typedef std::ptrdiff_t                                difference_type;

	typedef device_iterator<      value_type,typename make_unmanaged<pointer>::type      > iterator;
	typedef device_iterator<const value_type,typename make_unmanaged_const<pointer>::type> const_iterator;

	typedef reverse_device_iterator<iterator      > reverse_iterator;
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator;

private:
	pointer ptr;
	size_type length;

	template<typename U,class Q> friend class device_sequence;

protected:
	__HOST__ __DEVICE__ inline pointer&       get_pointer()       __NOEXCEPT__ { return ptr; }
	__HOST__ __DEVICE__ inline const pointer& get_pointer() const __NOEXCEPT__ { return ptr; }

public:
	__HOST__ __DEVICE__ device_sequence( pointer ptr = pointer(), size_type length = 0 ) : ptr(ptr), length(length) {}

	__HOST__ __DEVICE__ device_sequence( const device_sequence& src ) : ptr(src.ptr), length(src.length) {}

	template<typename U,class Q> __HOST__ __DEVICE__ device_sequence( const device_sequence<U,Q>& src ) : ptr(src.ptr), length(src.length) {}

	__HOST__ device_sequence& operator=( const device_sequence& src )
	{
		ptr = src.ptr;
		length = src.length;
		return *this;
	}

	#ifdef __CPP11_SUPPORTED__
	__HOST__ device_sequence( device_sequence&& src ) : ptr(std::move(src.ptr)), length(std::move(src.length)) {}
	__HOST__ device_sequence& operator=( device_sequence&& src )
	{
		ptr = std::move(src.ptr);
		length = std::move(src.length);
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline size_type size() const __NOEXCEPT__ { return length; }

	__DEVICE__ inline reference       operator[]( const size_type x )       { return *(unmanaged_cast( ptr ) + x); }
	__DEVICE__ inline const_reference operator[]( const size_type x ) const { return *(unmanaged_cast( ptr ) + x); }

	__HOST__ __DEVICE__ inline iterator       begin()        __NOEXCEPT__ { return iterator( unmanaged_cast(ptr) ); }
	__HOST__ __DEVICE__ inline iterator       end()          __NOEXCEPT__ { return iterator( unmanaged_cast(ptr) + size() ); }
	__HOST__ __DEVICE__ inline const_iterator begin() const  __NOEXCEPT__ { return const_iterator( unmanaged_cast(ptr) ); }
	__HOST__ __DEVICE__ inline const_iterator end() const    __NOEXCEPT__ { return const_iterator( unmanaged_cast(ptr) + size() ); }
	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ inline const_iterator cbegin() const __NOEXCEPT__ { return const_iterator( unmanaged_cast(ptr) ); }
	__HOST__ __DEVICE__ inline const_iterator cend() const   __NOEXCEPT__ { return const_iterator( unmanaged_cast(ptr) + size() ); }
	#endif

	__HOST__ __DEVICE__ inline reverse_iterator       rbegin()        __NOEXCEPT__ { return reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline reverse_iterator       rend()          __NOEXCEPT__ { return reverse_iterator(begin()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const  __NOEXCEPT__ { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const    __NOEXCEPT__ { return const_reverse_iterator(begin()); }
	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin() const __NOEXCEPT__ { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator crend() const   __NOEXCEPT__ { return const_reverse_iterator(begin()); }
	#endif

	//TODO: think about this - worth considering for user access
	//__HOST__ __DEVICE__ inline typename ecuda::add_pointer<value_type>::type data() const { return naked_cast<typename ecuda::add_pointer<value_type>::type>( ptr ); }
	//__HOST__ __DEVICE__ inline pointer data() const { return ptr; }

	__HOST__ __DEVICE__ void swap( device_sequence& other )
	{
		::ecuda::swap( ptr, other.ptr );
		::ecuda::swap( length, other.length );
	}

};

///
/// \brief Base representation of a fixed-size device-bound sequence.
///
/// This class assumes the allocated memory is contiguous in order to function
/// properly, otherwise any operations will be undefined. The caller is
/// responsible for ensuring this.
///
template<typename T,std::size_t N,class P=typename ecuda::add_pointer<T>::type>
class device_fixed_sequence
{

public:
	typedef T                                             value_type;
	typedef P                                             pointer;
	typedef typename ecuda::add_lvalue_reference<T>::type reference;
	typedef typename ecuda::add_const<reference>::type    const_reference;
	typedef std::size_t                                   size_type;
	typedef std::ptrdiff_t                                difference_type;

	typedef device_contiguous_iterator<value_type      > iterator;
	typedef device_contiguous_iterator<const value_type> const_iterator;

	typedef reverse_device_iterator<iterator      > reverse_iterator;
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator;

private:
	pointer ptr;

protected:
	__HOST__ __DEVICE__ inline pointer& get_pointer() { return ptr; }
	__HOST__ __DEVICE__ inline const pointer& get_pointer() const { return ptr; }

public:
	__HOST__ __DEVICE__ device_fixed_sequence( pointer ptr = pointer() ) : ptr(ptr) {}
	__HOST__ __DEVICE__ device_fixed_sequence( const device_fixed_sequence& src ) : ptr(src.ptr) {}
	__HOST__ device_fixed_sequence& operator=( const device_fixed_sequence& src )
	{
		ptr = src.ptr;
		return *this;
	}
	#ifdef __CPP11_SUPPORTED__
	__HOST__ device_fixed_sequence( device_fixed_sequence&& src ) { ecuda::swap( ptr, src.ptr ); }
	__HOST__ device_fixed_sequence& operator=( device_fixed_sequence&& src )
	{
		ecuda::swap( ptr, src.ptr );
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline __CONSTEXPR__ size_type size() const { return N; }

	__DEVICE__ inline reference       operator[]( const size_type x )       { return *(unmanaged_cast( ptr ) + x); }
	__DEVICE__ inline const_reference operator[]( const size_type x ) const { return *(unmanaged_cast( ptr ) + x); }

	__HOST__ __DEVICE__ inline iterator       begin()        { return iterator( unmanaged_cast(ptr) ); }
	__HOST__ __DEVICE__ inline iterator       end()          { return iterator( unmanaged_cast(ptr) + N ); }
	__HOST__ __DEVICE__ inline const_iterator begin() const  { return const_iterator( unmanaged_cast(ptr) ); }
	__HOST__ __DEVICE__ inline const_iterator end() const    { return const_iterator( unmanaged_cast(ptr) + N ); }
	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ inline const_iterator cbegin() const { return const_iterator( unmanaged_cast(ptr) ); }
	__HOST__ __DEVICE__ inline const_iterator cend() const   { return const_iterator( unmanaged_cast(ptr) + N ); }
	#endif

	__HOST__ __DEVICE__ inline reverse_iterator       rbegin()        { return reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline reverse_iterator       rend()          { return reverse_iterator(begin()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const  { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const    { return const_reverse_iterator(begin()); }
	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin() const { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator crend() const   { return const_reverse_iterator(begin()); }
	#endif

	__HOST__ __DEVICE__ inline void swap( device_fixed_sequence& other ) { ecuda::swap( ptr, other.ptr ); }

};

///
/// \brief Base representation of a contiguous device-bound sequence.
///
/// This class assumes the allocated memory is contiguous in order to function
/// properly, otherwise any operations will be undefined. The caller is
/// responsible for ensuring this.
///
template<typename T,class P=typename ecuda::add_pointer<T>::type>
class device_contiguous_sequence : public device_sequence<T,P>
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

	typedef device_contiguous_iterator<value_type>       iterator;
	typedef device_contiguous_iterator<const value_type> const_iterator;
	typedef reverse_device_iterator<iterator>            reverse_iterator;
	typedef reverse_device_iterator<const_iterator>      const_reverse_iterator;

public:
	__HOST__ __DEVICE__ device_contiguous_sequence( pointer ptr = pointer(), size_type length = 0 ) : base_type(ptr,length) {}
	__HOST__ __DEVICE__ device_contiguous_sequence( const device_contiguous_sequence& src ) : base_type(src) {}
	template<typename U,class PointerType2>	__HOST__ __DEVICE__ device_contiguous_sequence( const device_contiguous_sequence<U,PointerType2>& src ) : base_type(src) {}
	__HOST__ device_contiguous_sequence& operator=( const device_contiguous_sequence& src )
	{
		base_type::operator=(src);
		return *this;
	}
	#ifdef __CPP11_SUPPORTED__
	__HOST__ device_contiguous_sequence( device_contiguous_sequence&& src ) : base_type(src) {}
	__HOST__ device_contiguous_sequence& operator=( device_contiguous_sequence&& src )
	{
		base_type::operator=(src);
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline iterator       begin()        { return iterator( unmanaged_cast(base_type::get_pointer()) ); }
	__HOST__ __DEVICE__ inline iterator       end()          { return iterator( unmanaged_cast(base_type::get_pointer()) + base_type::size() ); }
	__HOST__ __DEVICE__ inline const_iterator begin() const  { return const_iterator( unmanaged_cast(base_type::get_pointer()) ); }
	__HOST__ __DEVICE__ inline const_iterator end() const    { return const_iterator( unmanaged_cast(base_type::get_pointer()) + base_type::size() ); }
	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ inline const_iterator cbegin() const { return const_iterator( unmanaged_cast(base_type::get_pointer()) ); }
	__HOST__ __DEVICE__ inline const_iterator cend() const   { return const_iterator( unmanaged_cast(base_type::get_pointer()) + base_type::size() ); }
	#endif

	__HOST__ __DEVICE__ inline reverse_iterator       rbegin()        { return reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline reverse_iterator       rend()          { return reverse_iterator(begin()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const  { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const    { return const_reverse_iterator(begin()); }
	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin() const { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator crend() const   { return const_reverse_iterator(begin()); }
	#endif

};

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

private:
	size_type rows;

public:
	__HOST__ __DEVICE__ device_matrix( pointer ptr = pointer(), size_type rows = 0, size_type columns = 0 ) : base_type(ptr,rows*columns), rows(rows) {}
	__HOST__ __DEVICE__ device_matrix( const device_matrix& src ) : base_type(src), rows(src.rows) {}
	__HOST__ device_matrix& operator=( const device_matrix& src ) {
		base_type::operator=(src);
		rows = src.rows;
		return *this;
	}
	#ifdef __CPP11_SUPPORTED__
	__HOST__ device_matrix( device_matrix&& src ) : base_type(src), rows(std::move(src.rows)) {}
	__HOST__ device_matrix& operator=( device_matrix&& src )
	{
		base_type::operator=( src );
		rows = std::move(src.rows);
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline size_type number_rows() const    __NOEXCEPT__ { return rows; }
	__HOST__ __DEVICE__ inline size_type number_columns() const __NOEXCEPT__ { return base_type::size()/rows; }

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

	__HOST__ __DEVICE__ void swap( device_matrix& other )
	{
		base_type::swap( other );
		::ecuda::swap( rows, other.rows );
	}

};

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

	#ifdef __CPP11_SUPPORTED__
	__HOST__ device_contiguous_row_matrix( device_contiguous_row_matrix&& src ) : base_type(src) {}
	__HOST__ device_contiguous_row_matrix& operator=( device_contiguous_row_matrix&& src )
	{
		base_type::operator=(src);
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline iterator begin() __NOEXCEPT__ { return iterator( unmanaged_cast(base_type::get_pointer()), base_type::number_columns() ); }
	__HOST__ __DEVICE__ iterator end() __NOEXCEPT__
	{
		typedef typename ecuda::make_unmanaged<pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type p = unmanaged_cast(base_type::get_pointer());
		p.skip_bytes( p.get_pitch()*base_type::number_rows() );
		return iterator( p, base_type::number_columns() );
	}
	__HOST__ __DEVICE__ inline const_iterator begin() const __NOEXCEPT__ { return const_iterator( unmanaged_cast(base_type::get_pointer()), base_type::number_columns() ); }
	__HOST__ __DEVICE__ const_iterator end() const __NOEXCEPT__
	{
		typedef typename ecuda::make_unmanaged_const<pointer>::type unmanaged_pointer_type;
		unmanaged_pointer_type p = unmanaged_cast(base_type::get_pointer());
		p.skip_bytes( p.get_pitch()*base_type::number_rows() );
		return const_iterator( p, base_type::number_columns() );
	}
	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ inline const_iterator cbegin() const __NOEXCEPT__ { return const_iterator( unmanaged_cast(base_type::get_pointer()) ); }
	__HOST__ __DEVICE__ inline const_iterator cend() const   __NOEXCEPT__ { return const_iterator( unmanaged_cast(base_type::get_pointer())+base_type::size() ); }
	#endif

	__HOST__ __DEVICE__ inline reverse_iterator       rbegin()        __NOEXCEPT__ { return reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline reverse_iterator       rend()          __NOEXCEPT__ { return reverse_iterator(begin()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const  __NOEXCEPT__ { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const    __NOEXCEPT__ { return const_reverse_iterator(begin()); }
	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin() const __NOEXCEPT__ { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator crend() const   __NOEXCEPT__ { return const_reverse_iterator(begin()); }
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

	__DEVICE__ inline reference at( const size_type row, const size_type column )
	{
		typename make_unmanaged<pointer>::type up = unmanaged_cast(base_type::get_pointer());
		up.skip_bytes( up.get_pitch()*row );
		up.operator+=( column );
		return *naked_cast<typename ecuda::add_pointer<value_type>::type>(up);
	}

	__DEVICE__ inline const_reference at( const size_type row, const size_type column ) const
	{
		typename make_unmanaged_const<pointer>::type up = unmanaged_cast(base_type::get_pointer());
		up.skip_bytes( up.get_pitch()*row );
		up.operator+=( column );
		return *naked_cast<typename ecuda::add_pointer<const value_type>::type>(up);
	}

};

} // namespace impl
/// \endcond

} // namespace ecuda

#endif
