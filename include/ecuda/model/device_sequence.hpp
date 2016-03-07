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
// model/device_sequence.hpp
//
// Lowest-level representation of a sequence in device memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MODEL_DEVICE_SEQUENCE_HPP
#define ECUDA_MODEL_DEVICE_SEQUENCE_HPP

#include "../global.hpp"
#include "../memory.hpp"
#include "../iterator.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace model {

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
	typedef T                                                   value_type;
	typedef P                                                   pointer;
	typedef typename ecuda::add_lvalue_reference<T>::type       reference;
	typedef typename ecuda::add_lvalue_reference<const T>::type const_reference;
	typedef std::size_t                                         size_type;
	typedef std::ptrdiff_t                                      difference_type;

	typedef device_iterator<      value_type,typename make_unmanaged<pointer>::type      > iterator;
	typedef device_iterator<const value_type,typename make_unmanaged_const<pointer>::type> const_iterator;

	typedef reverse_device_iterator<iterator      > reverse_iterator;
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator;

private:
	pointer ptr;
	size_type length;

	template<typename U,class Q> friend class device_sequence;

protected:
	__HOST__ __DEVICE__ inline pointer&       get_pointer()       ECUDA__NOEXCEPT { return ptr; }
	__HOST__ __DEVICE__ inline const pointer& get_pointer() const ECUDA__NOEXCEPT { return ptr; }

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

	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ device_sequence( device_sequence&& src ) : ptr(std::move(src.ptr)), length(std::move(src.length)) {}
	__HOST__ device_sequence& operator=( device_sequence&& src )
	{
		ptr = std::move(src.ptr);
		length = std::move(src.length);
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline size_type size() const ECUDA__NOEXCEPT { return length; }

	__DEVICE__ inline reference       operator[]( const size_type x )       { return *(unmanaged_cast( ptr ) + x); }
	__DEVICE__ inline const_reference operator[]( const size_type x ) const { return *(unmanaged_cast( ptr ) + x); }

	__HOST__ __DEVICE__ inline iterator       begin()        ECUDA__NOEXCEPT { return iterator( unmanaged_cast(ptr) ); }
	__HOST__ __DEVICE__ inline iterator       end()          ECUDA__NOEXCEPT { return iterator( unmanaged_cast(ptr) + size() ); }
	__HOST__ __DEVICE__ inline const_iterator begin() const  ECUDA__NOEXCEPT { return const_iterator( unmanaged_cast(ptr) ); }
	__HOST__ __DEVICE__ inline const_iterator end() const    ECUDA__NOEXCEPT { return const_iterator( unmanaged_cast(ptr) + size() ); }
	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ __DEVICE__ inline const_iterator cbegin() const __NOEXCEPT__ { return const_iterator( unmanaged_cast(ptr) ); }
	__HOST__ __DEVICE__ inline const_iterator cend() const   __NOEXCEPT__ { return const_iterator( unmanaged_cast(ptr) + size() ); }
	#endif

	__HOST__ __DEVICE__ inline reverse_iterator       rbegin()        ECUDA__NOEXCEPT { return reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline reverse_iterator       rend()          ECUDA__NOEXCEPT { return reverse_iterator(begin()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const  ECUDA__NOEXCEPT { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const    ECUDA__NOEXCEPT { return const_reverse_iterator(begin()); }
	#ifdef ECUDA_CPP11_AVAILABLE
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

} // namespace model
/// \endcond

} // namespace ecuda

#endif
