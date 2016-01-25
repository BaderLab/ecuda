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
// model/device_contiguous_sequence.hpp
//
// Lowest-level representation of a contiguous sequence in device memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MODEL_DEVICE_CONTIGUOUS_SEQUENCE_HPP
#define ECUDA_MODEL_DEVICE_CONTIGUOUS_SEQUENCE_HPP

#include "../global.hpp"
#include "../memory.hpp"
#include "../iterator.hpp"

#include "device_sequence.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace model {

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
	#ifdef ECUDA_CPP11_AVAILABLE
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
	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ __DEVICE__ inline const_iterator cbegin() const { return const_iterator( unmanaged_cast(base_type::get_pointer()) ); }
	__HOST__ __DEVICE__ inline const_iterator cend() const   { return const_iterator( unmanaged_cast(base_type::get_pointer()) + base_type::size() ); }
	#endif

	__HOST__ __DEVICE__ inline reverse_iterator       rbegin()        { return reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline reverse_iterator       rend()          { return reverse_iterator(begin()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const  { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const    { return const_reverse_iterator(begin()); }
	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin() const { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator crend() const   { return const_reverse_iterator(begin()); }
	#endif

};

} // namespace model
/// \endcond

} // namespace ecuda

#endif
