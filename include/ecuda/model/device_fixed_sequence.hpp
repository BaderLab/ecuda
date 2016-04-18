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
// model/device_fixed_sequence.hpp
//
// Lowest-level representation of a fixed-size sequence in device memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MODEL_DEVICE_FIXED_SEQUENCE_HPP
#define ECUDA_MODEL_DEVICE_FIXED_SEQUENCE_HPP

#include "../global.hpp"
#include "../memory.hpp"
#include "../iterator.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace model {

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
	typedef T                                                   value_type;
	typedef P                                                   pointer;
	typedef typename ecuda::add_lvalue_reference<T>::type       reference;
	typedef typename ecuda::add_lvalue_reference<const T>::type const_reference;
	typedef std::size_t                                         size_type;
	typedef std::ptrdiff_t                                      difference_type;

	typedef device_contiguous_iterator<value_type      > iterator;
	typedef device_contiguous_iterator<const value_type> const_iterator;

	typedef reverse_device_iterator<iterator      > reverse_iterator;
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator;

	template<typename U,std::size_t M,typename Q> friend class device_fixed_sequence;

private:
	pointer ptr;

protected:
	__HOST__ __DEVICE__ inline pointer& get_pointer() { return ptr; }
	__HOST__ __DEVICE__ inline const pointer& get_pointer() const { return ptr; }

public:
	__HOST__ __DEVICE__ device_fixed_sequence( pointer ptr = pointer() ) : ptr(ptr) {}

	__HOST__ __DEVICE__ device_fixed_sequence( const device_fixed_sequence& src ) : ptr(src.ptr) {}

	template<typename U,typename Q> __HOST__ __DEVICE__ device_fixed_sequence( const device_fixed_sequence<U,N,Q>& src ) : ptr(src.ptr) {}

	__HOST__ device_fixed_sequence& operator=( const device_fixed_sequence& src )
	{
		ptr = src.ptr;
		return *this;
	}
	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ device_fixed_sequence( device_fixed_sequence&& src ) { ecuda::swap( ptr, src.ptr ); }
	__HOST__ device_fixed_sequence& operator=( device_fixed_sequence&& src )
	{
		ecuda::swap( ptr, src.ptr );
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline ECUDA__CONSTEXPR size_type size() const { return N; }

	__DEVICE__ inline reference       operator()( const size_type x )       { return operator[](x); }
	__DEVICE__ inline const_reference operator()( const size_type x ) const { return operator[](x); }

	__DEVICE__ inline reference       operator[]( const size_type x )       { return *(unmanaged_cast( ptr ) + x); }
	__DEVICE__ inline const_reference operator[]( const size_type x ) const { return *(unmanaged_cast( ptr ) + x); }

	__DEVICE__ inline reference at( const size_type x )
	{
		if( x >= size() ) {
			#ifndef __CUDACC__
			throw std::out_of_range( ECUDA_EXCEPTION_MSG("ecuda::model::device_fixed_sequence::at() index parameter is out of range") );
			#else
			// this strategy is taken from:
			// http://stackoverflow.com/questions/12521721/crashing-a-kernel-gracefully
			ecuda::threadfence();
			asm("trap;");
			#endif
		}
		return operator()(x);
	}

	__DEVICE__ inline const_reference at( const size_type x ) const
	{
		if( x >= size() ) {
			#ifndef __CUDACC__
			throw std::out_of_range( ECUDA_EXCEPTION_MSG("ecuda::model::device_fixed_sequence::at() index parameter is out of range") );
			#else
			// this strategy is taken from:
			// http://stackoverflow.com/questions/12521721/crashing-a-kernel-gracefully
			ecuda::threadfence();
			asm("trap;");
			#endif
		}
		return operator()(x);
	}

	__HOST__ __DEVICE__ inline iterator       begin()        { return iterator( unmanaged_cast(ptr) ); }
	__HOST__ __DEVICE__ inline iterator       end()          { return iterator( unmanaged_cast(ptr) + N ); }
	__HOST__ __DEVICE__ inline const_iterator begin() const  { return const_iterator( unmanaged_cast(ptr) ); }
	__HOST__ __DEVICE__ inline const_iterator end() const    { return const_iterator( unmanaged_cast(ptr) + N ); }
	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ __DEVICE__ inline const_iterator cbegin() const { return const_iterator( unmanaged_cast(ptr) ); }
	__HOST__ __DEVICE__ inline const_iterator cend() const   { return const_iterator( unmanaged_cast(ptr) + N ); }
	#endif

	__HOST__ __DEVICE__ inline reverse_iterator       rbegin()        { return reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline reverse_iterator       rend()          { return reverse_iterator(begin()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rbegin() const  { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator rend() const    { return const_reverse_iterator(begin()); }
	#ifdef ECUDA_CPP11_AVAILABLE
	__HOST__ __DEVICE__ inline const_reverse_iterator crbegin() const { return const_reverse_iterator(end()); }
	__HOST__ __DEVICE__ inline const_reverse_iterator crend() const   { return const_reverse_iterator(begin()); }
	#endif

	__HOST__ __DEVICE__ inline void swap( device_fixed_sequence& other ) { ecuda::swap( ptr, other.ptr ); }

};

} // namespace model
/// \endcond

} // namespace ecuda

#endif
