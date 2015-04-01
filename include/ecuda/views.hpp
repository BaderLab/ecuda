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
// views.hpp
//
// Provides specialized view of the data within a container.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_VIEWS_HPP
#define ECUDA_VIEWS_HPP

#include <cstddef>
#include "global.hpp"
#include "apiwrappers.hpp"
#include "iterators.hpp"
#include "device_ptr.hpp"
#include "striding_ptr.hpp"
#include "padded_ptr.hpp"

#include <algorithm>
#ifdef __CPP11_SUPPORTED__
#include <initializer_list>
#endif
#include <limits>
#include <vector>

namespace ecuda {

template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type>
class device_memory_sequence
{
public:
	typedef T value_type; //!< element data type
	typedef PointerType pointer; //!< element pointer type
	typedef value_type& reference; //!< element reference type
	typedef const value_type& const_reference; //!< const element reference type
	typedef std::size_t size_type; //!< unsigned integral type
	typedef std::ptrdiff_t difference_type; //!< signed integral type

	typedef device_iterator<value_type,pointer> iterator; //!< iterator type
	typedef device_iterator<const value_type,/*const*/ pointer> const_iterator; //!< const iterator type
	typedef reverse_device_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

private:
	pointer ptr; //!< pointer to the start of the sequence
	size_type length; //!< number of elements in the sequence

protected:
	HOST DEVICE inline pointer& get_pointer() { return ptr; }
	HOST DEVICE inline const pointer& get_pointer() const { return ptr; }

public:
	HOST DEVICE device_memory_sequence() : ptr(nullptr), length(0) {}
	template<typename T2,typename PointerType2>
	HOST DEVICE device_memory_sequence( const device_memory_sequence<T2,PointerType2>& src ) : ptr(src.data()), length(src.size()) {}
	HOST DEVICE device_memory_sequence( pointer ptr, const size_type length ) : ptr(ptr), length(length) {}
	//HOST DEVICE ~device_memory_sequence() {}

	#ifdef __CPP11_SUPPORTED__
	HOST DEVICE device_memory_sequence( device_memory_sequence&& src ) : ptr(std::move(ptr)), length(std::move(length)) {}
	#endif

	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(ptr); }
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator(ptr+length); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(ptr); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(ptr+length); }

	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(end()); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(begin()); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(end()); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(begin()); }

	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return length; }
	HOST DEVICE __CONSTEXPR__ inline size_type max_size() const __NOEXCEPT__ { return std::numeric_limits<size_type>::max(); }
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return !length; }

	DEVICE inline reference operator[]( const size_type index ) { return *(ptr+index); }
	DEVICE inline const_reference operator[]( const size_type index ) const { return *(ptr+index); }

	DEVICE inline reference front() { return *ptr; }
	DEVICE inline reference back() { return *(ptr+(length-1)); }
	DEVICE inline const_reference front() const { return *ptr; }
	DEVICE inline const_reference back() const { return *(ptr+(length-1)); }

	HOST DEVICE inline pointer data() const __NOEXCEPT__ { return ptr; }

	template<class DeviceIterator>
	DEVICE void assign( DeviceIterator first, DeviceIterator last ) {
		for( iterator dest = begin(); dest != end() and first != last; ++dest, ++first ) *dest = *first;
	}

	HOST DEVICE void swap( device_memory_sequence& other ) {
		#ifdef __CUDA_ARCH__
		ecuda::swap( ptr, other.ptr );
		ecuda::swap( length, other.length );
		#else
		std::swap( ptr, other.ptr );
		std::swap( length, other.length );
		#endif
	}

	DEVICE bool operator==( const device_memory_sequence& other ) const {
		if( size() != other.size() ) return false;
		const_iterator iter1 = begin();
		const_iterator iter2 = other.begin();
		for( ; iter1 != end(); ++iter1, ++iter2 ) if( !(*iter1 == *iter2) ) return false;
		return true;
	}

	DEVICE inline bool operator!=( const device_memory_sequence& other ) const { return !operator==( other ); }

	DEVICE inline bool operator<( const device_memory_sequence& other ) const { return ecuda::lexicographical_compare( begin(), end(), other.begin(), other.end() ); }

	DEVICE inline bool operator>( const device_memory_sequence& other ) const { return ecuda::lexicographical_compare( other.begin(), other.end(), begin(), end() ); }

	DEVICE inline bool operator<=( const device_memory_sequence& other ) const { return !operator>(other); }

	DEVICE inline bool operator>=( const device_memory_sequence& other ) const { return !operator<(other); }

	DEVICE HOST device_memory_sequence& operator=( device_memory_sequence& src ) {
		ptr = src.ptr;
		length = src.length;
		return *this;
	}

};

template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type>
class device_contiguous_memory_sequence : public device_memory_sequence<T,PointerType>
{
private:
	typedef device_memory_sequence<T,PointerType> base_type;

public:
	typedef typename base_type::value_type value_type; //!< element data type
	typedef typename base_type::pointer pointer; //!< element pointer type
	typedef typename base_type::reference reference; //!< element reference type
	typedef typename base_type::const_reference const_reference; //!< const element reference type
	typedef typename base_type::size_type size_type; //!< unsigned integral type
	typedef typename base_type::difference_type difference_type; //!< signed integral type

	typedef contiguous_device_iterator<value_type> iterator; //!< iterator type
	typedef contiguous_device_iterator<const value_type> const_iterator; //!< const iterator type
	typedef reverse_device_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

	typedef typename std::vector<T>::const_iterator HostVectorConstIterator;
	typedef typename std::vector<T>::iterator HostVectorIterator;
	typedef const_iterator DeviceContiguousConstIterator;
	typedef iterator DeviceContiguousIterator;
	//typedef typename std::iterator<std::random_access_iterator_tag,T,std::ptrdiff_t,T*,T&> HostRandomAccessIterator;

private:
	typedef std::vector< value_type, host_allocator<value_type> > StagingVector;
	inline void copy_to_staging( StagingVector& v ) const {
		v.resize( base_type::size() );
		CUDA_CALL( cudaMemcpy<value_type>( &v.front(), base_type::data(), base_type::size(), cudaMemcpyDeviceToHost ) );
	}

public:
	HOST DEVICE device_contiguous_memory_sequence() : device_memory_sequence<T,PointerType>() {}
	template<typename T2>
	HOST DEVICE device_contiguous_memory_sequence( const device_contiguous_memory_sequence<T2>& src ) : device_memory_sequence<T,PointerType>( src ) {}
	HOST DEVICE device_contiguous_memory_sequence( pointer ptr, const size_type length ) : device_memory_sequence<T,PointerType>( ptr, length ) {}
	//HOST DEVICE ~device_contiguous_memory_sequence() {}

	#ifdef __CPP11_SUPPORTED__
	HOST DEVICE device_contiguous_memory_sequence( device_contiguous_memory_sequence&& src ) : device_memory_sequence( src ) {}
	#endif

	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(base_type::get_pointer()); }
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator(base_type::get_pointer()+base_type::size()); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(base_type::get_pointer()); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(base_type::get_pointer()+base_type::size()); }

	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(end()); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(begin()); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(end()); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(begin()); }

	HOST DEVICE void assign( DeviceContiguousIterator first, DeviceContiguousIterator last ) {
		#ifdef __CUDA_ARCH__
		iterator dest = begin();
		for( iterator dest = begin(); dest != end() and first != last; ++dest, ++first ) *dest = *first;
		#else
		const difference_type n = last-first;
		if( n < 0 ) throw std::length_error( "ecuda::device_contiguous_memory_sequence::assign() given iterator-based range oriented in wrong direction (are begin and end mixed up?)" );
		CUDA_CALL( cudaMemcpy<value_type>( base_type::data(), first.operator->(), std::min(static_cast<size_type>(n),base_type::size()), cudaMemcpyDeviceToDevice ) );
		#endif
	}

	HOST DEVICE void assign( DeviceContiguousConstIterator first, DeviceContiguousConstIterator last ) {
		#ifdef __CUDA_ARCH__
		iterator dest = begin();
		for( iterator dest = begin(); dest != end() and first != last; ++dest, ++first ) *dest = *first;
		#else
		const difference_type n = last-first;
		if( n < 0 ) throw std::length_error( "ecuda::device_contiguous_memory_sequence::assign() given iterator-based range oriented in wrong direction (are begin and end mixed up?)" );
		CUDA_CALL( cudaMemcpy<value_type>( base_type::data(), first.operator->(), std::min(static_cast<size_type>(n),base_type::size()), cudaMemcpyDeviceToDevice ) );
		#endif
	}

	HOST void assign( HostVectorIterator first, HostVectorIterator last ) {
		const difference_type n = last-first;
		if( n < 0 ) throw std::length_error( "ecuda::device_contiguous_memory_sequence::assign() given iterator-based range oriented in wrong direction (are begin and end mixed up?)" );
		CUDA_CALL( cudaMemcpy<value_type>( base_type::data(), first.operator->(), std::min(static_cast<size_type>(n),base_type::size()), cudaMemcpyHostToDevice ) );
	}

	HOST void assign( HostVectorConstIterator first, HostVectorConstIterator last ) {
		const difference_type n = last-first;
		if( n < 0 ) throw std::length_error( "ecuda::device_contiguous_memory_sequence::assign() given iterator-based range oriented in wrong direction (are begin and end mixed up?)" );
		CUDA_CALL( cudaMemcpy<value_type>( base_type::data(), first.operator->(), std::min(static_cast<size_type>(n),base_type::size()), cudaMemcpyHostToDevice ) );
	}

	template<class InputIterator>
	HOST void assign( InputIterator first, InputIterator last ) {
		StagingVector v( first, last );
		CUDA_CALL( cudaMemcpy<value_type>( base_type::data(), first.operator->(), std::min(v.size(),base_type::size()), cudaMemcpyHostToDevice ) );
	}

	HOST DEVICE bool operator==( const device_contiguous_memory_sequence& other ) const {
		#ifdef __CUDA_ARCH__
		if( base_type::size() != other.size() ) return false;
		const_iterator iter1 = begin();
		const_iterator iter2 = other.begin();
		for( ; iter1 != end(); ++iter1, ++iter2 ) if( !(*iter1 == *iter2) ) return false;
		return true;
		#else
		StagingVector v1, v2;
		copy_to_staging( v1 );
		other.copy_to_staging( v2 );
		return v1 == v2;
		#endif
	}

	HOST DEVICE inline bool operator!=( const device_contiguous_memory_sequence& other ) const { return !operator==(other); }

	HOST DEVICE inline bool operator<( const device_contiguous_memory_sequence& other ) const {
		#ifdef __CUDA_ARCH__
		return ecuda::lexicographical_compare( begin(), end(), other.begin(), other.end() );
		#else
		StagingVector v1, v2;
		copy_to_staging( v1 );
		other.copy_to_staging( v2 );
		return v1 < v2;
		#endif
	}

	HOST DEVICE inline bool operator>( const device_contiguous_memory_sequence& other ) const {
		#ifdef __CUDA_ARCH__
		return ecuda::lexicographical_compare( other.begin(), other.end(), begin(), end() );
		#else
		StagingVector v1, v2;
		copy_to_staging( v1 );
		other.copy_to_staging( v2 );
		return v1 > v2;
		#endif
	}

	HOST DEVICE inline bool operator<=( const device_contiguous_memory_sequence& other ) const { return !operator>(other); }

	HOST DEVICE inline bool operator>=( const device_contiguous_memory_sequence& other ) const { return !operator<(other); }

	template<class Alloc>
	HOST const device_contiguous_memory_sequence& operator>>( std::vector<value_type,Alloc>& vector ) const {
		vector.resize( base_type::size() );
		CUDA_CALL( cudaMemcpy<value_type>( &vector.front(), base_type::data(), base_type::size(), cudaMemcpyDeviceToHost ) );
		return *this;
	}

	template<class Alloc>
	HOST device_contiguous_memory_sequence& operator<<( std::vector<value_type,Alloc>& vector ) {
		CUDA_CALL( cudaMemcpy<value_type>( base_type::data(), &vector.front(), std::min(base_type::size(),vector.size()), cudaMemcpyHostToDevice ) );
		return *this;
	}

	DEVICE HOST device_contiguous_memory_sequence& operator=( device_contiguous_memory_sequence& src ) {
		base_type::operator=( src );
		return *this;
	}

};


///
/// \brief View of data sequence given a pointer and size.
///
/// Acts as a standalone representation of a linear fixed-size series of values
/// given a pointer and the desired size. Used to generate subsequences from
/// a larger memory structure (e.g. an individual row of a larger matrix). This
/// is a contrived structure to provide array-like operations, no
/// allocation/deallocation is done.
///
template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type>
class sequence_view
{
public:
	typedef T value_type; //!< element data type
	typedef PointerType pointer; //!< element pointer type
	typedef value_type& reference; //!< element reference type
	typedef const value_type& const_reference; //!< const element reference type
	typedef std::size_t size_type; //!< unsigned integral type
	typedef std::ptrdiff_t difference_type; //!< signed integral type

	typedef device_iterator<value_type,pointer> iterator; //!< iterator type
	typedef device_iterator<const value_type,/*const*/ pointer> const_iterator; //!< const iterator type
	typedef reverse_device_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

protected:
	pointer ptr; //!< pointer to the start of the array
	size_type length; //!< number of elements in the array

public:
	HOST DEVICE sequence_view() : ptr(nullptr), length(0) {}
	template<typename T2,typename PointerType2>
	HOST DEVICE sequence_view( const sequence_view<T2,PointerType2>& src ) : ptr(src.data()), length(src.size()) {}
	HOST DEVICE sequence_view( pointer ptr, size_type length ) : ptr(ptr), length(length) {}
	HOST DEVICE ~sequence_view() {}

	HOST DEVICE pointer data() const { return ptr; }

	// iterators:
	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(ptr); }
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator(ptr+static_cast<int>(length)); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(ptr); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(ptr+static_cast<int>(length)); }
	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(iterator(ptr+static_cast<int>(length))); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(iterator(ptr)); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(ptr+static_cast<int>(length))); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(ptr)); }

	#ifdef __CPP11_SUPPORTED__
	HOST DEVICE inline const_iterator cbegin() const __NOEXCEPT__ { return const_iterator(ptr); }
	HOST DEVICE inline const_iterator cend() const __NOEXCEPT__ { return const_iterator(ptr+static_cast<int>(length)); }
	HOST DEVICE inline const_reverse_iterator crbegin() __NOEXCEPT__ { return const_reverse_iterator(const_iterator(ptr+static_cast<int>(length))); }
	HOST DEVICE inline const_reverse_iterator crend() __NOEXCEPT__ { return const_reverse_iterator(const_iterator(ptr)); }
	#endif

	// capacity:
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return length; }
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return length == 0; }

	// element access:
	DEVICE inline reference operator[]( size_type index ) { return *(ptr+static_cast<int>(index)); }
	//DEVICE inline reference at( size_type index ) { return operator[]( index ); }
	DEVICE inline reference front() { return operator[](0); }
	DEVICE inline reference back() { return operator[](size()-1); }
	DEVICE inline const_reference operator[]( size_type index ) const {	return *(ptr+static_cast<int>(index)); }
	//DEVICE inline const_reference at( size_type index ) const { return operator[]( index ); }
	DEVICE inline const_reference front() const { return operator[](0); }
	DEVICE inline const_reference back() const { return operator[](size()-1); }

	template<class InputIterator>
	DEVICE void assign( InputIterator begin, InputIterator end ) {
		iterator iter = begin();
		while( begin != end and iter != end() ) {
			*iter = *begin;
			++iter;
			++begin;
		}
	}

	DEVICE void fill( const value_type& value ) {
		iterator iter = begin();
		while( iter != end() ) { *iter = value; ++iter; }
	}

	HOST DEVICE sequence_view& operator=( const sequence_view& other ) {
		ptr = other.ptr;
		length = other.length;
		return *this;
	}

};


///
/// \brief View of data sequence residing in contiguous memory given a pointer and size.
///
/// This is a subclass of sequence_view that imposes the requirement that the
/// underlying pointer refers to a contiguous block of memory.  Thus, PointerType
/// (the second template parameter of sequence_view) is strictly defined as a
/// naked pointer of type T*.
///
/// This allows the assign method to be made available safely.
///
template<typename T>
class contiguous_sequence_view : public sequence_view<T,T*>
{
private:
	typedef sequence_view<T,T*> base_type;

public:
	typedef typename sequence_view<T,T*>::value_type value_type; //!< element data type
	typedef typename sequence_view<T,T*>::pointer pointer; //!< element pointer type
	typedef typename sequence_view<T,T*>::reference reference; //!< element reference type
	typedef typename sequence_view<T,T*>::const_reference const_reference; //!< const element reference type
	typedef typename sequence_view<T,T*>::size_type size_type; //!< unsigned integral type
	typedef typename sequence_view<T,T*>::difference_type difference_type; //!< signed integral type

	typedef contiguous_device_iterator<T> iterator; //!< iterator type
	typedef contiguous_device_iterator<const T> const_iterator; //!< const iterator type
	typedef reverse_device_iterator<iterator> reverse_iterator; //!< reverse iterator type
	typedef reverse_device_iterator<const_iterator> const_reverse_iterator; //!< const reverse iterator type

public:
	HOST DEVICE contiguous_sequence_view() : sequence_view<T,T*>() {}
	template<typename T2>
	HOST DEVICE contiguous_sequence_view( const contiguous_sequence_view<T2>& src ) : sequence_view<T,T*>( src ) {}
	HOST DEVICE contiguous_sequence_view( pointer ptr, size_type length ) : sequence_view<T,T*>( ptr, length ) {}
	HOST DEVICE ~contiguous_sequence_view() {}

	// iterators:
	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(base_type::data()); }
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator(base_type::data()+static_cast<int>(base_type::size())); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(base_type::data()); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(base_type::data()+static_cast<int>(base_type::size())); }
	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(iterator(base_type::data()+static_cast<int>(base_type::size()))); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(iterator(base_type::data())); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(base_type::data()+static_cast<int>(base_type::size()))); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(base_type::data())); }

	HOST DEVICE void assign( contiguous_device_iterator<const T> begin, contiguous_device_iterator<const T> end ) {
		#ifdef __CUDA_ARCH__
		iterator dest = this->begin();
		while( begin != end and dest != this->end() ) { *dest = *begin; ++dest; ++begin; }
		#else
		const difference_type n = end-begin;
		if( n < 0 ) throw std::length_error( "ecuda::contiguous_sequence_view::assign() given iterator-based range oriented in wrong direction (are begin and end mixed up?)" );
		CUDA_CALL( cudaMemcpy<value_type>( base_type::data(), begin.operator->(), std::min(static_cast<typename base_type::size_type>(n),base_type::size()), cudaMemcpyDeviceToDevice ) );
		#endif
	}

	template<typename U,typename Q>
	DEVICE void assign( device_iterator<U,Q> begin, device_iterator<U,Q> end ) {
		iterator dest = this->begin();
		while( begin != end and dest != this->end() ) { *dest = *begin; ++dest; ++begin; }
	}

	template<class InputIterator>
	HOST DEVICE void assign( InputIterator begin, InputIterator end ) {
		#ifdef __CUDA_ARCH__
		iterator dest = this->begin();
		while( begin != end and dest != this->end() ) { *dest = *begin; ++dest; ++begin; }
		#else
		std::vector< value_type, host_allocator<value_type> > v( begin, end );
		CUDA_CALL( cudaMemcpy<value_type>( reinterpret_cast<value_type*>(sequence_view<T,T*>::data()), &v.front(), std::min(v.size(),sequence_view<T,T*>::size()), cudaMemcpyHostToDevice ) );
		#endif
	}

	HOST DEVICE void fill( const value_type& value ) {
		#ifdef __CUDA_ARCH__
		for( iterator iter = begin(); iter != end(); ++iter ) *iter = value;
		#else
		CUDA_CALL( cudaMemset<value_type>( base_type::data(), value, base_type::size() ) );
		#endif
	}

	HOST DEVICE contiguous_sequence_view& operator=( const contiguous_sequence_view& other ) {
		base_type::operator=( other );
		return *this;
	}


};

///
/// \brief View of data matrix residing given a pointer and dimensions.
///
/// Acts as a standalone representation of a fixed-size matrix of values
/// given a pointer and the desired dimensions. Used to generate submatrices from
/// a larger memory structure (e.g. an individual slice of a larger cube). This
/// is a contrived structure to provide matrix-like operations, no
/// allocation/deallocation is done.
///
template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type,class RowType=sequence_view<T,PointerType> >
class matrix_view : public RowType //sequence_view<T,PointerType>
{
private:
	//typedef sequence_view<T,PointerType> base_type;
	typedef RowType base_type;

public:
	typedef typename base_type::value_type value_type; //!< element data type
	typedef typename base_type::pointer pointer; //!< element pointer type
	typedef typename base_type::reference reference; //!< element reference type
	typedef typename base_type::const_reference const_reference; //!< const element reference type
	typedef typename base_type::size_type size_type; //!< unsigned integral type
	typedef typename base_type::difference_type difference_type; //!< signed integral type

	typedef typename base_type::iterator iterator; //!< iterator type
	typedef typename base_type::const_iterator const_iterator; //!< const iterator type
	typedef typename base_type::reverse_iterator reverse_iterator; //!< reverse iterator type
	typedef typename base_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

	typedef sequence_view<value_type,pointer> row_type;
	typedef sequence_view<const value_type,/*const*/ pointer> const_row_type;
	typedef sequence_view< value_type, striding_ptr<value_type,pointer> > column_type;
	typedef sequence_view< const value_type, striding_ptr<const value_type,/*const*/ pointer> > const_column_type;

private:
	size_type height;

public:
	HOST DEVICE matrix_view() : base_type(), height(0) {} //sequence_view<T>(), height(0) {}
	template<typename U>
	HOST DEVICE matrix_view( const matrix_view<U>& src ) : base_type(src), height(src.height) {} // sequence_view<T,PointerType>(src), height(src.height) {}
	HOST DEVICE matrix_view( pointer ptr, size_type width, size_type height ) : base_type(ptr,width*height), height(height) {} //sequence_view<T,PointerType>(ptr,width*height), height(height) {}
	HOST DEVICE ~matrix_view() {}

	// capacity:
	HOST DEVICE inline size_type get_width() const { return base_type::size()/height; }
	HOST DEVICE inline size_type get_height() const { return height; }

	// iterators:
	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(base_type::data()); }
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator(base_type::data()+static_cast<int>(base_type::size())); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(base_type::data()); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(base_type::data()+static_cast<int>(base_type::size())); }
	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(iterator(base_type::data()+static_cast<int>(base_type::size()))); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(iterator(base_type::data())); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(base_type::data()+static_cast<int>(base_type::size()))); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(base_type::data())); }

	// element access:
	HOST DEVICE inline row_type operator[]( size_type index ) {
		pointer ptr = base_type::data();
		ptr += index*get_width();
		return row_type( ptr, get_width() );
	}
	HOST DEVICE inline const_row_type operator[]( size_type index ) const {
		pointer ptr = base_type::data();
		ptr += index*get_width();
		return const_row_type( ptr, get_width() );
	}

	HOST DEVICE inline row_type get_row( size_type rowIndex ) { return operator[]( rowIndex ); }
	HOST DEVICE inline const_row_type get_row( size_type rowIndex ) const { return operator[]( rowIndex ); }

	HOST DEVICE inline column_type get_column( size_type columnIndex ) {
		pointer ptr = base_type::data();
		ptr += columnIndex;
		return column_type( striding_ptr<value_type,pointer>( ptr, get_width() ), get_height() );
	}

	HOST DEVICE inline const_column_type get_column( size_type columnIndex ) const {
		pointer ptr = base_type::data();
		ptr += columnIndex;
		return const_column_type( striding_ptr<const value_type,const pointer>( ptr, get_width() ), get_height() );
	}

	template<class InputIterator>
	DEVICE void assign( InputIterator begin, InputIterator end ) {
		iterator iter = begin();
		while( begin != end and iter != end() ) {
			*iter = *begin;
			++iter;
			++begin;
		}
	}

	DEVICE void fill( const value_type& value ) {
		iterator iter = begin();
		while( iter != end() ) { *iter = value; ++iter; }
	}


	HOST DEVICE matrix_view& operator=( const matrix_view& other ) {
		base_type::operator=( other );
		height = other.height;
		return *this;
	}

};

///
/// \brief View of data matrix residing in contiguous memory given a pointer and dimensions.
///
template<typename T>
class contiguous_matrix_view : private matrix_view< T, padded_ptr<T,T*,1>, contiguous_sequence_view<T> >
{
private:
	typedef matrix_view< T, padded_ptr<T,T*,1>, contiguous_sequence_view<T> > base_type;

public:
	typedef typename base_type::value_type value_type; //!< element data type
	typedef T* pointer; //!< element pointer type
	//typedef typename base_type::pointer pointer; //!< element pointer type
	typedef typename base_type::reference reference; //!< element reference type
	typedef typename base_type::const_reference const_reference; //!< const element reference type
	typedef typename base_type::size_type size_type; //!< unsigned integral type
	typedef typename base_type::difference_type difference_type; //!< signed integral type

	typedef typename base_type::iterator iterator; //!< iterator type
	typedef typename base_type::const_iterator const_iterator; //!< const iterator type
	typedef typename base_type::reverse_iterator reverse_iterator; //!< reverse iterator type
	typedef typename base_type::const_reverse_iterator const_reverse_iterator; //!< const reverse iterator type

	typedef contiguous_sequence_view<value_type> row_type;
	typedef contiguous_sequence_view<const value_type> const_row_type;
	typedef typename base_type::column_type column_type;
	typedef typename base_type::const_column_type const_column_type;

	typedef contiguous_device_iterator<const T> ContiguousDeviceIterator;

public:
	HOST DEVICE contiguous_matrix_view() : base_type() {}
	template<typename U>
	HOST DEVICE contiguous_matrix_view( const contiguous_matrix_view<U>& src ) : base_type(src) {}
	HOST DEVICE contiguous_matrix_view( pointer ptr, size_type width, size_type height, size_type paddingBytes=0 ) :
		base_type( padded_ptr<T,T*,1>( ptr, width, paddingBytes ), width, height ) {}
	HOST DEVICE ~contiguous_matrix_view() {}

	HOST DEVICE inline size_type get_width() const { return base_type::get_width(); }
	HOST DEVICE inline size_type get_height() const { return base_type::get_height(); }
	HOST DEVICE inline size_type get_pitch() const {
		padded_ptr<T,T*,1> ptr = base_type::data();
		const typename base_type::size_type pitch = ptr.get_data_length()*sizeof(value_type) + ptr.get_padding_length()*ptr.get_pad_length_units();
		return pitch;
	}

	// iterators:
	HOST DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(base_type::data()); }
	HOST DEVICE inline iterator end() __NOEXCEPT__ { return iterator(base_type::data()+static_cast<int>(base_type::size())); }
	HOST DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(base_type::data()); }
	HOST DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(base_type::data()+static_cast<int>(base_type::size())); }
	HOST DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(iterator(base_type::data()+static_cast<int>(base_type::size()))); }
	HOST DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(iterator(base_type::data())); }
	HOST DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(base_type::data()+static_cast<int>(base_type::size()))); }
	HOST DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(base_type::data())); }

	HOST DEVICE inline row_type operator[]( size_type index ) {
		pointer p = base_type::data();
		p += index*base_type::get_height();
		typename row_type::pointer np = p;
		return row_type( np, base_type::get_height() );
	}

	HOST DEVICE inline const_row_type operator[]( size_type index ) const {
		pointer p = base_type::data();
		p += index*base_type::get_height();
		typename const_row_type::pointer np = p;
		return const_row_type( np, base_type::get_height() );
	}

	HOST DEVICE inline row_type get_row( size_type rowIndex ) { return operator[]( rowIndex ); }
	HOST DEVICE inline const_row_type get_row( size_type rowIndex ) const { return operator[]( rowIndex ); }

	HOST DEVICE inline column_type get_column( size_type columnIndex ) { return base_type::get_column(); }
	HOST DEVICE inline const_column_type get_column( size_type columnIndex ) const { return base_type::get_column(); }

	HOST void assign( ContiguousDeviceIterator begin, ContiguousDeviceIterator end ) {
		const std::ptrdiff_t n = end-begin;
		if( n != (get_width()*get_height()) ) throw std::length_error( "ecuda::contiguous_matrix_view::assign() given iterator-based range that does not have width x height elements" );
		if( n < 0 ) throw std::length_error( "ecuda::contiguous_matrix_view::assign() given iterator-based range oriented in wrong direction (are begin and end mixed up?)" );
		CUDA_CALL( cudaMemcpy2D<value_type>( base_type::data(), get_pitch(), begin.operator->(), get_width()*sizeof(value_type), get_width(), get_height(), cudaMemcpyHostToDevice ) );
	}

	template<typename U,typename Q>
	DEVICE void assign( device_iterator<U,Q> begin, device_iterator<U,Q> end ) {
		iterator dest = this->begin();
		while( begin != end and dest != this->end() ) { *dest = *begin; ++dest; ++begin; }
	}

	template<class InputIterator>
	HOST void assign( InputIterator begin, InputIterator end ) {
		std::vector< value_type, host_allocator<value_type> > v( begin, end );
		if( v.size() != (get_width()*get_height()) ) throw std::length_error( "ecuda::contiguous_matrix_view::assign() given iterator-based range that does not have width x height elements" );
		CUDA_CALL( cudaMemcpy2D<value_type>( base_type::data(), get_pitch(), &v.front(), get_width()*sizeof(value_type), get_width(), get_height(), cudaMemcpyHostToDevice ) );
	}

	HOST DEVICE void fill( const value_type& value ) {
		#ifdef __CUDA_ARCH__
		iterator iter = begin();
		while( iter != end() ) { *begin = value; ++iter; }
		#else
		CUDA_CALL( cudaMemset2D<value_type>( base_type::data(), get_pitch(), value, get_width(), get_height() ) );
		#endif
	}

	HOST DEVICE contiguous_matrix_view& operator=( const contiguous_matrix_view& other ) {
		base_type::operator=( other );
		return *this;
	}

};

} // namespace ecuda

#endif
