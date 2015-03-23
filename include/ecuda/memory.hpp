/*
Copyright (c) 2014, Scott Zuyderduyn
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
// memory.hpp
//
// Smart pointers to device memory and simple homespun unique_ptr in absense
// of C++11 memory library.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_MEMORY_HPP
#define ECUDA_MEMORY_HPP

#include <cstddef>
#include "global.hpp"
#include "iterators.hpp"

#ifdef __CPP11_SUPPORTED__
#include <memory>
#endif

namespace ecuda {

template<typename T>
class deleter {
public:
	typedef T element_type;
	typedef T* pointer;
public:
	HOST DEVICE deleter() {}
	HOST DEVICE deleter( const deleter& other ) {}
	HOST DEVICE ~deleter() {}
	HOST inline void operator()( pointer ptr ) { if( ptr ) CUDA_CALL( cudaFree(ptr) ); }
};


///
/// A smart pointer for device memory.
///
/// This class keeps a pointer to allocated device memory and automatically
/// deallocates it when it goes out of scope.  The workings are similar to
/// a C++11 shared_ptr.  Since deallocation can only be done from host code
/// reference counting only occurs within host code.  On the device the pointer
/// is passed around freely without regards to reference counting and will
/// never undergo deallocation.
///
template<typename T>
class device_ptr {

public:
	typedef T element_type; //!< data type represented in allocated memory
	typedef T* pointer; //!< data type pointer
	typedef T& reference; //!< data type reference
	//typedef void** allocation_pointer; //!< pointer to pointer used by CUDA API to allocate device memory
	typedef std::size_t size_type; //!< size type for pointer arithmetic and reference counting

private:
	pointer ptr; //!< pointer to device memory
	size_type* shared_count; //!< pointer to reference count

public:
	HOST DEVICE device_ptr() : ptr(NULL) {
		#ifndef __CUDA_ARCH__
		shared_count = new size_type;
		*shared_count = 1;
		#endif
	}
	HOST DEVICE device_ptr( pointer ptr ) : ptr(ptr) {
		#ifndef __CUDA_ARCH__
		shared_count = new size_type;
		*shared_count = 1;
		#endif
	}

	HOST DEVICE device_ptr( const device_ptr<T>& src ) : ptr(src.ptr), shared_count(src.shared_count) {
		#ifndef __CUDA_ARCH__
		++(*shared_count);
		#endif
	}

	#ifdef __CPP11_SUPPORTED__
	HOST DEVICE device_ptr( device_ptr<T>&& src ) : ptr(src.ptr), shared_count(src.shared_count) {
		src.ptr = NULL;
		src.shared_count = NULL;
	}
	#endif

	// destroys the smart pointer, if instantiated from the host this will decrement the share count,
	// if instantiated from the device nothing happens, iff. the share count is zero and the smart
	// pointer resides on the host, the underlying device memory will be deallocated.
	HOST DEVICE ~device_ptr() {
		#ifndef __CUDA_ARCH__
		--(*shared_count);
		if( !(*shared_count) ) {
			deleter<T>()(ptr);
			delete shared_count;
		}
		#endif
	}

	// both host and device can get the pointer itself
	HOST DEVICE inline pointer get() const { return ptr; }
	HOST DEVICE inline operator bool() const { return get() != NULL; }

	// only device can dereference the pointer or call for the pointer in the context of acting upon the object
	DEVICE inline reference operator*() const { return *ptr; }
	DEVICE inline pointer   operator->() const { return ptr; }
	DEVICE inline reference operator[]( size_type index ) const { return *(ptr+index); }

	// get a pointer to the pointer to the device memory suitable for use with cudaMalloc...()-style calls
	//HOST inline allocation_pointer alloc_ptr() { return reinterpret_cast<void**>(&ptr); }

	// both host and device can do comparisons on the pointer
	HOST DEVICE inline bool operator==( const device_ptr<T>& other ) const { return ptr == other.ptr; }
	HOST DEVICE inline bool operator!=( const device_ptr<T>& other ) const { return ptr != other.ptr; }
	HOST DEVICE inline bool operator< ( const device_ptr<T>& other ) const { return ptr <  other.ptr; }
	HOST DEVICE inline bool operator> ( const device_ptr<T>& other ) const { return ptr >  other.ptr; }
	HOST DEVICE inline bool operator<=( const device_ptr<T>& other ) const { return ptr <= other.ptr; }
	HOST DEVICE inline bool operator>=( const device_ptr<T>& other ) const { return ptr >= other.ptr; }

	HOST DEVICE device_ptr<T>& operator=( const device_ptr<T>& other ) {
		#ifndef __CUDA_ARCH__
		~device_ptr();
		#endif
		ptr = other.ptr;
		#ifndef __CUDA_ARCH__
		shared_count = other.shared_count;
		++(*shared_count);
		#endif
		return *this;
	}

};

/*
template<typename T>
class device_pitched_ptr : public device_ptr {

public:
	typedef typename device_ptr<T>::element_type element_type; //!< data type represented in allocated memory
	typedef typename device_ptr<T>::pointer pointer; //!< data type pointer
	typedef typename device_ptr<T>::pointer reference; //!< data type reference
	typedef typename device_ptr<T>::pointer size_type; //!< size type for pointer arithmetic and reference counting

private:
	size_type pitch; //!< padded memory width in bytes

};
*/

template<typename T> struct cast_to_char;
template<typename T> struct cast_to_char<T*> { typedef char* type; };
template<typename T> struct cast_to_char<const T*> { typedef const char* type; };

///
/// A pointer class that implements all pointer-compatible operators for strided memory.
///
/// Strided memory is a block of contiguous memory where elements are separated by
/// fixed-length padding.  Thus, one has to "stride" over the padding to reach the
/// next element.  The term was borrowed from the GNU Scientific Library.
///
/// An optional second template parameter StrideBytes can be specified when the stride
/// is not a multiple of the byte size of the first template parameter T. By default
/// StrideBytes=sizeof(T).
///
template<typename T,std::size_t StrideBytes=sizeof(T)>
class strided_ptr {
public:
	typedef T element_type;
	typedef T* pointer;
	typedef T& reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

private:
	typename cast_to_char<pointer>::type ptr;
	//char* ptr;
	//pointer ptr;
	size_type stride;

public:
	HOST DEVICE strided_ptr( pointer p = pointer(), const size_type stride = 1 ) : ptr(reinterpret_cast<typename cast_to_char<pointer>::type>(p)), stride(stride*StrideBytes) {}
	HOST DEVICE strided_ptr( const strided_ptr<T,StrideBytes>& src ) : ptr(src.ptr), stride(src.stride) {}
	//template<typename U,std::size_t StrideBytes2>
	//strided_ptr( const strided_ptr<U,StrideBytes2>& src ) : ptr(src.ptr), stride(src.stride) {}
	HOST DEVICE ~strided_ptr() {}

	HOST DEVICE inline size_type get_stride() const { return stride/StrideBytes; }
	HOST DEVICE inline __CONSTEXPR__ size_type get_stride_bytes() const { return StrideBytes; }

	HOST DEVICE inline pointer get() const { return reinterpret_cast<pointer>(ptr); }
	HOST DEVICE inline operator bool() const { return ptr != nullptr; }

	HOST DEVICE inline strided_ptr& operator++() { ptr += stride; return *this; }
	HOST DEVICE inline strided_ptr operator++( int ) {
		strided_ptr tmp(*this);
		ptr += stride;
		return tmp;
	}

	HOST DEVICE inline strided_ptr& operator--() { ptr -= stride; return *this; }
	HOST DEVICE inline strided_ptr operator--( int ) {
		strided_ptr tmp(*this);
		ptr -= stride;
		return tmp;
	}

	HOST DEVICE inline strided_ptr& operator+=( const int strides ) { ptr += strides*stride; return *this; }
	HOST DEVICE inline strided_ptr& operator-=( const int strides ) { ptr -= strides*stride; return *this; }

	HOST DEVICE inline strided_ptr operator+( const int strides ) const { return strided_ptr<T,StrideBytes>( reinterpret_cast<pointer>(ptr+(strides*stride)), stride/StrideBytes ); }
	HOST DEVICE inline strided_ptr operator-( const int strides ) const { return strided_ptr<T,StrideBytes>( reinterpret_cast<pointer>(ptr-(strides*stride)), stride/StrideBytes ); }

	HOST DEVICE inline difference_type operator-( const strided_ptr& other ) const { return ptr-other.ptr; } // strided_ptr<T>( ptr-other.ptr, stride ); }

	DEVICE inline reference operator*() const { return *get(); }
	DEVICE inline pointer operator->() const { return get(); }

	template<std::size_t StrideBytes2> HOST DEVICE inline bool operator==( const strided_ptr<T,StrideBytes2>& other ) const { return ptr == other.ptr; }
	template<std::size_t StrideBytes2> HOST DEVICE inline bool operator!=( const strided_ptr<T,StrideBytes2>& other ) const { return ptr != other.ptr; }
	template<std::size_t StrideBytes2> HOST DEVICE inline bool operator< ( const strided_ptr<T,StrideBytes2>& other ) const { return ptr < other.ptr; }
	template<std::size_t StrideBytes2> HOST DEVICE inline bool operator> ( const strided_ptr<T,StrideBytes2>& other ) const { return ptr > other.ptr; }
	template<std::size_t StrideBytes2> HOST DEVICE inline bool operator<=( const strided_ptr<T,StrideBytes2>& other ) const { return ptr <= other.ptr; }
	template<std::size_t StrideBytes2> HOST DEVICE inline bool operator>=( const strided_ptr<T,StrideBytes2>& other ) const { return ptr >= other.ptr; }

};

///
/// A proxy to a pre-allocated block of contiguous memory.
///
/// The class merely holds the pointer and the length of the block
/// and provides the means to manipulate the sequence.  No
/// allocation/deallocation is done.
///
template<typename T,typename PointerType=typename ecuda::reference<T>::pointer_type>
class contiguous_memory_proxy
{
public:
	typedef T value_type; //!< The first template parameter (T)
	typedef PointerType pointer;
	//typedef value_type* pointer; //!< value_type*
	typedef value_type& reference; //!< value_type&
	typedef const T* const_pointer;
	// nvcc V6.0.1 produced a warning "type qualifiers are meaningless here", but above replacement line is fine
	//typedef const pointer const_pointer; //!< const value_type*
	typedef const T& const_reference;
	// nvcc V6.0.1 produced a warning "type qualifiers are meaningless here", but above replacement line is fine
	//typedef const reference const_reference; //!< const value_type&
	typedef std::ptrdiff_t difference_type;
	typedef std::size_t size_type;

	typedef pointer_iterator<value_type,pointer> iterator;
	typedef pointer_iterator<const value_type,pointer> const_iterator;
	typedef pointer_reverse_iterator<iterator> reverse_iterator;
	typedef pointer_reverse_iterator<const_iterator> const_reverse_iterator;

protected:
	pointer ptr;
	size_type length;

public:
	HOST DEVICE contiguous_memory_proxy() : ptr(nullptr), length(0) {}
	template<typename T2,typename PointerType2>
	HOST DEVICE contiguous_memory_proxy( const contiguous_memory_proxy<T2,PointerType2>& src ) : ptr(src.data()), length(src.size()) {}
	HOST DEVICE contiguous_memory_proxy( pointer ptr, size_type length ) : ptr(ptr), length(length) {}
	/*
	template<class Container>
	HOST DEVICE contiguous_memory_proxy(
		Container& container,
		typename Container::size_type length, // = typename Container::size_type(),
		typename Container::size_type offset = typename Container::size_type()
	) : ptr(container.data()+offset), length(length) {}
	*/
	HOST DEVICE virtual ~contiguous_memory_proxy() {}

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
	DEVICE inline reference at( size_type index ) { return operator[]( index ); }
	DEVICE inline reference front() { return operator[](0); }
	DEVICE inline reference back() { return operator[](size()-1); }
	DEVICE inline const_reference operator[]( size_type index ) const {	return *(ptr+static_cast<int>(index)); }
	DEVICE inline const_reference at( size_type index ) const { return operator[]( index ); }
	DEVICE inline const_reference front() const { return operator[](0); }
	DEVICE inline const_reference back() const { return operator[](size()-1); }

	HOST DEVICE contiguous_memory_proxy& operator=( const contiguous_memory_proxy& other ) {
		ptr = other.ptr;
		length = other.length;
		return *this;
	}

};

///
///
///
template<typename T>
class contiguous_2d_memory_proxy : public contiguous_memory_proxy<T>
{
protected:
	typedef contiguous_memory_proxy<T> base_type;

public:
	typedef typename base_type::value_type value_type;
	typedef typename base_type::pointer pointer;
	typedef typename base_type::reference reference;
	typedef typename base_type::const_pointer const_pointer;
	typedef typename base_type::const_reference const_reference;
	typedef typename base_type::iterator iterator;
	typedef typename base_type::const_iterator const_iterator;
	typedef typename base_type::reverse_iterator reverse_iterator;
	typedef typename base_type::const_reverse_iterator const_reverse_iterator;
	typedef typename base_type::difference_type difference_type;
	typedef typename base_type::size_type size_type;

	typedef contiguous_memory_proxy<T> row_type;
	typedef contiguous_memory_proxy<const T> const_row_type;
	//typedef strided_memory_proxy<T> column_type;

protected:
	size_type numberBlocks;
	size_type pitch;

public:
	HOST DEVICE contiguous_2d_memory_proxy() : contiguous_memory_proxy<T>(), numberBlocks(0) {}
	template<typename U>
	HOST DEVICE contiguous_2d_memory_proxy( const contiguous_2d_memory_proxy<U>& src ) : contiguous_memory_proxy<T>(src), numberBlocks(src.numberBlocks), pitch(src.pitch) {}
	HOST DEVICE contiguous_2d_memory_proxy( pointer ptr, size_type length, size_type numberBlocks, size_type pitch = 0 ) : contiguous_memory_proxy<T>(ptr,length), numberBlocks(numberBlocks), pitch(pitch) {}
	/*
	template<class Container>
	HOST DEVICE contiguous_2d_memory_proxy(
		Container& container,
		typename Container::size_type numberBlocks,
		typename Container::size_type blockSize,
		//typename Container::size_type width,
		//typename Container::size_type height,
		typename Container::size_type offset = typename Container::size_type()
	) :	contiguous_memory_proxy<T>( container, numberBlocks*blockSize, offset ), numberBlocks(numberBlocks) {
		//std::cerr << "contiguous_2d_memory_proxy( width=" << width << ", height=" << height << ", offset=" << offset << " )" << std::endl;
	}
	*/
	HOST DEVICE virtual ~contiguous_2d_memory_proxy() {}

	// capacity:
	HOST DEVICE inline size_type get_number_blocks() const { return numberBlocks; }
	HOST DEVICE inline size_type get_block_size() const { return contiguous_memory_proxy<T>::size()/numberBlocks; }
	HOST DEVICE inline size_type get_pitch() const { return pitch; }

	// element access:
	HOST DEVICE inline row_type operator[]( size_type index ) {
		char* ptr = reinterpret_cast<char*>(base_type::data());
		ptr += get_pitch();
		return row_type( reinterpret_cast<pointer>(ptr), get_block_size() );
	}
	HOST DEVICE inline const_row_type operator[]( size_type index ) const {
		const char* ptr = reinterpret_cast<const char*>(base_type::data());
		ptr += get_pitch();
		return row_type( reinterpret_cast<const_pointer>(ptr), get_block_size() );
	}	

	HOST DEVICE contiguous_2d_memory_proxy& operator=( const contiguous_2d_memory_proxy& other ) {
		base_type::operator=( other );
		numberBlocks = other.numberBlocks;
		return *this;
	}

};

#ifdef __CPP11_SUPPORTED__
// some future proofing for the glorious day when
// nvcc will support C++11 and we can just use the
// prepackaged implementations
template<typename T> typedef std::unique_ptr<T> unique_ptr<T>;
template<typename T> typedef std::unique_ptr<T[]> unique_ptr<T[]>;
#else
template<typename T>
class unique_ptr {

public:
	typedef T element_type;
	typedef T* pointer;
	typedef T& reference;

private:
	T* ptr;

public:
	HOST DEVICE unique_ptr( T* ptr=NULL ) : ptr(ptr) {}
	HOST DEVICE ~unique_ptr() {
		#ifndef __CUDA_ARCH__
		if( ptr ) delete ptr;
		#endif
	}

	HOST DEVICE inline pointer get() const { return ptr; }
	HOST DEVICE inline operator bool() const { return get() != NULL; }
	HOST DEVICE inline reference operator*() const { return *ptr; }
	HOST DEVICE inline pointer operator->() const { return ptr; }

	HOST DEVICE inline bool operator==( const unique_ptr<T>& other ) const { return ptr == other.ptr; }
	HOST DEVICE inline bool operator!=( const unique_ptr<T>& other ) const { return ptr != other.ptr; }
	HOST DEVICE inline bool operator<( const unique_ptr<T>& other ) const { return ptr < other.ptr; }
	HOST DEVICE inline bool operator>( const unique_ptr<T>& other ) const { return ptr > other.ptr; }
	HOST DEVICE inline bool operator<=( const unique_ptr<T>& other ) const { return ptr <= other.ptr; }
	HOST DEVICE inline bool operator>=( const unique_ptr<T>& other ) const { return ptr >= other.ptr; }

	HOST DEVICE unique_ptr<T>& operator=( T* p ) {
		ptr = p;
		return *this;
	}

};

template<typename T>
class unique_ptr<T[]> {

public:
	typedef T element_type;
	typedef T* pointer;
	typedef T& reference;
	typedef std::size_t size_type;

private:
	T* ptr;

public:
	HOST DEVICE unique_ptr<T[]>( T* ptr=NULL ) : ptr(ptr) {}
	//unique_ptr<T[]>( const unique_ptr<T[]>& src ) : ptr(src.ptr) {}
	HOST DEVICE ~unique_ptr<T[]>() { if( ptr ) delete [] ptr; }

	HOST DEVICE inline pointer get() const { return ptr; }
	HOST DEVICE inline operator bool() const { return get() != NULL; }
	HOST DEVICE inline reference operator[]( const size_type index ) const { return *(ptr+index); }

	HOST DEVICE inline bool operator==( const unique_ptr<T[]>& other ) const { return ptr == other.ptr; }
	HOST DEVICE inline bool operator!=( const unique_ptr<T[]>& other ) const { return ptr != other.ptr; }
	HOST DEVICE inline bool operator<( const unique_ptr<T[]>& other ) const { return ptr < other.ptr; }
	HOST DEVICE inline bool operator>( const unique_ptr<T[]>& other ) const { return ptr > other.ptr; }
	HOST DEVICE inline bool operator<=( const unique_ptr<T[]>& other ) const { return ptr <= other.ptr; }
	HOST DEVICE inline bool operator>=( const unique_ptr<T[]>& other ) const { return ptr >= other.ptr; }

	HOST DEVICE unique_ptr<T>& operator=( T* p ) {
		ptr = p;
		return *this;
	}

};
#endif

} // namespace ecuda

#endif
