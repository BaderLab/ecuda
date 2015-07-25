#ifndef ECUDA_PTR_STRIDING_PTR_HPP
#define ECUDA_PTR_STRIDING_PTR_HPP

#include <iostream>

#include "../global.hpp"
#include "../type_traits.hpp"
#include "common.hpp"

namespace ecuda {

template<typename T,typename PointerType=typename type_traits<T>::pointer>
class striding_ptr
{

public:
	typedef T element_type;
	typedef PointerType pointer;
	typedef T& reference;
	typedef const T& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

private:
	pointer ptr;
	size_type stride;

	template<typename T2,typename PointerType2> friend class striding_ptr;

public:
	__HOST__ __DEVICE__ striding_ptr( pointer ptr = pointer(), size_type stride = 0 ) : ptr(ptr), stride(stride) {}
	__HOST__ __DEVICE__ striding_ptr( const striding_ptr& src ) : ptr(src.ptr), stride(src.stride) {}

	__HOST__ __DEVICE__ inline pointer get() const { return ptr; }
	__HOST__ __DEVICE__ inline size_type get_stride() const { return stride; }

	__DEVICE__ inline reference operator*() { return *ptr; }
	__DEVICE__ inline const_reference operator*() const { return *ptr; }
	__DEVICE__ inline pointer operator->() const { return ptr; }
	__DEVICE__ inline reference operator[]( std::size_t i ) { return striding_ptr(*this).operator+=(i).operator*(); }
	__DEVICE__ inline const_reference operator[]( std::size_t i ) const { return striding_ptr(*this).operator+=(i).operator*(); }

	__HOST__ __DEVICE__ inline striding_ptr& operator++() { ptr += stride; return *this; }
	__HOST__ __DEVICE__ inline striding_ptr& operator--() { ptr -= stride; return *this; }
	__HOST__ __DEVICE__ inline striding_ptr operator++( int ) { striding_ptr tmp(*this); ++(*this); return tmp; }
	__HOST__ __DEVICE__ inline striding_ptr operator--( int ) { striding_ptr tmp(*this); --(*this); return tmp; }

	__HOST__ __DEVICE__ inline striding_ptr& operator+=( int x ) { ptr += x*stride; return *this; }
	__HOST__ __DEVICE__ inline striding_ptr& operator-=( int x ) { ptr -= x*stride; return *this; }

	__HOST__ __DEVICE__ inline striding_ptr operator+( int x ) const { striding_ptr tmp(*this); tmp += x; return tmp; }
	__HOST__ __DEVICE__ inline striding_ptr operator-( int x ) const { striding_ptr tmp(*this); tmp -= x; return tmp; }

	template<typename T2,typename PointerType2> __HOST__ __DEVICE__ bool operator==( const striding_ptr<T2,PointerType2>& other ) const { return ptr == other.ptr; }
	template<typename T2,typename PointerType2> __HOST__ __DEVICE__ bool operator!=( const striding_ptr<T2,PointerType2>& other ) const { return ptr != other.ptr; }
	template<typename T2,typename PointerType2> __HOST__ __DEVICE__ bool operator< ( const striding_ptr<T2,PointerType2>& other ) const { return ptr <  other.ptr; }
	template<typename T2,typename PointerType2> __HOST__ __DEVICE__ bool operator> ( const striding_ptr<T2,PointerType2>& other ) const { return ptr >  other.ptr; }
	template<typename T2,typename PointerType2> __HOST__ __DEVICE__ bool operator<=( const striding_ptr<T2,PointerType2>& other ) const { return ptr <= other.ptr; }
	template<typename T2,typename PointerType2> __HOST__ __DEVICE__ bool operator>=( const striding_ptr<T2,PointerType2>& other ) const { return ptr >= other.ptr; }

	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const striding_ptr& ptr ) {
		out << "striding_ptr(ptr=" << ptr.ptr << ";stride=" << ptr.stride << ")";
		return out;
	}

};

} // namespace ecuda

#endif
