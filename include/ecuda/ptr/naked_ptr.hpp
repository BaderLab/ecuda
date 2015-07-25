#ifndef ECUDA_PTR_NAKED_PTR_HPP
#define ECUDA_PTR_NAKED_PTR_HPP

#include <iostream>

#include "../global.hpp"
#include "../type_traits.hpp"
#include "common.hpp"

namespace ecuda {

template<typename T>
class naked_ptr
{

public:
	typedef T element_type;
	typedef typename type_traits<T>::pointer pointer;
	//typedef typename __pointer_type<T>::pointer pointer;
	typedef T& reference;
	typedef const T& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

	template<typename U> friend class naked_ptr;

private:
	pointer ptr;

public:
	__HOST__ __DEVICE__ naked_ptr( pointer ptr = pointer() ) : ptr(ptr) {}
	__HOST__ __DEVICE__ naked_ptr( const naked_ptr& src ) : ptr(src.ptr) {}
	template<typename U>
	__HOST__ __DEVICE__ naked_ptr( const naked_ptr<U>& src ) : ptr(src.ptr) {}

	__HOST__ __DEVICE__ pointer get() const { return ptr; }

	__DEVICE__ inline reference operator*() { return *ptr; }
	__DEVICE__ inline const_reference operator*() const { return *ptr; }
	__DEVICE__ inline pointer operator->() const { return ptr; }

	__HOST__ __DEVICE__ inline naked_ptr& operator++() { ++ptr; return *this; }
	__HOST__ __DEVICE__ inline naked_ptr& operator--() { --ptr; return *this; }
	__HOST__ __DEVICE__ inline naked_ptr operator++( int ) { naked_ptr tmp(*this); ++(*this); return tmp; }
	__HOST__ __DEVICE__ inline naked_ptr operator--( int ) { naked_ptr tmp(*this); --(*this); return tmp; }

	__HOST__ __DEVICE__ inline naked_ptr& operator+=( int x ) { ptr += x; return *this; }
	__HOST__ __DEVICE__ inline naked_ptr& operator-=( int x ) { ptr -= x; return *this; }

	__HOST__ __DEVICE__ inline naked_ptr operator+( int x ) const { naked_ptr tmp(*this); tmp += x; return tmp; }
	__HOST__ __DEVICE__ inline naked_ptr operator-( int x ) const { naked_ptr tmp(*this); tmp -= x; return tmp; }

	__HOST__ __DEVICE__ inline difference_type operator-( const naked_ptr& other ) const { return ptr-other.ptr; }

	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const naked_ptr& ptr ) {
		out << "naked_ptr(ptr=" << ptr.ptr << ")";
		return out;
	}

};


} // namespace ecuda

#endif
