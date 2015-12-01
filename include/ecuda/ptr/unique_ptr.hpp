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
// ptr/unique_ptr.hpp
//
// A managed pointer to device memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#ifndef ECUDA_PTR_UNIQUE_PTR_HPP
#define ECUDA_PTR_UNIQUE_PTR_HPP

#include <iostream>

#include "../global.hpp"
#include "common.hpp"
#include "../type_traits.hpp"
//#include "../algorithm.hpp"

namespace ecuda {

///
/// \brief A smart pointer that retains sole ownership of an object.
///
/// ecuda::unique_ptr is a smart pointer that retains sole ownership of a
/// device object through a pointer and destroys that object when the
/// unique_ptr goes out of scope. No two unique_ptr instances can manage
/// the same object.
///
/// The object is destroyed and its memory deallocated when either of the
/// following happens:
///
/// - unique_ptr managing the object is destroyed
/// - unique_ptr managing the object is assigned another pointer via operator=() or reset()
///
/// The object is destroyed using a potentially user-supplied deleter by
/// calling Deleter(ptr). The deleter calls the destructor of the object
/// and dispenses the memory.
///
/// A unique_ptr may alternatively own no object, in which case it is called
/// empty.
///
/// There is no separate specialization of unique_ptr for dynamically-allocated
/// arrays of objects (i.e. T[]) since the underlying CUDA API makes no
/// distinction between individual objects versus arrays in terms of memory
/// allocation/deallocation.
///
/// The class satisfies the requirements of MoveConstructible and MoveAssignable,
/// but not the requirements of either CopyConstructible or CopyAssignable.
///
/// Deleter must be Functionobject or lvalue_reference to a FunctionObject or
/// lvalue reference to function, callable with an argument of type
/// unique_ptr<T,Deleter>::pointer.
///
template< typename T, class Deleter=default_device_delete<T> >
class unique_ptr
{

	/* SFINAE strat used in c++ stdlib to get T::pointer if it exists - unfortunately
	 * relies on decltype which we don't have prior to C++11 but we can do without
	 * this for now.
	class _Pointer {
		template<typename U> static typename U::pointer test(typename U::pointer*);
		template<typename U> static T* test(...);
		typedef typename std::remove_reference<Deleter>::type _Del;
	public:
		typedef decltype(test<Deleter>(0)) type;
	};
	*/

public:
	typedef T element_type;
	typedef typename ecuda::add_pointer<T>::type pointer;
	typedef Deleter deleter_type;

private:
	pointer current_ptr;
	deleter_type deleter;

private:
	__HOST__ __DEVICE__ unique_ptr( const unique_ptr& ); // disabled

public:
	__HOST__ __DEVICE__ __CONSTEXPR__ unique_ptr() __NOEXCEPT__ : current_ptr(NULL) {}
	__HOST__ __DEVICE__ explicit unique_ptr( T* ptr ) __NOEXCEPT__ : current_ptr(ptr) {}
	__HOST__ __DEVICE__ unique_ptr( T* ptr, Deleter deleter ) __NOEXCEPT__ : current_ptr(ptr), deleter(deleter) {}

	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ unique_ptr( unique_ptr&& src ) __NOEXCEPT__ : current_ptr(src.release()) {}
	template<typename U,class E> __HOST__ __DEVICE__ unique_ptr( unique_ptr<U,E>&& src ) __NOEXCEPT__ : current_ptr(src.release()), deleter_type(src.get_deleter()) {}
	#endif

	__HOST__ __DEVICE__ ~unique_ptr() { deleter(current_ptr); }

	#ifdef __CPP11_SUPPORTED__
	//TODO: review this block
	__HOST__ __DEVICE__ inline unique_ptr& operator=( unique_ptr&& src ) __NOEXCEPT__ {
		current_ptr = std::move(src.ptr);
		deleter = std::move(src.deleter);
		return *this;
	}
	template<typename U,class E> __HOST__ __DEVICE__ inline unique_ptr& operator=( unique_ptr<U,E>&& src ) __NOEXCEPT__ {
		current_ptr = std::move(src.ptr);
		deleter = std::move(src.deleter);
		return *this;
	}
	#endif

	//template<typename U>
	//__HOST__ __DEVICE__ inline unique_ptr& operator=( U* ptr ) {
	//	reset(release());
	//	current_ptr = ptr;
	//	return *this;
	//}

	__HOST__ __DEVICE__ inline pointer release() __NOEXCEPT__ {
		pointer old_ptr = current_ptr;
		current_ptr = NULL;
		return old_ptr;
	}

	__HOST__ __DEVICE__ inline void reset( pointer ptr = pointer() ) __NOEXCEPT__ {
		pointer old_ptr = current_ptr;
		current_ptr = ptr;
		if( old_ptr ) get_deleter()( old_ptr );
	}

	__HOST__ __DEVICE__ inline void swap( unique_ptr& other ) __NOEXCEPT__ { ::ecuda::swap( current_ptr, other.current_ptr ); }

	__HOST__ __DEVICE__ inline pointer get() const { return current_ptr; }

	__HOST__ __DEVICE__ inline deleter_type& get_deleter() { return deleter; }
	__HOST__ __DEVICE__ inline const deleter_type& get_deleter() const { return deleter; }

	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ explicit operator bool() const { return get() != NULL; }
	#else
	__HOST__ __DEVICE__ operator bool() const { return get() != NULL; }
	#endif

	__DEVICE__ inline typename ecuda::add_lvalue_reference<T>::type operator*() const __NOEXCEPT__ { return *current_ptr; }

	__HOST__ __DEVICE__ inline pointer operator->() const __NOEXCEPT__ { return current_ptr; }

	__DEVICE__ inline typename ecuda::add_lvalue_reference<T>::type operator[]( std::size_t i ) const {
		//return *pointer_traits<pointer>().increment( current_ptr, i );
		return *(current_ptr+i);
	}

	template<typename T2,class D2> __HOST__ __DEVICE__ bool operator==( const unique_ptr<T2,D2>& other ) const { return get() == other.get(); }
	template<typename T2,class D2> __HOST__ __DEVICE__ bool operator!=( const unique_ptr<T2,D2>& other ) const { return get() != other.get(); }
	template<typename T2,class D2> __HOST__ __DEVICE__ bool operator< ( const unique_ptr<T2,D2>& other ) const { return get() <  other.get(); }
	template<typename T2,class D2> __HOST__ __DEVICE__ bool operator> ( const unique_ptr<T2,D2>& other ) const { return get() >  other.get(); }
	template<typename T2,class D2> __HOST__ __DEVICE__ bool operator<=( const unique_ptr<T2,D2>& other ) const { return get() <= other.get(); }
	template<typename T2,class D2> __HOST__ __DEVICE__ bool operator>=( const unique_ptr<T2,D2>& other ) const { return get() >= other.get(); }

};

} // namespace ecuda

#endif
