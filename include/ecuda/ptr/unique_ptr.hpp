#ifndef ECUDA_PTR_UNIQUE_PTR_HPP
#define ECUDA_PTR_UNIQUE_PTR_HPP

#include <iostream>

#include "../global.hpp"
#include "common.hpp"

namespace ecuda {

template< typename T, class Deleter=default_delete<T> >
class unique_ptr
{

public:
	typedef T element_type;
	typedef T* pointer;
	typedef Deleter deleter_type;

private:
	pointer current_ptr;
	deleter_type deleter;

public:
	__host__ __device__ explicit unique_ptr( T* ptr = pointer() ) : current_ptr(ptr) {}
	__host__ __device__ unique_ptr( T* ptr, Deleter deleter ) : current_ptr(ptr), deleter(deleter) {}

	__host__ __device__ ~unique_ptr() { get_deleter()( current_ptr ); }

	template<typename U>
	__host__ __device__ inline unique_ptr& operator=( U* ptr ) {
		reset(release());
		current_ptr = ptr;
		return *this;
	}

	__host__ __device__ inline pointer release() __NOEXCEPT__ {
		pointer old_ptr = current_ptr;
		current_ptr = NULL;
		return old_ptr;
	}

	__host__ __device__ inline void reset( pointer ptr = pointer() ) __NOEXCEPT__ {
		pointer old_ptr = current_ptr;
		current_ptr = ptr;
		if( old_ptr ) get_deleter()( old_ptr );
	}

	__host__ __device__ inline void swap( unique_ptr& other ) __NOEXCEPT__ { ::ecuda::swap( current_ptr, other.current_ptr ); }

	__host__ __device__ inline pointer get() const { return current_ptr; }

	__host__ __device__ inline deleter_type& get_deleter() { return deleter; }
	__host__ __device__ inline const deleter_type& get_deleter() const { return deleter; }

	#ifdef __CPP11_SUPPORTED__
	__host__ __device__ explicit operator bool() const { return get() != NULL; }
	#else
	__host__ __device__ operator bool() const { return get() != NULL; }
	#endif

	__device__ inline typename add_lvalue_reference<T>::type operator*() const __NOEXCEPT__ { return *current_ptr; }

	__host__ __device__ inline pointer operator->() const __NOEXCEPT__ { return current_ptr; }

	template<typename T2,class D2> __host__ __device__ bool operator==( const unique_ptr<T2,D2>& other ) const { return get() == other.get(); }
	template<typename T2,class D2> __host__ __device__ bool operator!=( const unique_ptr<T2,D2>& other ) const { return get() != other.get(); }
	template<typename T2,class D2> __host__ __device__ bool operator< ( const unique_ptr<T2,D2>& other ) const { return get() <  other.get(); }
	template<typename T2,class D2> __host__ __device__ bool operator> ( const unique_ptr<T2,D2>& other ) const { return get() >  other.get(); }
	template<typename T2,class D2> __host__ __device__ bool operator<=( const unique_ptr<T2,D2>& other ) const { return get() <= other.get(); }
	template<typename T2,class D2> __host__ __device__ bool operator>=( const unique_ptr<T2,D2>& other ) const { return get() >= other.get(); }

};

} // namespace ecuda

#endif
