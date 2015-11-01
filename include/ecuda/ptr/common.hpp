#ifndef ECUDA_PTR_COMMON_HPP
#define ECUDA_PTR_COMMON_HPP

#include "../global.hpp"

namespace ecuda {

namespace detail {

// this is hacky structure that takes any pointer (const or not)
// and casts it to void* so it can be used by the deleter dispose() method
// and other places that call cudaFree
template<typename T> struct void_cast;
template<typename T> struct void_cast<const T*> { __HOST__ __DEVICE__ inline void* operator()( const T* ptr ) { return reinterpret_cast<void*>( const_cast<T*>(ptr) ); } };
template<typename T> struct void_cast           { __HOST__ __DEVICE__ inline void* operator()( T ptr        ) { return reinterpret_cast<void*>(ptr); } };

} // namespace detail

///
/// \brief The default destruction policy used by smart pointers.
///
/// The CUDA API function cudaFree() is used to deallocate memory.
///
template<typename T>
struct default_delete {

	///
	/// \brief Constructs an ecuda::default_delete object.
	///
	__HOST__ __DEVICE__ __CONSTEXPR__ default_delete() __NOEXCEPT__ {}

	///
	/// \brief Constructs an ecuda::default_delete object from another one.
	///
	/// This constructor will only participate in overload resolution
	/// if U* is implicitly convertible to T*.
	///
	template<typename U> __HOST__ __DEVICE__ default_delete( const default_delete<U>& src ) __NOEXCEPT__ {}

	///
	/// \brief Calls cudaFree() on a pointer.
	/// \param ptr an object or array to delete
	///
	__HOST__ __DEVICE__ inline void operator()( T* ptr ) const {
		#ifdef __CUDA_ARCH__
		//ptr = NULL;
		#else
		if( ptr ) cudaFree( detail::void_cast<T*>()(ptr) );
		//if( ptr ) cudaFree(ptr);
		#endif
	}

};

template<typename T>
struct default_host_delete {
	__HOST__ __DEVICE__ __CONSTEXPR__ default_host_delete() __NOEXCEPT__ {}
	template<typename U> __HOST__ __DEVICE__ default_host_delete( const default_host_delete<U>& src ) __NOEXCEPT__ {}
	__HOST__ __DEVICE__ inline void operator()( T* ptr ) const {
		#ifdef __CUDA_ARCH__
		#else
		if( ptr ) cudaFreeHost( detail::void_cast<T*>()(ptr) );
		#endif
	}
};

} // namespace ecuda


#endif
