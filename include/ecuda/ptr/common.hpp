#ifndef ECUDA_PTR_COMMON_HPP
#define ECUDA_PTR_COMMON_HPP

#include "../global.hpp"

namespace ecuda {

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
	__host__ __device__ __CONSTEXPR__ default_delete() __NOEXCEPT__ {}

	///
	/// \brief Constructs an ecuda::default_delete object from another one.
	///
	/// This constructor will only participate in overload resolution
	/// if U* is implicitly convertible to T*.
	///
	template<typename U> __host__ __device__ default_delete( const default_delete<U>& src ) __NOEXCEPT__ {}

	///
	/// \brief Calls cudaFree() on a pointer.
	/// \param ptr an object or array to delete
	///
	__host__ __device__ inline void operator()( T* ptr ) const {
		#ifdef __CUDA_ARCH__
		//ptr = NULL;
		#else
		if( ptr ) cudaFree(ptr);
		#endif
	}

};

} // namespace ecuda


#endif
