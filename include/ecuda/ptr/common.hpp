#ifndef ECUDA_PTR_COMMON_HPP
#define ECUDA_PTR_COMMON_HPP

namespace ecuda {

template<typename T>
struct default_delete {
	__host__ __device__ inline void operator()( T* ptr ) const {
		#ifdef __CUDA_ARCH__
		//ptr = NULL;
		#else
		if( ptr ) cudaFree(ptr);
		#endif
	}
};

template<typename T>
struct add_lvalue_reference {
	typedef T& type;
};

template<typename T>
struct add_lvalue_reference<T&> {
	typedef T& type;
};

} // namespace ecuda


#endif
