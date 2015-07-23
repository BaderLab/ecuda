#pragma once
#ifndef ECUDA_UTILITY_HPP
#define ECUDA_UTILITY_HPP

#include "global.hpp"
#include "type_traits.hpp"

namespace ecuda {

#ifdef __CPP11_SUPPORTED__
template<typename T>
__host__ __device__ __CONSTEXPR__ typename std::remove_reference<T>::type&& move( T&& t ) __NOEXCEPT__ { return static_cast<typename std::remove_reference<T>::type&&>(t); }
#endif

template<typename T1,typename T2>
struct pair {
	typedef T1 first_type;
	typedef T2 second_type;
	T1 first;
	T2 second;
	__host__ __device__ pair() {}
	template<typename U,typename V> __host__ __device__ pair( const pair<U,V>& pr ) : first(pr.first), second(pr.second) {}
	__host__ __device__ pair( const first_type& a, const second_type& b ) : first(a), second(b) {}
};

} // namespace ecuda

#endif
