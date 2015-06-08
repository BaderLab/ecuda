#pragma once
#ifndef ECUDA_UTILITY_HPP
#define ECUDA_UTILITY_HPP

#include "global.hpp"

namespace ecuda {

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
