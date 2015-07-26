#ifndef ECUDA_ALGORITHM_HPP
#define ECUDA_ALGORITHM_HPP

#include <iterator>
#include <vector>

#include "global.hpp"

#include "algo/copy.hpp" // equivalent to std::copy
#include "algo/equal.hpp" // equivalent to std::equal
#include "algo/fill.hpp" // equivalent to std::fill
#include "algo/lexicographical_compare.hpp" // equivalent to std::lexicographical_compare
// NOTE: there are additional includes at the end of the file;
//       some algos require functions decleared in this file

#include "iterator.hpp"
#include <algorithm>
#include <utility>

namespace ecuda {

template<class InputIterator,typename T>
inline __HOST__ __DEVICE__ InputIterator find( InputIterator first, InputIterator last, const T& value ) {
	while( first != last ) {
		if( *first == value ) return first;
		++first;
	}
	return first;
}

template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ InputIterator find_if( InputIterator first, InputIterator last, UnaryPredicate p ) {
	while( first != last ) {
		if( p(*first) ) return first;
		++first;
	}
	return first;
}

template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ InputIterator find_if_not( InputIterator first, InputIterator last, UnaryPredicate p ) {
	while( first != last ) {
		if( !p(*first) ) return first;
		++first;
	}
	return first;
}


template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ bool all_of( InputIterator first, InputIterator last, UnaryPredicate p ) {
	return ecuda::find_if_not( first, last, p ) == last;
}

template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ bool any_of( InputIterator first, InputIterator last, UnaryPredicate p ) {
	return ecuda::find_if( first, last, p ) != last;
}

template<class InputIterator,class UnaryPredicate>
inline __HOST__ __DEVICE__ bool none_of( InputIterator first, InputIterator last, UnaryPredicate p ) {
	return ecuda::find_if( first, last, p ) == last;
}



template<typename T> __HOST__ __DEVICE__ inline void swap( T& a, T& b ) __NOEXCEPT__ { T& tmp = a; a = b; b = tmp; } // equivalent to std::swap

template<class InputIterator,class UnaryFunction> __HOST__ __DEVICE__ inline UnaryFunction __for_each( InputIterator first, InputIterator last, UnaryFunction f, detail::__false_type ) {
	#ifdef __CUDA_ARCH__
	return f; // never called from device code
	#else
	// just defer to STL
	return std::for_each( first, last, f );
	#endif
}

template<class InputIterator,class UnaryFunction> __HOST__ __DEVICE__ inline UnaryFunction __for_each( InputIterator first, InputIterator last, UnaryFunction f, detail::__true_type ) {
	#ifdef __CUDA_ARCH__
	while( first != last ) { f(*first); ++first; }
	return f; // never called from device code
	#else
	std::vector<typename ecuda::iterator_traits<InputIterator>::value_type> v( ecuda::distance(first,last) );
	ecuda::copy( first, last, v.begin() );
	__for_each( v.begin(), v.end(), f, detail::__false_type() );
	#endif
	return f;
}

template<class InputIterator,class UnaryFunction> __HOST__ __DEVICE__ inline UnaryFunction for_each( InputIterator first, InputIterator last, UnaryFunction f ) {
	return __for_each( first, last, f, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}



template<class InputIterator,typename T>
__HOST__ __DEVICE__ inline typename ecuda::iterator_traits<InputIterator>::difference_type __count( InputIterator first, InputIterator last, const T& value, detail::__false_type ) {
	#ifdef __CUDA_ARCH__
	return 0; // never called from device code
	#else
	// just defer to STL
	return std::count( first, last, value );
	#endif
}

template<class InputIterator,typename T>
__HOST__ __DEVICE__ inline typename ecuda::iterator_traits<InputIterator>::difference_type __count( InputIterator first, InputIterator last, const T& value, detail::__true_type ) {
	#ifdef __CUDA_ARCH__
	typename ecuda::iterator_traits<InputIterator>::difference_type n = 0;
	while( first != last ) {
		if( *first == value ) ++n;
		++first;
	}
	return n;
	#else
	std::vector<typename ecuda::iterator_traits<InputIterator>::value_type> v( ecuda::distance(first,last) );
	ecuda::copy( first, last, v.begin() );
	return __count( first, last, value, detail::__false_type() );
	#endif
}

template<class InputIterator,typename T>
__HOST__ __DEVICE__ inline typename ecuda::iterator_traits<InputIterator>::difference_type count( InputIterator first, InputIterator last, const T& value ) {
	return __count( first, last, value, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}



template<class InputIterator,class UnaryPredicate>
__HOST__ __DEVICE__ inline typename ecuda::iterator_traits<InputIterator>::difference_type __count_if( InputIterator first, InputIterator last, UnaryPredicate p, detail::__false_type ) {
	#ifdef __CUDA_ARCH__
	return 0; // never called from device code
	#else
	// just defer to STL
	return std::count_if( first, last, p );
	#endif
}

template<class InputIterator,class UnaryPredicate>
__HOST__ __DEVICE__ inline typename ecuda::iterator_traits<InputIterator>::difference_type __count_if( InputIterator first, InputIterator last, UnaryPredicate p, detail::__true_type ) {
	#ifdef __CUDA_ARCH__
	typename ecuda::iterator_traits<InputIterator>::difference_type n = 0;
	while( first != last ) {
		if( p(*first) ) ++n;
		++first;
	}
	return n;
	#else
	std::vector<typename ecuda::iterator_traits<InputIterator>::value_type> v( ecuda::distance(first,last) );
	ecuda::copy( first, last, v.begin() );
	return __count_if( v.begin(), v.end(), p, detail::__false_type() );
	#endif
}

template<class InputIterator,class UnaryPredicate>
__HOST__ __DEVICE__ inline typename ecuda::iterator_traits<InputIterator>::difference_type count_if( InputIterator first, InputIterator last, UnaryPredicate p ) {
	return __count_if( first, last, p, typename ecuda::iterator_traits<InputIterator>::is_device_iterator() );
}


} // namespace ecuda

#include "algo/reverse.hpp" // equivalent to std::reverse


#endif

