#ifndef ECUDA_ALGORITHM_HPP
#define ECUDA_ALGORITHM_HPP

#include <iterator>
#include <vector>

#include "global.hpp"
#include "apiwrappers.hpp"
#include "iterator.hpp"

#include "algo/copy.hpp"
#include "algo/equal.hpp"
#include "algo/fill.hpp"
#include "algo/lexicographical_compare.hpp"

namespace ecuda {

 // forward declarations


template<class Iterator> __host__ __device__ inline typename std::iterator_traits<Iterator>::difference_type distance( Iterator first, Iterator last );

template<typename T> __host__ __device__ void swap( T& a, T& b ) __NOEXCEPT__ { T& tmp = a; a = b; b = tmp; }










} // namespace ecuda

#endif

