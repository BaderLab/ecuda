//----------------------------------------------------------------------------
// algorithm.hpp
//
// CUDA implementations from STL header <algorithm>.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#ifndef ECUDA_ALGORITHM_HPP
#define ECUDA_ALGORITHM_HPP

#include <iterator>
#include <vector>

#include "global.hpp"

namespace ecuda {

template<typename T>
__HOST__ __DEVICE__ inline const T&
min(const T& a, const T& b)
{
    return b < a ? b : a;
}
template<typename T, class Compare>
__HOST__ __DEVICE__ inline const T&
min(const T& a, const T& b, Compare cmp)
{
    return cmp(b, a) ? b : a;
}

template<typename T>
__HOST__ __DEVICE__ inline const T&
max(const T& a, const T& b)
{
    return b > a ? b : a;
}
template<typename T, class Compare>
__HOST__ __DEVICE__ inline const T&
max(const T& a, const T& b, Compare cmp)
{
    return cmp(a, b) ? b : a;
}

template<typename T>
__HOST__ __DEVICE__ inline void
swap(T& a, T& b) ECUDA__NOEXCEPT
{
    T tmp = a;
    a = b;
    b = tmp;
} // equivalent to std::swap

} // namespace ecuda

#include "iterator.hpp"
#include "type_traits.hpp"

#include "algo/copy.hpp"                    // equivalent to std::copy
#include "algo/count.hpp"                   // equivalent to std::count
#include "algo/count_if.hpp"                // equivalent to std::count_if
#include "algo/equal.hpp"                   // equivalent to std::equal
#include "algo/fill.hpp"                    // equivalent to std::fill
#include "algo/find.hpp"                    // equivalent to std::find
#include "algo/find_if.hpp"                 // equivalent to std::find_if
#include "algo/find_if_not.hpp"             // equivalent to std::find_if_not (NOTE: C++11 only)
#include "algo/for_each.hpp"                // equivalent to std::for_each
#include "algo/lexicographical_compare.hpp" // equivalent to std::lexicographical_compare
#include "algo/max_element.hpp"             // equivalent to std::max_element
#include "algo/mismatch.hpp"                // equivalent to std::mismatch
#include "algo/reverse.hpp"                 // equivalent to std::reverse

namespace ecuda {

ECUDA_SUPPRESS_HD_WARNINGS
template<class InputIterator, class UnaryPredicate>
inline __HOST__ __DEVICE__ bool
any_of(InputIterator first, InputIterator last, UnaryPredicate p)
{
    return ecuda::find_if(first, last, p) != last;
}

ECUDA_SUPPRESS_HD_WARNINGS
template<class InputIterator, class UnaryPredicate>
inline __HOST__ __DEVICE__ bool
none_of(InputIterator first, InputIterator last, UnaryPredicate p)
{
    return ecuda::find_if(first, last, p) == last;
}

} // namespace ecuda

#endif
