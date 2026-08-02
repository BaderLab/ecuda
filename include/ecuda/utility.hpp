//----------------------------------------------------------------------------
// utility.hpp
//
// Metaprogramming templates to get type traits at compile-time.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_UTILITY_HPP
#define ECUDA_UTILITY_HPP

#include "global.hpp"
#include "type_traits.hpp"

namespace ecuda {

///
/// \brief Couples together a pair of values.
///
/// This class is equivalent to the std::pair class
///
template<typename T1, typename T2>
struct pair
{
    typedef T1 first_type;
    typedef T2 second_type;
    T1 first;
    T2 second;
    ECUDA_SUPPRESS_HD_WARNINGS
    __HOST__ __DEVICE__ pair() {}
    ECUDA_SUPPRESS_HD_WARNINGS
    template<typename U, typename V>
    __HOST__ __DEVICE__ pair(const pair<U, V>& pr)
      : first(pr.first)
      , second(pr.second)
    {
    }
    ECUDA_SUPPRESS_HD_WARNINGS
    __HOST__ __DEVICE__ pair(const first_type& a, const second_type& b)
      : first(a)
      , second(b)
    {
    }
};

} // namespace ecuda

#endif
