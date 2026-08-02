//----------------------------------------------------------------------------
// algo/count_if.hpp
//
// Extension of std::count_if that recognizes device memory and can be called from
// host or device code.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ALGO_COUNT_IF_HPP
#define ECUDA_ALGO_COUNT_IF_HPP

#include <algorithm>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class InputIterator, class UnaryPredicate>
inline __HOST__ __DEVICE__ typename ecuda::iterator_traits<InputIterator>::difference_type
count_if(InputIterator first, InputIterator last, UnaryPredicate p, ecuda::false_type) // host memory
{
#ifdef __CUDA_ARCH__
    return 0; // never called from device code
#else
    // just defer to STL
    return std::count_if(first, last, p);
#endif
}

template<class InputIterator, class UnaryPredicate>
__HOST__ __DEVICE__ typename ecuda::iterator_traits<InputIterator>::difference_type
count_if(InputIterator first, InputIterator last, UnaryPredicate p, ecuda::true_type) // device memory
{
#ifdef __CUDA_ARCH__
    typename ecuda::iterator_traits<InputIterator>::difference_type n = 0;
    while (first != last) {
        if (p(*first)) ++n;
        ++first;
    }
    return n;
#else
    typedef typename ecuda::remove_const<typename ecuda::iterator_traits<InputIterator>::value_type>::type value_type;
    std::vector<value_type, host_allocator<value_type>> v(ecuda::distance(first, last));
    ecuda::copy(first, last, v.begin());
    return std::count_if(v.begin(), v.end(), p);
#endif
}

} // namespace impl
/// \endcond

ECUDA_SUPPRESS_HD_WARNINGS
template<class InputIterator, class UnaryPredicate>
inline __HOST__ __DEVICE__ typename ecuda::iterator_traits<InputIterator>::difference_type
count_if(InputIterator first, InputIterator last, UnaryPredicate p)
{
    return impl::count_if(first, last, p, typename ecuda::iterator_traits<InputIterator>::is_device_iterator());
}

} // namespace ecuda

#endif
