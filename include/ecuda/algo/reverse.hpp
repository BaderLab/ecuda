//----------------------------------------------------------------------------
// algo/reverse.hpp
//
// Extension of std::reverse that recognizes device memory and can be called
// from host or device code.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------
#pragma once
#ifndef ECUDA_ALGO_REVERSE_HPP
#define ECUDA_ALGO_REVERSE_HPP

#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"
// #include "../utility.hpp"
#include "../algorithm.hpp"

namespace ecuda {

// forward declaration
template<class BidirectionalIterator>
__HOST__ __DEVICE__ inline void
reverse(BidirectionalIterator first, BidirectionalIterator last);

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

template<class ForwardIterator>
__HOST__ __DEVICE__ inline void
reverse(ForwardIterator first,
        ForwardIterator last,
        ecuda::false_type // host memory
)
{
#ifdef __CUDA_ARCH__
// never called from device code
#else
    std::reverse(first, last);
#endif
}

template<class ForwardIterator>
__HOST__ __DEVICE__ inline void
reverse(ForwardIterator first,
        ForwardIterator last,
        ecuda::true_type // device memory
)
{
#ifdef __CUDA_ARCH__
    while ((first != last) and (first != --last)) {
        ecuda::swap(*first, *last);
        ++first;
    }
#else
    {
        typedef typename ecuda::iterator_traits<ForwardIterator>::iterator_category iterator_category;
        typedef typename ecuda::iterator_traits<ForwardIterator>::is_contiguous iterator_contiguity;
        const bool isSomeKindOfContiguous =
          ecuda::is_same<iterator_contiguity, ecuda::true_type>::value ||
          ecuda::is_same<iterator_category, device_contiguous_block_iterator_tag>::value;
        ECUDA_STATIC_ASSERT(isSomeKindOfContiguous, CANNOT_REVERSE_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_MEMORY);
    }
    typedef typename ecuda::remove_const<typename ecuda::iterator_traits<ForwardIterator>::value_type>::type value_type;
    std::vector<value_type, host_allocator<value_type>> v(::ecuda::distance(first, last));
    ::ecuda::copy(first, last, v.begin());
    ::ecuda::reverse(v.begin(), v.end());
    ::ecuda::copy(v.begin(), v.end(), first);
#endif
}

} // namespace impl
/// \endcond

ECUDA_SUPPRESS_HD_WARNINGS
template<class BidirectionalIterator>
__HOST__ __DEVICE__ inline void
reverse(BidirectionalIterator first, BidirectionalIterator last)
{
    impl::reverse(first, last, typename ecuda::iterator_traits<BidirectionalIterator>::is_device_iterator());
}

} // namespace ecuda

#endif
