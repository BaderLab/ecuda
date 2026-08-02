//----------------------------------------------------------------------------
// algo/fill.hpp
//
// Extension of std::fill that recognizes device memory and can be called from
// host or device code.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ALGO_FILL_HPP
#define ECUDA_ALGO_FILL_HPP

#include <iterator>
#include <vector>

#include "../global.hpp"
#include "../iterator.hpp"
// #include "../utility.hpp"

namespace ecuda {

// forward declaration
template<class ForwardIterator, typename T>
__HOST__ __DEVICE__ inline void
fill(ForwardIterator first, ForwardIterator last, const T& val);

/// \cond DEVELOPER_DOCUMENTATION
namespace impl {

namespace fill_device {

template<class ForwardIterator, typename T>
__HOST__ __DEVICE__ inline void
fill(ForwardIterator first, ForwardIterator last, const T& val)
{
#ifdef __CUDA_ARCH__
    while (first != last) {
        *first = val;
        ++first;
    }
#else
    typedef typename ecuda::iterator_traits<ForwardIterator>::is_contiguous iterator_contiguity;
    {
        const bool isContiguous = ecuda::is_same<iterator_contiguity, ecuda::true_type>::value;
        ECUDA_STATIC_ASSERT(isContiguous, CANNOT_FILL_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_ITERATOR);
    }
    typename ecuda::iterator_traits<ForwardIterator>::difference_type n = ecuda::distance(first, last);
    typedef typename ecuda::iterator_traits<ForwardIterator>::value_type value_type;
    CUDA_CALL(cudaMemset<value_type>(first.operator->(), val, static_cast<std::size_t>(n)));
#endif
}

template<typename T, typename P>
__HOST__ __DEVICE__ inline void
fill(device_contiguous_block_iterator<T, P> first, device_contiguous_block_iterator<T, P> last, const T& val)
{
#ifdef __CUDA_ARCH__
    while (first != last) {
        *first = val;
        ++first;
    }
#else
    typedef device_contiguous_block_iterator<T, P> input_iterator_type;
    typename ecuda::iterator_traits<input_iterator_type>::difference_type n = std::distance(first, last);
    if (first.get_offset()) {
        const int leading = first.get_width() - first.get_offset();
        ::ecuda::fill(first.contiguous_begin(), first.contiguous_begin() + leading, val);
        first += leading;
        n -= leading;
    }
    const int rows = n / first.get_width();
    typedef typename ecuda::iterator_traits<device_contiguous_block_iterator<T, P>>::value_type value_type;
    typedef typename ecuda::add_pointer<value_type>::type pointer_type;
    CUDA_CALL(cudaMemset2D<value_type>(
      naked_cast<pointer_type>(first.operator->()), first.operator->().get_pitch(), val, first.get_width(), rows));
    n -= rows * first.get_width();
    if (n) ::ecuda::fill(first.contiguous_begin(), first.contiguous_begin() + n, val);
#endif
}

} // namespace fill_device

ECUDA_SUPPRESS_HD_WARNINGS
template<class ForwardIterator, typename T>
__HOST__ __DEVICE__ inline void
fill(ForwardIterator first,
     ForwardIterator last,
     const T& val,
     ecuda::true_type // device memory
)
{
#ifdef __CUDA_ARCH__
    while (first != last) {
        *first = val;
        ++first;
    }
#else
    typedef typename ecuda::iterator_traits<ForwardIterator>::is_contiguous iterator_contiguity;
    typedef typename ecuda::iterator_traits<ForwardIterator>::iterator_category iterator_category;
    {
        const bool isSomeKindOfContiguous =
          ecuda::is_same<iterator_contiguity, ecuda::true_type>::value ||
          ecuda::is_same<iterator_category, device_contiguous_block_iterator_tag>::value;
        ECUDA_STATIC_ASSERT(isSomeKindOfContiguous, CANNOT_FILL_RANGE_REPRESENTED_BY_NONCONTIGUOUS_DEVICE_ITERATOR);
    }
    if (ecuda::is_same<typename ecuda::iterator_traits<ForwardIterator>::value_type, T>::value) {
        fill_device::fill(first, last, val);
    } else {
        typedef typename ecuda::iterator_traits<ForwardIterator>::value_type value_type;
        const value_type val2(val);
        fill_device::fill(first, last, val2);
    }
#endif
}

template<class ForwardIterator, typename T>
__HOST__ __DEVICE__ inline void
fill(ForwardIterator first,
     ForwardIterator last,
     const T& val,
     ecuda::false_type // host memory
)
{
#ifdef __CUDA_ARCH__
// never called from device code
#else
    std::fill(first, last, val);
#endif
}

} // namespace impl
/// \endcond

ECUDA_SUPPRESS_HD_WARNINGS
template<class ForwardIterator, typename T>
__HOST__ __DEVICE__ inline void
fill(ForwardIterator first, ForwardIterator last, const T& val)
{
    impl::fill(first, last, val, typename ecuda::iterator_traits<ForwardIterator>::is_device_iterator());
}

} // namespace ecuda

#endif
