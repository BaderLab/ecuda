/*
Copyright (c) 2014-2015, Scott Zuyderduyn
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
*/

//----------------------------------------------------------------------------
// type_traits.hpp
//
// A set of classes to obtain type information at compile-time.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_TYPE_TRAITS_HPP
#define ECUDA_TYPE_TRAITS_HPP

#include "global.hpp"
#include "iterators.hpp"
#include "striding_ptr.hpp"
#include "padded_ptr.hpp"

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION

///
/// Pointer traits are a way of getting const versions of specialized pointers.
///
/// For example, the const version of padded_ptr<int,int*,1> is
/// padded_ptr<const int,const int*,1>, NOT const padded_ptr<int,int*,1>.
///
/// Naked pointers are also converted properly. int* becomes const int* NOT
/// int* const.
///

template<typename T> struct __pointer_traits {
	typedef T* pointer;
	typedef const T* const_pointer;
};

template<typename T> struct __pointer_traits<T*> {
	typedef T* pointer;
	typedef const T* const_pointer;
};

template<typename T,typename PointerType,std::size_t PaddedUnitSize>
struct __pointer_traits< padded_ptr<T,PointerType,PaddedUnitSize> > {
	typedef padded_ptr<T,PointerType,PaddedUnitSize> pointer;
	typedef padded_ptr<const T,typename __pointer_traits<PointerType>::const_pointer,PaddedUnitSize> const_pointer;
};

template<typename T,typename PointerType>
struct __pointer_traits< striding_ptr<T,PointerType> > {
	typedef striding_ptr<T,PointerType> pointer;
	typedef striding_ptr<const T,typename __pointer_traits<PointerType>::const_pointer> const_pointer;
};


///
/// These are tags used to mark whether the dimensions of a particular memory
/// model are continguous or non-contiguous.
///
/// For example, a __device_sequence tagged with __dimension_contiguous_tag will
/// allow other containers that might wish to copy a range of values to be aware
/// that a direct transfer is possible (rather than copying the elements to a
/// staging area before transfer).
///
/// Similarly, __device_grid has two such tags, one for each dimension (row
/// and column).  For example, if rows are non-contiguous, but columns are, then a
/// fast row-by-row transfer can and should be performed.
///
/// These tags are primarily useful for deciding how to handle the contents of
/// source containers prior to participating in a cudaMemcpy call.
///

struct __dimension_contiguous_tag {};
struct __dimension_noncontiguous_tag {};

///
/// These are tags used to mark whether a container is "base" or "derived". The
/// naming choice may be suboptimal.  In practice, it just marks whether a
/// container is carrying a smart pointer or not.  For example, a base
/// __device_sequence will always be carrying a device_ptr, which we don't
/// really care about as far as the workings of the container.  We actually
/// care about the pointer that the device_ptr is reference-counting.  If
/// the container is a subset of the data of a base container, then the container
/// is derived, and will be carrying the correct type of pointer to start
/// (never a device_ptr).
///
/// These are primarily useful for correctly determing the typedef of
/// Container::pointer.  If the container is given a device_ptr, then
/// Container::pointer will be device_ptr<T>::pointer.  If the container is
/// derived, then Container::pointer is the same type as the pointer
/// given.
///

struct __container_type_base_tag {};
struct __container_type_derived_tag {};

///
/// These are templates that use the dimension and container_type tags attached
/// to a __device_sequence container to determine the type of the pointer visible
/// outside of the container and the template definition for container iterators.
///

template<typename T,typename PointerType,typename DimensionType,typename ContainerType> struct __device_sequence_traits;

template<typename T,typename PointerType> struct __device_sequence_traits<T,PointerType,__dimension_contiguous_tag,__container_type_base_tag> {
	typedef typename PointerType::pointer pointer;
	typedef contiguous_device_iterator<T> iterator;
	typedef contiguous_device_iterator<const T> const_iterator;
};

template<typename T,typename PointerType> struct __device_sequence_traits<T,PointerType,__dimension_noncontiguous_tag,__container_type_base_tag> {
	typedef typename PointerType::pointer pointer;
	typedef device_iterator<      T,typename __pointer_traits<PointerType>::pointer      > iterator;
	typedef device_iterator<const T,typename __pointer_traits<PointerType>::const_pointer> const_iterator;
};

template<typename T,typename PointerType> struct __device_sequence_traits<T,PointerType,__dimension_contiguous_tag,__container_type_derived_tag> {
	typedef PointerType pointer;
	typedef contiguous_device_iterator<T> iterator;
	typedef contiguous_device_iterator<const T> const_iterator;
};

template<typename T,typename PointerType> struct __device_sequence_traits<T,PointerType,__dimension_noncontiguous_tag,__container_type_derived_tag> {
	typedef PointerType pointer;
	typedef device_iterator<      T,typename __pointer_traits<PointerType>::pointer      > iterator;
	typedef device_iterator<const T,typename __pointer_traits<PointerType>::const_pointer> const_iterator;
};

///
/// These are templates that use the dimensions and container_type tags attached
/// to a __device_grid container to determine the type of the pointer visible
/// outside of the container and the template definition for container iterators.
///

template<typename T,typename PointerType,typename RowDimensionType,typename ColumnDimensionType,typename ContainerType> struct __device_grid_traits;

template<typename T,typename PointerType,typename RowDimensionType,typename ColumnDimensionType> struct __device_grid_traits<T,PointerType,RowDimensionType,ColumnDimensionType,__container_type_base_tag> {
	typedef typename PointerType::pointer pointer;
	typedef device_iterator<      T,typename __pointer_traits<typename PointerType::pointer>::pointer      > iterator;
	typedef device_iterator<const T,typename __pointer_traits<typename PointerType::pointer>::const_pointer> const_iterator;
};

template<typename T,typename PointerType> struct __device_grid_traits<T,PointerType,__dimension_contiguous_tag,__dimension_contiguous_tag,__container_type_base_tag> {
	typedef typename PointerType::pointer pointer;
	typedef contiguous_device_iterator<T> iterator;
	typedef contiguous_device_iterator<const T> const_iterator;
};

template<typename T,typename PointerType,typename RowDimensionType,typename ColumnDimensionType> struct __device_grid_traits<T,PointerType,RowDimensionType,ColumnDimensionType,__container_type_derived_tag> {
	typedef PointerType pointer;
	typedef device_iterator<      T,typename __pointer_traits<PointerType>::pointer      > iterator;
	typedef device_iterator<const T,typename __pointer_traits<PointerType>::const_pointer> const_iterator;
};

template<typename T,typename PointerType> struct __device_grid_traits<T,PointerType,__dimension_contiguous_tag,__dimension_contiguous_tag,__container_type_derived_tag> {
	typedef PointerType pointer;
	typedef contiguous_device_iterator<T> iterator;
	typedef contiguous_device_iterator<const T> const_iterator;
};

/// \endcond

} // namespace ecuda

#endif
