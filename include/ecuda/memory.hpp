/*
Copyright (c) 2014-2016, Scott Zuyderduyn
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
// memory.hpp
//
// CUDA implementations from STL header <memory> and additional helper
// templates and functions for compile-time manipulation of the ecuda
// pointer specializations.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#ifndef ECUDA_DEVICE_PTR_HPP
#define ECUDA_DEVICE_PTR_HPP

#include "global.hpp"

/*
#ifdef ECUDA_CPP11_AVAILABLE
#include <memory>
#else

namespace std {

template<class Alloc>
struct allocator_traits {
	typedef Alloc allocator_type;
	typedef typename Alloc::value_type      value_type;
	typedef typename Alloc::pointer         pointer;
	typedef typename Alloc::const_pointer   const_pointer;
	typedef void*                           void_pointer;
	typedef const void*                     const_void_pointer;
	typedef typename Alloc::difference_type difference_type;
	typedef typename Alloc::size_type       size_type;
	//propagate_on_container_copy_assignment
	//TODO: finish this
	static __HOST__ __DEVICE__ allocator_type select_on_container_copy_construction( const allocator_type& alloc ) { return alloc; }
};

} // namespace std

#endif
*/

///
/// \cond DEVELOPER_DOCUMENTATION
///
/// \section ecuda_pointers Pointer specializations
///
/// All but the most low-level memory access calls inside the ecuda API
/// are done using four pointer specialization classes: shared_ptr,
/// padded_ptr, striding_ptr, and unique_ptr.
///
/// shared_ptr and unique_ptr are functionally identical to their C++11
/// STL counterparts (i.e. std::shared_ptr), but are written to utilize
/// CUDA-allocated device memory (most notably by providing deallocation
/// through cudaFree).
///
/// tbc...
///
/// \endcond
///

// #include "ptr/naked_ptr.hpp" // deprecated
#include "ptr/padded_ptr.hpp"
#include "ptr/shared_ptr.hpp"
#include "ptr/striding_ptr.hpp"
#include "ptr/unique_ptr.hpp"
#include "ptr/striding_padded_ptr.hpp"

namespace ecuda {

template<typename T> struct owner_less;

///
/// This function object provides owner-based (as opposed to value-based) mixed-type
/// ordering of ecuda::shared_ptr. The order is such that two smart pointers compare
/// equivalent only if they are both empty or if they both manage the same object,
/// even if the values of the raw pointers obtained by get() are different (e.g.
/// because they point at different subobjects within the same object).
///
/// This class template is the preferred comparison predicate when building
/// associative containers with ecuda::shared_ptr as keys, that is:
/// std::map< ecuda::shared_ptr<T>, U, ecuda::owner_less<ecuda::shared_ptr<T> > >.
///
///
template<typename T>
struct owner_less< shared_ptr<T> >
{
	typedef bool result_type;
	typedef shared_ptr<T> first_argument_type;
	typedef shared_ptr<T> second_argument_type;
	///
	/// \brief Compares lhs and rhs using owner-based semantics.
	///
	/// \param lhs,rhs shared-ownership pointers to compare
	/// \return true if lhs is less than rhs as determined by the owner-based ordering
	///
	__HOST__ __DEVICE__ inline bool operator()( const shared_ptr<T>& lhs, const shared_ptr<T>& rhs ) const { return lhs.owner_before(rhs); }
};

} // namespace ecuda


#endif
