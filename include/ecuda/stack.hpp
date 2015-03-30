/*
Copyright (c) 2014, Scott Zuyderduyn
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
// stack.hpp
// An STL-like structure that resides in video memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_STACK_HPP
#define ECUDA_STACK_HPP

#include <cstddef>
#include <limits>
#include <vector>
#include "iterators.hpp"
#include "global.hpp"
#include "vector.hpp"

/// \cond NOT_SOMETHING_TO_CARE_ABOUT_CURRENTLY

namespace ecuda {

///
/// A video memory-bound stack structure.
///
template< typename T, class Alloc=device_allocator<T>, typename Container=vector<T,Alloc> >
class stack {

public:
	typedef T value_type; //!< cell data type
	typedef Alloc allocator_type; //!< allocator type
	typedef std::size_t size_type; //!< index data type
	typedef Container container_type; //!< type of the underlying container
	typedef value_type& reference; //!< cell reference type
	typedef const value_type& const_reference; //!< cell const reference type

private:
	container_type container; //!< underlying container

public:
	HOST stack( const container_type& ctnr = container_type() ) : container(ctnr) {}
	HOST stack( const stack<T,Alloc,Container>& other ) : container(other.container) {}
	HOST DEVICE virtual ~stack() {}

	HOST DEVICE inline bool empty() const { return container.empty(); }
	HOST DEVICE inline size_type size() const { return container.size(); }

	DEVICE inline reference top() { return container.back(); }
	DEVICE inline const_reference top() const { return container.back(); }
	HOST inline void push( const value_type& val ) { container.push_back(val); }
	HOST DEVICE inline void pop() { container.pop_back(); }

};

} // namespace ecuda

\endcond

#endif
