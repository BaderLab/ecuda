//----------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
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
#include "memory.hpp"
#include "vector.hpp"

namespace ecuda {

///
/// A video memory-bound stack structure.
///
template< typename T,typename Container=vector<T> >
class stack {

public:
	typedef T value_type; //!< cell data type
	typedef std::size_t size_type; //!< index data type
	typedef Container container_type; //!< type of the underlying container
	typedef value_type& reference; //!< cell reference type
	typedef const value_type& const_reference; //!< cell const reference type

private:
	container_type container; //!< underlying container

public:
	HOST DEVICE stack( const container_type& ctnr = container_type() ) : container(ctnr) {}
	HOST DEVICE virtual ~stack() {}

	HOST DEVICE inline bool empty() const { return container.empty(); }
	HOST DEVICE inline size_type size() const { return container.size(); }

	DEVICE inline reference top() { return container.back(); }
	DEVICE inline const_reference top() const { return container.back(); }
	HOST inline void push( const value_type& val ) { container.push_back(val); }
	HOST DEVICE inline void pop() { container.pop_back(); }

};

} // namespace ecuda

#endif
