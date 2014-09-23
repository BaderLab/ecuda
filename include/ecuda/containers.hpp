//----------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// containers.hpp
// Custom STL-like container classes.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------
#pragma once
#ifndef ECUDA_CONTAINERS_HPP
#define ECUDA_CONTAINERS_HPP

#include <algorithm>
#include <set>
#include <vector>

#include "global.hpp"
#include "iterators.hpp"

namespace ecuda {

///
/// Helper container that encloses another container and skips elements at a defined offset and interval.
///
template<class ContainerType,class IndexType=typename ContainerType::size_type>
class OffsettingContainer
{
public:
	typedef typename ContainerType::value_type value_type;
	typedef typename ContainerType::reference reference;
	typedef typename ContainerType::const_reference const_reference;
	typedef typename ContainerType::pointer pointer;
	typedef typename ContainerType::const_pointer const_pointer;
	typedef RandomAccessIterator< OffsettingContainer<ContainerType,IndexType>, pointer > iterator;
	typedef RandomAccessIterator< const OffsettingContainer<ContainerType,IndexType>, const_pointer > const_iterator;
	typedef ReverseIterator< RandomAccessIterator< OffsettingContainer<ContainerType,IndexType>, pointer > > reverse_iterator;
	typedef ReverseIterator< RandomAccessIterator< const OffsettingContainer<ContainerType,IndexType>, const_pointer > > const_reverse_iterator;
	typedef typename ContainerType::difference_type difference_type;
	typedef typename ContainerType::size_type size_type;

private:
public:
	ContainerType container;
	const IndexType extent;
	const IndexType offset;
	const IndexType increment;

public:
	///
	/// \param container The parent container.
	/// \param extent The desired number of elements to enclose (wrt. enclosing container).
	/// \param offset The offset of the starting element (wrt. parent container).
	/// \param increment The number of elements (wrt. parent container) to increment for each increase in the index (wrt. enclosing container).
	///
	/// e.g. Let a std::vector v have elements [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... ].
	///      Then OffsettingContainer( v, 3, 4, 2 ) will reflect elements [ 4, 6, 8 ].
	///
	HOST DEVICE OffsettingContainer( ContainerType& container, const IndexType extent, const IndexType offset=0, const IndexType increment=1 ) : container(container), extent(extent), offset(offset), increment(increment) {}
	HOST DEVICE OffsettingContainer( const OffsettingContainer& src ) : container(src.container), extent(src.extent), offset(src.offset), increment(src.increment) {}
	HOST DEVICE virtual ~OffsettingContainer() {}

	// iterators:
	DEVICE inline iterator begin() __NOEXCEPT__ { return iterator(this,0); }
	DEVICE inline iterator end() __NOEXCEPT__ { return iterator(this,size()); }
	DEVICE inline const_iterator begin() const __NOEXCEPT__ { return const_iterator(this,0); }
	DEVICE inline const_iterator end() const __NOEXCEPT__ { return const_iterator(this,size()); }
	DEVICE inline reverse_iterator rbegin() __NOEXCEPT__ { return reverse_iterator(iterator(this,size())); }
	DEVICE inline reverse_iterator rend() __NOEXCEPT__ { return reverse_iterator(iterator(this,0)); }
	DEVICE inline const_reverse_iterator rbegin() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(this,size())); }
	DEVICE inline const_reverse_iterator rend() const __NOEXCEPT__ { return const_reverse_iterator(const_iterator(this,0)); }

	// capacity:
	HOST DEVICE inline size_type size() const __NOEXCEPT__ { return extent; }
	HOST DEVICE inline bool empty() const __NOEXCEPT__ { return !size() ? true : false; }

	// element access:
	DEVICE inline reference operator[]( size_type index ) { return container.at( offset+index*increment ); }
	DEVICE inline reference at( size_type index ) { return operator[]( index ); }
	DEVICE inline reference front() { return operator[](0); }
	DEVICE inline reference back() { return operator[](size()-1); }
	DEVICE inline const_reference operator[]( size_type index ) const { return container.at( offset+index*increment ); }
	DEVICE inline const_reference at( size_type index ) const { return operator[]( index ); }
	DEVICE inline const_reference front() const { return operator[](0); }
	DEVICE inline const_reference back() const { return operator[](size()-1); }

};

} // namespace estd

#endif
