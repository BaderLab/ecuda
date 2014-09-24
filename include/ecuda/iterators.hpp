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
// iterators.hpp
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ITERATORS_HPP
#define ECUDA_ITERATORS_HPP

#include <iterator>
#include "global.hpp"

//
// Iterators are fashioned after STL iterators.  In this case, however, the logic
// doesn't rely on pointer math to traverse contents.  Rather, an index is stored
// and used to call the underlying container's operator[].  This allows containers
// where consecutive elements are not necessarily side-by-side in memory.
//
// A PointerType is a required parameter to deal with const/non-const iterators.
// For example, the container may be const, and accessing elements within that
// container in some contexts may need the elements themselves to be const.
// However, this cannot be determined easily at compile-time (prior to C++11).
// Therefore, passing a const PointerType will achieve this effect.
//
// These are essentially the iterator definitions from estd:: with __device__
// added to the appropriate function definitions.
//

namespace ecuda {

///
/// Base iterator.
///
/// Holds a pointer to the underlying container and the index of the element.
/// Implements the prefix and postfix ++ operator to advance the iterator.
///
template<class ContainerType,typename PointerType,class Category>
class Iterator : public std::iterator<Category,typename ContainerType::value_type,typename ContainerType::difference_type,PointerType,typename dereference<PointerType>::type>
{
protected:
	typedef std::iterator<Category,typename ContainerType::value_type,typename ContainerType::difference_type,PointerType,typename dereference<PointerType>::type> base_iterator_type;
public:
	typedef ContainerType container_type;
	typedef typename base_iterator_type::iterator_category iterator_category; // reverse iterator needs to access this
	typedef typename base_iterator_type::pointer pointer; // reverse iterator needs to access this

protected:
	ContainerType* pContainer; //!< pointer to underlying container
	typename ContainerType::size_type index; //!< index of underlying container element

protected:
	HOST DEVICE virtual Iterator copy() const { return Iterator(*this); }

public:
	HOST DEVICE Iterator( ContainerType* pContainer, const typename ContainerType::size_type index = 0 ) : pContainer(pContainer), index(index) {}
	HOST DEVICE Iterator( const Iterator<ContainerType,PointerType,Category>& src ) : pContainer(src.pContainer), index(src.index) {}
	HOST DEVICE virtual ~Iterator() {}

	HOST DEVICE inline Iterator& operator++() {	if( index < pContainer->size() ) ++index; return *this; }
	HOST DEVICE inline Iterator operator++( int ) const { return ++(this->copy()); }

};

///
/// Input iterator.
///
/// Implements the == and != operators to compare different iterators.
/// Implements const-only * and -> operators to access the underlying element.
///
template<class ContainerType,typename PointerType,class Category=std::input_iterator_tag>
class InputIterator : public virtual Iterator<ContainerType,PointerType,Category>
{
protected:
	typedef Iterator<ContainerType,PointerType,Category> base_iterator_type;
public:
	typedef ContainerType container_type;
	typedef typename base_iterator_type::iterator_category iterator_category; // reverse iterator needs to access this
	typedef typename base_iterator_type::pointer pointer; // reverse iterator needs to access this
	typedef const typename base_iterator_type::base_iterator_type::pointer const_pointer;
	typedef typename dereference<const_pointer>::type const_reference;

protected:
	HOST DEVICE virtual base_iterator_type copy() const { return InputIterator(*this); }

public:
	HOST DEVICE InputIterator( ContainerType* pContainer, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,PointerType,Category>( pContainer, index ) {}
	HOST DEVICE InputIterator( const InputIterator<ContainerType,PointerType,Category>& src ) : Iterator<ContainerType,PointerType,Category>( src ) {}
	HOST DEVICE virtual ~InputIterator() {}

	HOST DEVICE virtual bool operator==( const InputIterator& other ) const { return base_iterator_type::index == other.index; }
	HOST DEVICE virtual bool operator!=( const InputIterator& other ) const { return !operator==(other); }
	DEVICE virtual const_reference operator*() const { return base_iterator_type::pContainer->operator[]( base_iterator_type::index ); }
	DEVICE virtual const_pointer operator->() const { return &(base_iterator_type::pContainer->operator[]( base_iterator_type::index )); }
};

///
/// Output iterator.
///
/// Implements non-explicit const * and -> operators to access the underlying element (however, const can be introduced by using a const PointerType).
///
template<class ContainerType,typename PointerType,class Category=std::output_iterator_tag>
class OutputIterator : public virtual Iterator<ContainerType,PointerType,Category>
{
protected:
	typedef Iterator<ContainerType,PointerType,Category> base_iterator_type;
public:
	typedef ContainerType container_type;
	typedef typename base_iterator_type::iterator_category iterator_category; // reverse iterator needs to access this
	typedef typename base_iterator_type::base_iterator_type::pointer pointer;
	typedef typename dereference<pointer>::type reference;

protected:
	HOST DEVICE virtual base_iterator_type copy() const { return OutputIterator(*this); }

public:
	HOST DEVICE OutputIterator( ContainerType* pContainer, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,PointerType,Category>( pContainer, index ) {}
	HOST DEVICE OutputIterator( const OutputIterator<ContainerType,PointerType,Category>& src ) : Iterator<ContainerType,PointerType,Category>( src ) {}
	HOST DEVICE virtual ~OutputIterator() {}

	DEVICE virtual reference operator*() {	return base_iterator_type::pContainer->operator[]( base_iterator_type::index ); }
	DEVICE virtual pointer operator->() { return &(base_iterator_type::pContainer->operator[]( base_iterator_type::index )); }
};

///
/// Forward iterator.
///
/// Combines both the InputIterator and OutputIterator into a single iterator type.
///
template<class ContainerType,typename PointerType,class Category=std::forward_iterator_tag>
class ForwardIterator : public InputIterator<ContainerType,PointerType,Category>, public OutputIterator<ContainerType,PointerType,Category>
{
protected:
	typedef InputIterator<ContainerType,PointerType,Category> super_input_iterator_type;
	typedef OutputIterator<ContainerType,PointerType,Category> super_output_iterator_type;
	typedef typename super_input_iterator_type::base_iterator_type base_iterator_type;
public:
	typedef ContainerType container_type;
	typedef typename base_iterator_type::iterator_category iterator_category; // reverse iterator needs to access this
	typedef typename super_output_iterator_type::pointer pointer;
	typedef typename super_output_iterator_type::reference reference;
	typedef typename super_input_iterator_type::const_pointer const_pointer;
	typedef typename super_input_iterator_type::const_reference const_reference;

protected:
	HOST DEVICE virtual base_iterator_type copy() const { return ForwardIterator(*this); }

public:
	HOST DEVICE ForwardIterator() : Iterator<ContainerType,PointerType,Category>(), InputIterator<ContainerType,PointerType,Category>( nullptr, 0 ), OutputIterator<ContainerType,PointerType,Category>( nullptr, 0 ) {}
	HOST DEVICE ForwardIterator( ContainerType* pContainer, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,PointerType,Category>( pContainer, index ), InputIterator<ContainerType,PointerType,Category>( pContainer, index ), OutputIterator<ContainerType,PointerType,Category>( pContainer, index ) {}
	HOST DEVICE ForwardIterator( const ForwardIterator<ContainerType,PointerType,Category>& src ) : Iterator<ContainerType,PointerType,Category>( src ), InputIterator<ContainerType,PointerType,Category>( src ), OutputIterator<ContainerType,PointerType,Category>( src ) {}
	HOST DEVICE virtual ~ForwardIterator() {}

	DEVICE virtual const_reference operator*() const { return super_input_iterator_type::operator*(); }
	DEVICE virtual const_pointer operator->() const { return super_input_iterator_type::operator->(); }
	DEVICE virtual reference operator*() { return super_output_iterator_type::operator*(); }
	DEVICE virtual pointer operator->() { return super_output_iterator_type::operator->(); }
};

///
/// Bidirectional iterator.
///
/// Implements the prefix and postfix -- operator to regress the iterator.
///
template<class ContainerType,typename PointerType,class Category=std::bidirectional_iterator_tag>
class BidirectionalIterator : public ForwardIterator<ContainerType,PointerType,Category>
{
protected:
	typedef ForwardIterator<ContainerType,PointerType,Category> super_iterator_type;
	typedef typename super_iterator_type::base_iterator_type base_iterator_type;
public:
	typedef ContainerType container_type;
	typedef typename base_iterator_type::iterator_category iterator_category; // reverse iterator needs to access this
	typedef typename super_iterator_type::pointer pointer;
	typedef typename super_iterator_type::reference reference;
	typedef typename super_iterator_type::const_pointer const_pointer;
	typedef typename super_iterator_type::const_reference const_reference;

protected:
	HOST DEVICE virtual base_iterator_type copy() const { return BidirectionalIterator(*this); }

public:
	HOST DEVICE BidirectionalIterator() : Iterator<ContainerType,PointerType,Category>(), ForwardIterator<ContainerType,PointerType,Category>() {}
	HOST DEVICE BidirectionalIterator( ContainerType* pContainer, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,PointerType,Category>( pContainer, index ), ForwardIterator<ContainerType,PointerType,Category>( pContainer, index ) {}
	HOST DEVICE BidirectionalIterator( const BidirectionalIterator<ContainerType,PointerType,Category>& src ) : Iterator<ContainerType,PointerType,Category>( src ), ForwardIterator<ContainerType,PointerType,Category>( src ) {}
	HOST DEVICE virtual ~BidirectionalIterator() {}

	HOST DEVICE inline BidirectionalIterator& operator--() { if( base_iterator_type::index ) --base_iterator_type::index; return *this; }
	HOST DEVICE inline BidirectionalIterator operator--( int ) const { return BidirectionalIterator<ContainerType,PointerType,Category>( base_iterator_type::pContainer, base_iterator_type::index-1 ); }
};

///
/// Random-access iterator.
///
/// Implements the operator-( RandomAccessIterator ) to determine the difference in position between two iterators.
/// Implements the operator+(int), operator+=(int), operator-(int), and operator-=(int) to advance or regress the element a given number of positions.
/// Implements the operator<( RandomAccessIterator ), operator<=( RandomAccessIterator ), operator>( RandomAccessIterator ),
///   operator>=( RandomAccessIterator ) to compare the position of two iterators.
/// Implements the operator[](int) to access an element some number ahead relative to the position of the iterator.
///

template<class ContainerType,typename PointerType,class Category=std::random_access_iterator_tag>
class RandomAccessIterator : public BidirectionalIterator<ContainerType,PointerType,Category>
{
protected:
	typedef BidirectionalIterator<ContainerType,PointerType,Category> super_iterator_type;
	typedef typename super_iterator_type::super_iterator_type::super_input_iterator_type super_input_iterator_type;
	typedef typename super_iterator_type::super_iterator_type::base_iterator_type base_iterator_type;
public:
	typedef ContainerType container_type;
	typedef typename base_iterator_type::iterator_category iterator_category; // reverse iterator needs to access this
	typedef typename super_iterator_type::pointer pointer;
	typedef typename super_iterator_type::reference reference;
	typedef typename super_iterator_type::const_pointer const_pointer;
	typedef typename super_iterator_type::const_reference const_reference;
	typedef typename base_iterator_type::difference_type difference_type;

protected:
	HOST DEVICE virtual base_iterator_type copy() const { return RandomAccessIterator(*this); }

public:
	HOST DEVICE RandomAccessIterator() : Iterator<ContainerType,PointerType,Category>(), BidirectionalIterator<ContainerType,PointerType,Category>() {}
	HOST DEVICE RandomAccessIterator( ContainerType* pContainer, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,PointerType,Category>( pContainer, index ), BidirectionalIterator<ContainerType,PointerType,Category>( pContainer, index ) {}
	HOST DEVICE RandomAccessIterator( const RandomAccessIterator<ContainerType,PointerType,Category>& src ) : Iterator<ContainerType,PointerType,Category>( src ), BidirectionalIterator<ContainerType,PointerType,Category>( src ) {}
	HOST DEVICE virtual ~RandomAccessIterator() {}

	HOST DEVICE virtual difference_type operator-( const RandomAccessIterator& other ) { return base_iterator_type::index - other.index; }

	HOST DEVICE inline RandomAccessIterator operator+( int x ) const { return RandomAccessIterator( base_iterator_type::pContainer, base_iterator_type::index+x ); }
	HOST DEVICE inline RandomAccessIterator operator-( int x ) const { return RandomAccessIterator( base_iterator_type::pContainer, base_iterator_type::index-x ); }

	HOST DEVICE virtual bool operator<( const RandomAccessIterator& other ) const { return base_iterator_type::index < other.index; }
	HOST DEVICE virtual bool operator>( const RandomAccessIterator& other ) const { return base_iterator_type::index > other.index; }
	HOST DEVICE virtual bool operator<=( const RandomAccessIterator& other ) const { return operator<(other) or super_input_iterator_type::operator==(other); }
	HOST DEVICE virtual bool operator>=( const RandomAccessIterator& other ) const { return operator>(other) or super_input_iterator_type::operator==(other); }

	HOST DEVICE inline RandomAccessIterator& operator+=( int x ) { base_iterator_type::index += x; return *this; }
	HOST DEVICE inline RandomAccessIterator& operator-=( int x ) { base_iterator_type::index -= x; return *this; }

	DEVICE virtual reference operator[]( int x ) { return base_iterator_type::pContainer->at( base_iterator_type::index+x ); }
	DEVICE virtual const_reference operator[]( int x ) const { return base_iterator_type::pContainer->at( base_iterator_type::index+x ); }

};

///
/// Reverse iterator.
///
/// Takes any class of the above container and traverses the elements in reverse order.
///

template<class ParentIterator>
class ReverseIterator : public ForwardIterator<typename ParentIterator::container_type,typename ParentIterator::pointer,typename ParentIterator::iterator_category>
{
protected:
	typedef ForwardIterator<typename ParentIterator::container_type,typename ParentIterator::pointer,typename ParentIterator::iterator_category> super_iterator_type;
	typedef typename super_iterator_type::base_iterator_type base_iterator_type;

public:
	typedef typename ParentIterator::container_type container_type;
	typedef typename ParentIterator::iterator_category iterator_category;
	typedef typename ParentIterator::pointer pointer;
	typedef typename dereference<pointer>::type reference;
	typedef const typename ParentIterator::pointer const_pointer;
	typedef typename dereference<const_pointer>::type const_reference;

private:
	ParentIterator parentIterator;

public:
	HOST DEVICE ReverseIterator() : Iterator<typename ParentIterator::container_type,typename ParentIterator::pointer_type,typename ParentIterator::iterator_category>(nullptr), ForwardIterator<typename ParentIterator::container_type,typename ParentIterator::pointer_type,typename ParentIterator::iterator_category>(nullptr){}
	HOST DEVICE ReverseIterator( ParentIterator parentIterator ) : Iterator<typename ParentIterator::container_type,typename ParentIterator::pointer,typename ParentIterator::iterator_category>(nullptr), ForwardIterator<typename ParentIterator::container_type,typename ParentIterator::pointer,typename ParentIterator::iterator_category>(nullptr), parentIterator(parentIterator) {}

	HOST DEVICE ParentIterator base() const { return parentIterator; }

	HOST DEVICE virtual bool operator==( const ReverseIterator<ParentIterator>& other ) const { return parentIterator.operator==( other.parentIterator ); }
	HOST DEVICE virtual bool operator!=( const ReverseIterator<ParentIterator>& other ) const { return parentIterator.operator!=( other.parentIterator ); }
	DEVICE virtual const_reference operator*() const { return parentIterator.operator--(0).operator*(); }
	DEVICE virtual const_pointer operator->() const { return parentIterator.operator--(0).operator->(); }

	HOST DEVICE virtual bool operator==( const ReverseIterator<ParentIterator>& other ) { return parentIterator.operator==( other.parentIterator ); }
	HOST DEVICE virtual bool operator!=( const ReverseIterator<ParentIterator>& other ) { return parentIterator.operator!=( other.parentIterator ); }
	DEVICE virtual reference operator*() { return parentIterator.operator--(0).operator*(); }
	DEVICE virtual pointer operator->() { return parentIterator.operator--(0).operator->(); }

	HOST DEVICE ReverseIterator<ParentIterator>& operator++() { --parentIterator; return *this; }
	HOST DEVICE ReverseIterator<ParentIterator> operator++( int x ) const { return ReverseIterator<ParentIterator>(*this).operator--(x); } // ReverseIterator<ParentIterator>(parentIterator.operator--(x)); }

	HOST DEVICE ReverseIterator<ParentIterator>& operator--() { ++parentIterator; return *this; }
	HOST DEVICE ReverseIterator<ParentIterator> operator--( int x ) const { return ReverseIterator<ParentIterator>(*this).operator++(x); } // ReverseIterator<ParentIterator>(parentIterator.operator++(x)); }

	HOST DEVICE ReverseIterator<ParentIterator> operator+( int x ) { return ReverseIterator<ParentIterator>(*this).operator-(x); }
	HOST DEVICE ReverseIterator<ParentIterator> operator-( int x ) { return ReverseIterator<ParentIterator>(*this).operator+(x); }
	HOST DEVICE bool operator<( const ReverseIterator<ParentIterator>& other ) const { return parentIterator.operator>=(other); }
	HOST DEVICE bool operator>( const ReverseIterator<ParentIterator>& other ) const { return parentIterator.operator<=(other); }
	HOST DEVICE bool operator<=( const ReverseIterator<ParentIterator>& other ) const { return operator<(other) or operator==(other); }
	HOST DEVICE bool operator>=( const ReverseIterator<ParentIterator>& other ) const { return operator>(other) or operator==(other); }
	HOST DEVICE ReverseIterator<ParentIterator>& operator+=( int x ) { parentIterator.operator-=(x); return *this; }
	HOST DEVICE ReverseIterator<ParentIterator>& operator-=( int x ) { parentIterator.operator+=(x); return *this; }

	DEVICE inline reference operator[]( int x ) { return parentIterator.operator[]( -x ); }
	DEVICE inline const_reference operator[]( int x ) const { return parentIterator.operator[]( -x ); }

};

} // namespace ecuda

#endif
