//----------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// iterators.hpp
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------
#pragma once
#ifndef ECUDA_ITERATORS_HPP
#define ECUDA_ITERATORS_HPP

#include <iterator>
#include "global.hpp"

namespace ecuda {

template<class ContainerType,class Category>
class Iterator : public std::iterator<Category,typename ContainerType::value_type,typename ContainerType::difference_type,typename ContainerType::pointer,typename ContainerType::reference>
{
public:
	typedef ContainerType container_type;
	typedef Category iterator_category;
	typedef typename ContainerType::value_type value_type;
	typedef typename ContainerType::difference_type difference_type;
	typedef typename ContainerType::pointer pointer;
	typedef typename ContainerType::reference reference;

protected:
	ContainerType* pContainer;
	typename ContainerType::size_type index;

protected:
	__device__ virtual Iterator<ContainerType,Category> copy() const { return Iterator<ContainerType,Category>(*this); }

public:
	__device__ Iterator( ContainerType* pContainer, const typename ContainerType::size_type index = 0 ) : pContainer(pContainer), index(index) {}
	__device__ Iterator( const Iterator<ContainerType,Category>& src ) : pContainer(src.pContainer), index(src.index) {}
	__device__ virtual ~Iterator() {}

	//virtual difference_type operator-( const RandomAccessIterator<ContainerType,Category>& other ) { return index-other.index; }

	__device__ Iterator<ContainerType,Category>& operator++() {	if( index < pContainer->size() ) ++index; return *this; }
	__device__ Iterator<ContainerType,Category> operator++( int ) const { return ++(this->copy()); }

};

template<class ContainerType,class Category=std::input_iterator_tag>
class InputIterator : public virtual Iterator<ContainerType,Category>
{
public:
	typedef ContainerType container_type;
	typedef Category iterator_category;
	typedef typename ContainerType::value_type value_type;
	typedef typename ContainerType::difference_type difference_type;
	typedef typename ContainerType::const_pointer const_pointer;
	typedef typename ContainerType::const_reference const_reference;

protected:
	__device__ virtual Iterator<ContainerType,Category> copy() const { return InputIterator<ContainerType,Category>(*this); }

public:
	__device__ InputIterator( ContainerType* container, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,Category>( container, index ) {}
	__device__ InputIterator( const InputIterator<ContainerType,Category>& src ) : Iterator<ContainerType,Category>( src ) {}
	__device__ virtual ~InputIterator() {}

	__device__ virtual bool operator==( const InputIterator<ContainerType,Category>& other ) const { return /*Iterator<ContainerType,IteratorTag>::pContainer == other.pContainer and*/ Iterator<ContainerType,Category>::index == other.index; }
	__device__ virtual bool operator!=( const InputIterator<ContainerType,Category>& other ) const { return !operator==(other); }
	__device__ virtual const_reference operator*() const { return Iterator<ContainerType,Category>::pContainer->operator[]( Iterator<ContainerType,Category>::index ); }
	__device__ virtual const_pointer operator->() const { return &(Iterator<ContainerType,Category>::pContainer->operator[]( Iterator<ContainerType,Category>::index )); }
};

template<class ContainerType,class Category=std::output_iterator_tag>
class OutputIterator : public virtual Iterator<ContainerType,Category>
{
public:
	typedef ContainerType container_type;
	typedef Category iterator_category;
	typedef typename ContainerType::value_type value_type;
	typedef typename ContainerType::difference_type difference_type;
	typedef typename ContainerType::pointer pointer;
	typedef typename ContainerType::reference reference;

protected:
	__device__ virtual Iterator<ContainerType,Category> copy() const { return OutputIterator<ContainerType,Category>(*this); }

public:
	__device__ OutputIterator( ContainerType* container, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,Category>( container, index ) {}
	__device__ OutputIterator( const OutputIterator<ContainerType,Category>& src ) : Iterator<ContainerType,Category>( src ) {}
	__device__ virtual ~OutputIterator() {}

	__device__ virtual reference operator*() {	return Iterator<ContainerType,Category>::pContainer->operator[]( Iterator<ContainerType,Category>::index ); }
	__device__ virtual pointer operator->() { return &(Iterator<ContainerType,Category>::pContainer->operator[]( Iterator<ContainerType,Category>::index )); }
};

template<class ContainerType,class Category=std::forward_iterator_tag>
class ForwardIterator : public InputIterator<ContainerType,Category>, public OutputIterator<ContainerType,Category>
{
public:
	typedef ContainerType container_type;
	typedef Category iterator_category;
	typedef typename ContainerType::value_type value_type;
	typedef typename ContainerType::difference_type difference_type;
	typedef typename ContainerType::pointer pointer;
	typedef typename ContainerType::reference reference;
	typedef typename ContainerType::const_pointer const_pointer;
	typedef typename ContainerType::const_reference const_reference;

protected:
	virtual Iterator<ContainerType,Category> copy() const { return ForwardIterator<ContainerType,Category>(*this); }

public:
	__device__ ForwardIterator() : Iterator<ContainerType,Category>(), InputIterator<ContainerType,Category>( nullptr, 0 ), OutputIterator<ContainerType,Category>( nullptr, 0 ) {}
	__device__ ForwardIterator( ContainerType* container, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,Category>( container, index ), InputIterator<ContainerType,Category>( container, index ), OutputIterator<ContainerType,Category>( container, index ) {}
	__device__ ForwardIterator( const ForwardIterator<ContainerType,Category>& src ) : Iterator<ContainerType,Category>( src ), InputIterator<ContainerType,Category>( src ), OutputIterator<ContainerType,Category>( src ) {}
	__device__ virtual ~ForwardIterator() {}

	__device__ virtual const_reference operator*() const { return InputIterator<ContainerType,Category>::operator*(); }
	__device__ virtual const_pointer operator->() const { return InputIterator<ContainerType,Category>::operator->(); }
	__device__ virtual reference operator*() { return OutputIterator<ContainerType,Category>::operator*(); }
	__device__ virtual pointer operator->() { return OutputIterator<ContainerType,Category>::operator->(); }
};

template<class ContainerType,class Category=std::bidirectional_iterator_tag>
class BidirectionalIterator : public ForwardIterator<ContainerType,Category>
{
public:
	typedef ContainerType container_type;
	typedef Category iterator_category;
	typedef typename ContainerType::value_type value_type;
	typedef typename ContainerType::difference_type difference_type;
	typedef typename ContainerType::pointer pointer;
	typedef typename ContainerType::reference reference;
	typedef typename ContainerType::const_pointer const_pointer;
	typedef typename ContainerType::const_reference const_reference;

protected:
	__device__ virtual Iterator<ContainerType,Category> copy() const { return BidirectionalIterator<ContainerType,Category>(*this); }

public:
	__device__ BidirectionalIterator() : Iterator<ContainerType,Category>(), ForwardIterator<ContainerType,Category>() {}
	__device__ BidirectionalIterator( ContainerType* container, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,Category>( container, index ), ForwardIterator<ContainerType,Category>( container, index ) {}
	__device__ BidirectionalIterator( const BidirectionalIterator<ContainerType,Category>& src ) : Iterator<ContainerType,Category>( src ), ForwardIterator<ContainerType,Category>( src ) {}
	__device__ virtual ~BidirectionalIterator() {}

	__device__ BidirectionalIterator<ContainerType,Category>& operator--() { if( Iterator<ContainerType,Category>::index ) --Iterator<ContainerType,Category>::index; return *this; }
	__device__ BidirectionalIterator<ContainerType,Category> operator--( int ) const { return BidirectionalIterator<ContainerType,Category>( Iterator<ContainerType,Category>::pContainer, Iterator<ContainerType,Category>::index-1 ); }
};

template<class ContainerType,class Category=std::random_access_iterator_tag>
class RandomAccessIterator : public BidirectionalIterator<ContainerType,Category>
{
public:
	typedef ContainerType container_type;
	typedef Category iterator_category;
	typedef typename ContainerType::value_type value_type;
	typedef typename ContainerType::difference_type difference_type;
	typedef typename ContainerType::pointer pointer;
	typedef typename ContainerType::reference reference;
	typedef typename ContainerType::const_pointer const_pointer;
	typedef typename ContainerType::const_reference const_reference;

protected:
	__device__ virtual Iterator<ContainerType,Category> copy() const { return RandomAccessIterator<ContainerType,Category>(*this); }

public:
	__device__ RandomAccessIterator() : Iterator<ContainerType,Category>(), BidirectionalIterator<ContainerType,Category>() {}
	__device__ RandomAccessIterator( ContainerType* container, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,Category>( container, index ), BidirectionalIterator<ContainerType,Category>( container, index ) {}
	__device__ RandomAccessIterator( const RandomAccessIterator<ContainerType,Category>& src ) : Iterator<ContainerType,Category>( src ), BidirectionalIterator<ContainerType,Category>( src ) {}
	__device__ virtual ~RandomAccessIterator() {}

	//virtual difference_type operator-( const Iterator<ContainerType,Category>& other ) { return Iterator<ContainerType,Category>::operator-(other); }
	__device__ virtual difference_type operator-( const RandomAccessIterator<ContainerType,Category>& other ) { return Iterator<ContainerType,Category>::index - other.index; }

	__device__ RandomAccessIterator<ContainerType,Category> operator+( int x ) const { return RandomAccessIterator<ContainerType,Category>( Iterator<ContainerType,Category>::pContainer, Iterator<ContainerType,Category>::index+x ); }
	__device__ RandomAccessIterator<ContainerType,Category> operator-( int x ) const { return RandomAccessIterator<ContainerType,Category>( Iterator<ContainerType,Category>::pContainer, Iterator<ContainerType,Category>::index-x ); }

	__device__ virtual bool operator<( const RandomAccessIterator<ContainerType,Category>& other ) const { return /*Iterator<ContainerType,Category>::pContainer == other.pContainer and*/ Iterator<ContainerType,Category>::index < other.index; }
	__device__ virtual bool operator>( const RandomAccessIterator<ContainerType,Category>& other ) const { return /*Iterator<ContainerType,Category>::pContainer == other.pContainer and*/ Iterator<ContainerType,Category>::index > other.index; }
	__device__ virtual bool operator<=( const RandomAccessIterator<ContainerType,Category>& other ) const { return operator<(other) or InputIterator<ContainerType,Category>::operator==(other); }
	__device__ virtual bool operator>=( const RandomAccessIterator<ContainerType,Category>& other ) const { return operator>(other) or InputIterator<ContainerType,Category>::operator==(other); }

	__device__ RandomAccessIterator<ContainerType,Category>& operator+=( int x ) { Iterator<ContainerType,Category>::index += x; return *this; }
	__device__ RandomAccessIterator<ContainerType,Category>& operator-=( int x ) { Iterator<ContainerType,Category>::index -= x; return *this; }

	__device__ virtual reference operator[]( int x ) { return Iterator<ContainerType,Category>::pContainer->at( Iterator<ContainerType,Category>::index+x ); }
	__device__ virtual const_reference operator[]( int x ) const { return Iterator<ContainerType,Category>::pContainer->at( Iterator<ContainerType,Category>::index+x ); }

};

template<class ParentIterator>
class ReverseIterator : public ForwardIterator<typename ParentIterator::container_type,typename ParentIterator::iterator_category>
{
public:
	typedef typename ParentIterator::container_type container_type;
	typedef typename ParentIterator::iterator_category iterator_category;
	typedef typename ParentIterator::value_type value_type;
	typedef typename ParentIterator::difference_type difference_type;
	typedef typename ParentIterator::pointer pointer;
	typedef typename ParentIterator::reference reference;
	typedef typename ParentIterator::const_pointer const_pointer;
	typedef typename ParentIterator::const_reference const_reference;

private:
	ParentIterator parentIterator;

public:
	__device__ ReverseIterator() : Iterator<typename ParentIterator::container_type,typename ParentIterator::iterator_category>(nullptr), ForwardIterator<typename ParentIterator::container_type,typename ParentIterator::iterator_category>(nullptr){}
	__device__ ReverseIterator( ParentIterator parentIterator ) : Iterator<typename ParentIterator::container_type,typename ParentIterator::iterator_category>(nullptr), ForwardIterator<typename ParentIterator::container_type,typename ParentIterator::iterator_category>(nullptr), parentIterator(parentIterator) {}

	__device__ ParentIterator base() const { return parentIterator; }

	__device__ virtual bool operator==( const ReverseIterator<ParentIterator>& other ) const { return parentIterator.operator==( other.parentIterator ); }
	__device__ virtual bool operator!=( const ReverseIterator<ParentIterator>& other ) const { return parentIterator.operator!=( other.parentIterator ); }
	__device__ virtual const_reference operator*() const { return parentIterator.operator--(0).operator*(); }
	__device__ virtual const_pointer operator->() const { return parentIterator.operator--(0).operator->(); }

	__device__ virtual bool operator==( const ReverseIterator<ParentIterator>& other ) { return parentIterator.operator==( other.parentIterator ); }
	__device__ virtual bool operator!=( const ReverseIterator<ParentIterator>& other ) { return parentIterator.operator!=( other.parentIterator ); }
	__device__ virtual reference operator*() { return parentIterator.operator--(0).operator*(); }
	__device__ virtual pointer operator->() { return parentIterator.operator--(0).operator->(); }

	__device__ ReverseIterator<ParentIterator>& operator++() { --parentIterator; return *this; }
	__device__ ReverseIterator<ParentIterator> operator++( int x ) const { return ReverseIterator<ParentIterator>(parentIterator.operator--(x)); }

	__device__ ReverseIterator<ParentIterator>& operator--() { ++parentIterator; return *this; }
	__device__ ReverseIterator<ParentIterator> operator--( int x ) const { return ReverseIterator<ParentIterator>(parentIterator.operator++(x)); }

	__device__ ReverseIterator<ParentIterator> operator+( int x ) { return ReverseIterator<ParentIterator>(parentIterator.operator-(x)); }
	__device__ ReverseIterator<ParentIterator> operator-( int x ) { return ReverseIterator<ParentIterator>(parentIterator.operator+(x)); }
	__device__ bool operator<( const ReverseIterator<ParentIterator>& other ) const { return parentIterator.operator>=(other); }
	__device__ bool operator>( const ReverseIterator<ParentIterator>& other ) const { return parentIterator.operator<=(other); }
	__device__ bool operator<=( const ReverseIterator<ParentIterator>& other ) const { return operator<(other) or operator==(other); }
	__device__ bool operator>=( const ReverseIterator<ParentIterator>& other ) const { return operator>(other) or operator==(other); }
	__device__ ReverseIterator<ParentIterator>& operator+=( int x ) { parentIterator.operator-=(x); return *this; }
	__device__ ReverseIterator<ParentIterator>& operator-=( int x ) { parentIterator.operator+=(x); return *this; }

};

} // namespace ecuda

#endif
