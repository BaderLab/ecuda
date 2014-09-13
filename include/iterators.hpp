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

//#ifdef __CUDA_ARCH__
#define nullptr NULL
//#endif

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
	DEVICE virtual Iterator<ContainerType,Category> copy() const { return Iterator<ContainerType,Category>(*this); }

public:
	DEVICE Iterator( ContainerType* pContainer, const typename ContainerType::size_type index = 0 ) : pContainer(pContainer), index(index) {}
	DEVICE Iterator( const Iterator<ContainerType,Category>& src ) : pContainer(src.pContainer), index(src.index) {}
	DEVICE virtual ~Iterator() {}

	//virtual difference_type operator-( const RandomAccessIterator<ContainerType,Category>& other ) { return index-other.index; }

	DEVICE Iterator<ContainerType,Category>& operator++() {	if( index < pContainer->size() ) ++index; return *this; }
	DEVICE Iterator<ContainerType,Category> operator++( int ) const { return ++(this->copy()); }

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
	DEVICE virtual Iterator<ContainerType,Category> copy() const { return InputIterator<ContainerType,Category>(*this); }

public:
	DEVICE InputIterator( ContainerType* container, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,Category>( container, index ) {}
	DEVICE InputIterator( const InputIterator<ContainerType,Category>& src ) : Iterator<ContainerType,Category>( src ) {}
	DEVICE virtual ~InputIterator() {}

	DEVICE virtual bool operator==( const InputIterator<ContainerType,Category>& other ) const { return /*Iterator<ContainerType,IteratorTag>::pContainer == other.pContainer and*/ Iterator<ContainerType,Category>::index == other.index; }
	DEVICE virtual bool operator!=( const InputIterator<ContainerType,Category>& other ) const { return !operator==(other); }
	DEVICE virtual const_reference operator*() const { return Iterator<ContainerType,Category>::pContainer->operator[]( Iterator<ContainerType,Category>::index ); }
	DEVICE virtual const_pointer operator->() const { return &(Iterator<ContainerType,Category>::pContainer->operator[]( Iterator<ContainerType,Category>::index )); }
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
	DEVICE virtual Iterator<ContainerType,Category> copy() const { return OutputIterator<ContainerType,Category>(*this); }

public:
	DEVICE OutputIterator( ContainerType* container, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,Category>( container, index ) {}
	DEVICE OutputIterator( const OutputIterator<ContainerType,Category>& src ) : Iterator<ContainerType,Category>( src ) {}
	DEVICE virtual ~OutputIterator() {}

	DEVICE virtual reference operator*() {	return Iterator<ContainerType,Category>::pContainer->operator[]( Iterator<ContainerType,Category>::index ); }
	DEVICE virtual pointer operator->() { return &(Iterator<ContainerType,Category>::pContainer->operator[]( Iterator<ContainerType,Category>::index )); }
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
	DEVICE ForwardIterator() : Iterator<ContainerType,Category>(), InputIterator<ContainerType,Category>( nullptr, 0 ), OutputIterator<ContainerType,Category>( nullptr, 0 ) {}
	DEVICE ForwardIterator( ContainerType* container, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,Category>( container, index ), InputIterator<ContainerType,Category>( container, index ), OutputIterator<ContainerType,Category>( container, index ) {}
	DEVICE ForwardIterator( const ForwardIterator<ContainerType,Category>& src ) : Iterator<ContainerType,Category>( src ), InputIterator<ContainerType,Category>( src ), OutputIterator<ContainerType,Category>( src ) {}
	DEVICE virtual ~ForwardIterator() {}

	DEVICE virtual const_reference operator*() const { return InputIterator<ContainerType,Category>::operator*(); }
	DEVICE virtual const_pointer operator->() const { return InputIterator<ContainerType,Category>::operator->(); }
	DEVICE virtual reference operator*() { return OutputIterator<ContainerType,Category>::operator*(); }
	DEVICE virtual pointer operator->() { return OutputIterator<ContainerType,Category>::operator->(); }
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
	DEVICE virtual Iterator<ContainerType,Category> copy() const { return BidirectionalIterator<ContainerType,Category>(*this); }

public:
	DEVICE BidirectionalIterator() : Iterator<ContainerType,Category>(), ForwardIterator<ContainerType,Category>() {}
	DEVICE BidirectionalIterator( ContainerType* container, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,Category>( container, index ), ForwardIterator<ContainerType,Category>( container, index ) {}
	DEVICE BidirectionalIterator( const BidirectionalIterator<ContainerType,Category>& src ) : Iterator<ContainerType,Category>( src ), ForwardIterator<ContainerType,Category>( src ) {}
	DEVICE virtual ~BidirectionalIterator() {}

	DEVICE BidirectionalIterator<ContainerType,Category>& operator--() { if( Iterator<ContainerType,Category>::index ) --Iterator<ContainerType,Category>::index; return *this; }
	DEVICE BidirectionalIterator<ContainerType,Category> operator--( int ) const { return BidirectionalIterator<ContainerType,Category>( Iterator<ContainerType,Category>::pContainer, Iterator<ContainerType,Category>::index-1 ); }
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
	DEVICE virtual Iterator<ContainerType,Category> copy() const { return RandomAccessIterator<ContainerType,Category>(*this); }

public:
	DEVICE RandomAccessIterator() : Iterator<ContainerType,Category>(), BidirectionalIterator<ContainerType,Category>() {}
	DEVICE RandomAccessIterator( ContainerType* container, const typename ContainerType::size_type index = 0 ) : Iterator<ContainerType,Category>( container, index ), BidirectionalIterator<ContainerType,Category>( container, index ) {}
	DEVICE RandomAccessIterator( const RandomAccessIterator<ContainerType,Category>& src ) : Iterator<ContainerType,Category>( src ), BidirectionalIterator<ContainerType,Category>( src ) {}
	DEVICE virtual ~RandomAccessIterator() {}

	//virtual difference_type operator-( const Iterator<ContainerType,Category>& other ) { return Iterator<ContainerType,Category>::operator-(other); }
	DEVICE virtual difference_type operator-( const RandomAccessIterator<ContainerType,Category>& other ) { return Iterator<ContainerType,Category>::index - other.index; }

	DEVICE RandomAccessIterator<ContainerType,Category> operator+( int x ) const { return RandomAccessIterator<ContainerType,Category>( Iterator<ContainerType,Category>::pContainer, Iterator<ContainerType,Category>::index+x ); }
	DEVICE RandomAccessIterator<ContainerType,Category> operator-( int x ) const { return RandomAccessIterator<ContainerType,Category>( Iterator<ContainerType,Category>::pContainer, Iterator<ContainerType,Category>::index-x ); }

	DEVICE virtual bool operator<( const RandomAccessIterator<ContainerType,Category>& other ) const { return /*Iterator<ContainerType,Category>::pContainer == other.pContainer and*/ Iterator<ContainerType,Category>::index < other.index; }
	DEVICE virtual bool operator>( const RandomAccessIterator<ContainerType,Category>& other ) const { return /*Iterator<ContainerType,Category>::pContainer == other.pContainer and*/ Iterator<ContainerType,Category>::index > other.index; }
	DEVICE virtual bool operator<=( const RandomAccessIterator<ContainerType,Category>& other ) const { return operator<(other) or InputIterator<ContainerType,Category>::operator==(other); }
	DEVICE virtual bool operator>=( const RandomAccessIterator<ContainerType,Category>& other ) const { return operator>(other) or InputIterator<ContainerType,Category>::operator==(other); }

	DEVICE RandomAccessIterator<ContainerType,Category>& operator+=( int x ) { Iterator<ContainerType,Category>::index += x; return *this; }
	DEVICE RandomAccessIterator<ContainerType,Category>& operator-=( int x ) { Iterator<ContainerType,Category>::index -= x; return *this; }

	DEVICE virtual reference operator[]( int x ) { return Iterator<ContainerType,Category>::pContainer->at( Iterator<ContainerType,Category>::index+x ); }
	DEVICE virtual const_reference operator[]( int x ) const { return Iterator<ContainerType,Category>::pContainer->at( Iterator<ContainerType,Category>::index+x ); }

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
	DEVICE ReverseIterator() : Iterator<typename ParentIterator::container_type,typename ParentIterator::iterator_category>(nullptr), ForwardIterator<typename ParentIterator::container_type,typename ParentIterator::iterator_category>(nullptr){}
	DEVICE ReverseIterator( ParentIterator parentIterator ) : Iterator<typename ParentIterator::container_type,typename ParentIterator::iterator_category>(nullptr), ForwardIterator<typename ParentIterator::container_type,typename ParentIterator::iterator_category>(nullptr), parentIterator(parentIterator) {}

	DEVICE ParentIterator base() const { return parentIterator; }

	DEVICE virtual bool operator==( const ReverseIterator<ParentIterator>& other ) const { return parentIterator.operator==( other.parentIterator ); }
	DEVICE virtual bool operator!=( const ReverseIterator<ParentIterator>& other ) const { return parentIterator.operator!=( other.parentIterator ); }
	DEVICE virtual const_reference operator*() const { return parentIterator.operator--(0).operator*(); }
	DEVICE virtual const_pointer operator->() const { return parentIterator.operator--(0).operator->(); }

	DEVICE virtual bool operator==( const ReverseIterator<ParentIterator>& other ) { return parentIterator.operator==( other.parentIterator ); }
	DEVICE virtual bool operator!=( const ReverseIterator<ParentIterator>& other ) { return parentIterator.operator!=( other.parentIterator ); }
	DEVICE virtual reference operator*() { return parentIterator.operator--(0).operator*(); }
	DEVICE virtual pointer operator->() { return parentIterator.operator--(0).operator->(); }

	DEVICE ReverseIterator<ParentIterator>& operator++() { --parentIterator; return *this; }
	DEVICE ReverseIterator<ParentIterator> operator++( int x ) const { return ReverseIterator<ParentIterator>(parentIterator.operator--(x)); }

	DEVICE ReverseIterator<ParentIterator>& operator--() { ++parentIterator; return *this; }
	DEVICE ReverseIterator<ParentIterator> operator--( int x ) const { return ReverseIterator<ParentIterator>(parentIterator.operator++(x)); }

	DEVICE ReverseIterator<ParentIterator> operator+( int x ) { return ReverseIterator<ParentIterator>(parentIterator.operator-(x)); }
	DEVICE ReverseIterator<ParentIterator> operator-( int x ) { return ReverseIterator<ParentIterator>(parentIterator.operator+(x)); }
	DEVICE bool operator<( const ReverseIterator<ParentIterator>& other ) const { return parentIterator.operator>=(other); }
	DEVICE bool operator>( const ReverseIterator<ParentIterator>& other ) const { return parentIterator.operator<=(other); }
	DEVICE bool operator<=( const ReverseIterator<ParentIterator>& other ) const { return operator<(other) or operator==(other); }
	DEVICE bool operator>=( const ReverseIterator<ParentIterator>& other ) const { return operator>(other) or operator==(other); }
	DEVICE ReverseIterator<ParentIterator>& operator+=( int x ) { parentIterator.operator-=(x); return *this; }
	DEVICE ReverseIterator<ParentIterator>& operator-=( int x ) { parentIterator.operator+=(x); return *this; }

};

} // namespace ecuda

#endif
