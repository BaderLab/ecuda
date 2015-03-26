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
// iterators.hpp
// Iterators using pointers to device memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_ITERATORS_HPP
#define ECUDA_ITERATORS_HPP

#include <iterator>

#include "global.hpp"

namespace ecuda {

///
/// \brief Iterator template compatible with pointers to device memory.
///
/// This general iterator definition builds on top of the standard STL iterator but
/// is functional using device memory and within device code. Almost all of the
/// capabilities of a standard random access STL iterator are present, except for the
/// difference operator.  Thus, the category tag is set to indicate a bidirectional
/// iterator, even though most of the capabilities of a random access iterator are
/// also present. The iterator simply carries around a "pointer" (or an object with
/// pointer semantics) to device memory which is incremented or decremented as
/// appropriate.
///
/// Providing the ability to specify the template parameter PointerType explicitly
/// allows specialized pointers to be used (rather than assuming a naked pointer T*).
/// Specialized pointers can be designed to accomodate issues that arise from
/// a) traversing matrices and cubes in non-contiguous order and/or b) the memory padding
/// that exists in pitched memory allocations on the device. striding_ptr and pitched_ptr
/// are two such pointer specializations that deal with these issues, respectively.
///
/// Since this definition doesn't necessarily imply contiguous memory, a subclass
/// of this template called contiguous_pointer_iterator (which does impose this
/// requirement) is also available (since it includes the difference operator it
/// is considered a true random access iterator).
///
template<typename T,typename PointerType,class Category=std::bidirectional_iterator_tag>
class pointer_iterator : public std::iterator<Category,T,std::ptrdiff_t,PointerType>
{
private:
	typedef std::iterator<Category,T,std::ptrdiff_t,PointerType> base_iterator_type; //!< redeclares base STL iterator type to make later typedefs more compact

public:
	typedef typename base_iterator_type::iterator_category iterator_category; //!< STL iterator category
	typedef typename base_iterator_type::value_type value_type; //!< type of elements pointed by the iterator
	typedef typename base_iterator_type::difference_type difference_type; //!< type to represent difference between two iterators
	typedef typename base_iterator_type::pointer pointer; //!< type to represent a pointer to an element pointed by the iterator
	typedef typename base_iterator_type::reference reference; //!< type to represent a reference to an element pointed by the iterator

private:
	pointer ptr;

public:
	///
	/// \brief Default constructor.
	///
	/// \param ptr Pointer to device memory location that holds the element to be pointed at.
	///
	HOST DEVICE pointer_iterator( const PointerType& ptr = PointerType() ) : ptr(ptr) {}

	///
	/// \brief Copy constructor.
	///
	/// \param src Another iterator whose contents are to be copied.
	///
	HOST DEVICE pointer_iterator( const pointer_iterator<T,PointerType,Category>& src ) : ptr(src.ptr) {}

	///
	/// \brief Copy constructor.
	///
	/// The element type and pointer type of the other iterator can be of a different types than
	/// this iterator, but they must be implicitly convertible to the type(s) in this iterator.
	/// This is currently utilized to allow an iterator pointing to non-const elements to be converted
	/// to one that does.
	///
	/// \param src Another iterator whose contents are to be copied.
	///
	template<typename T2,typename PointerType2>
	HOST DEVICE pointer_iterator( const pointer_iterator<T2,PointerType2,Category>& src ) : ptr(src.operator->()) {}

	///
	/// \brief Destructor.
	///
	HOST DEVICE ~pointer_iterator() {}

	///
	/// \brief Prefix increments the position of the iterator.
	///
	/// This ability is required of all STL iterators.
	///
	HOST DEVICE inline pointer_iterator& operator++() { ++ptr; return *this; }

	///
	/// \brief Postfix increments the position of the iterator.
	///
	/// This ability is required of all STL iterators.
	///
	HOST DEVICE inline pointer_iterator operator++( int ) {
		pointer_iterator tmp(*this);
		++(*this);
		// operator++(); // nvcc V6.0.1 didn't like this but above line works
		return tmp;
	}

	///
	/// \brief Prefix decrements the position of the iterator.
	///
	/// This ability is required of bidirectional STL iterators.
	///
	HOST DEVICE inline pointer_iterator& operator--() { --ptr; return *this; }

	///
	/// \brief Postfix decrements the position of the iterator.
	///
	/// This ability is required of bidirectional STL iterators.
	///
	HOST DEVICE inline pointer_iterator& operator--( int ) const {
		pointer_iterator tmp(*this);
		--(*this);
		// operator--(); // nvcc V6.0.1 didn't like this but above line works
		return tmp;
	}

	///
	/// \brief Equality comparison of this iterator with another.
	///
	/// This ability is required of input STL iterators.
	///
	/// \returns true if this iterator points to the same element as the other.
	///
	HOST DEVICE bool operator==( const pointer_iterator& other ) const { return ptr == other.ptr; }

	///
	/// \brief Inequality comparison of this iterator with another.
	///
	/// This ability is required of input STL iterators.
	///
	/// \returns true if this iterator does not point to the same element as the other.
	///
	HOST DEVICE bool operator!=( const pointer_iterator& other ) const { return !operator==(other); }

	///
	/// \brief Gets a reference to the element pointed at by this iterator.
	///
	/// Since this is an lvalue (even though the type may be const) it satisfies
	/// the requirements of both an input and output STL iterator.
	///
	/// \returns a reference to the element pointed at by this iterator
	///
	DEVICE reference operator*() const { return *ptr; }

	///
	/// \brief Gets a pointer to the element pointed at by this iterator.
	///
	/// This ability is required of input STL iterators.
	///
	/// \returns a pointer to the element pointed at by this iterator
	///
	HOST DEVICE pointer operator->() const { return ptr; } // have to declare HOST here to allow conversion pointer_iterator<T,T*> -> pointer_iterator<const T,const T*>

	///
	/// \brief Creates an iterator pointing to a later element some specified positions away.
	///
	/// This ability is required of random access STL iterators.
	///
	/// \param x The number of positions ahead to move the new iterator to.
	/// \returns Another iterator which points to an element x positions after this iterator's element.
	///
	HOST DEVICE inline pointer_iterator operator+( int x ) const { return pointer_iterator( ptr + x ); }

	///
	/// \brief Creates an iterator pointing to a prior element some specified positions away.
	///
	/// This ability is required of random access STL iterators.
	///
	/// \param x The number of positions prior to move the new iterator to.
	/// \returns Another iterator which points to an element x positions before this iterator's element.
	///
	HOST DEVICE inline pointer_iterator operator-( int x ) const { return pointer_iterator( ptr - x ); }

	///
	/// \brief Checks if the element pointed at by this iterator comes before another.
	///
	/// This ability is required of random access STL iterators.
	///
	/// \param other Another iterator to compare element location with.
	/// \returns true if the element pointed at by this iterator comes before the element pointed at by other.
	///
	HOST DEVICE bool operator<( const pointer_iterator& other ) const { return ptr < other.ptr; }

	///
	/// \brief Checks if the element pointed at by this iterator comes after another.
	///
	/// This ability is required of random access STL iterators.
	///
	/// \param other Another iterator to compare element location with.
	/// \returns true if the element pointed at by this iterator comes after the element pointed at by other.
	///
	HOST DEVICE bool operator>( const pointer_iterator& other ) const { return ptr > other.ptr; }

	///
	/// \brief Checks if the element pointed at by this iterator is equal to or comes before another.
	///
	/// This ability is required of random access STL iterators.
	///
	/// \param other Another iterator to compare element location with.
	/// \returns true if the element pointed at by this iterator is equal to or comes before the element pointed at by other.
	///
	HOST DEVICE bool operator<=( const pointer_iterator& other ) const { return operator<(other) or operator==(other); }

	///
	/// \brief Checks if the element pointed at by this iterator is equal to or comes after another.
	///
	/// This ability is required of random access STL iterators.
	///
	/// \param other Another iterator to compare element location with.
	/// \returns true if the element pointed at by this iterator is equal to or comes after the element pointed at by other.
	///
	HOST DEVICE bool operator>=( const pointer_iterator& other ) const { return operator>(other) or operator==(other); }

	///
	/// \brief Increments the position of this iterator by some amount.
	///
	/// This ability is required of random access STL iterators.
	///
	/// \param x The number of positions to increment this iterator by.
	///
	HOST DEVICE inline pointer_iterator& operator+=( int x ) { ptr += x; return *this; }

	///
	/// \brief Decrements the position of this iterator by some amount.
	///
	/// This ability is required of random access STL iterators.
	///
	/// \param x The number of positions to increment this iterator by.
	///
	HOST DEVICE inline pointer_iterator& operator-=( int x ) { ptr -= x; return *this; }

	///
	/// \brief Gets a reference to an element whose position is offset by a specified amount from this iterator's element.
	/// \param x The amount to offset the current position by (can be positive or negative).
	/// \returns a reference to the offset element
	///
	DEVICE reference operator[]( int x ) const { return *(ptr+x); }

	///
	/// \brief Assigns a copy of another iterators position to this iterator.
	/// \param other another iterator whose position should be assigned to this iterator
	///
	HOST DEVICE pointer_iterator& operator=( const pointer_iterator<T,PointerType,Category>& other ) {
		ptr = other.ptr;
		return *this;
	}

	///
	/// \brief Assigns a copy of another iterators position to this iterator.
	///
	/// The element type and pointer type of the other iterator can be of a different types than
	/// this iterator, but they must be implicitly convertible to the type(s) in this iterator.
	/// This is currently utilized to allow an iterator pointing to non-const elements to be converted
	/// to one that does.
	///
	/// \param other another iterator whose position should be assigned to this iterator
	///
	template<typename T2,typename PointerType2>
	HOST DEVICE pointer_iterator& operator=( const pointer_iterator<T2,PointerType2,Category>& other ) {
		ptr = other.ptr;
		return *this;
	}

};

///
/// \brief Iterator template for use with naked pointers to contiguous device memory.
///
template<typename T>
class contiguous_pointer_iterator : public pointer_iterator<T,T*,std::random_access_iterator_tag>
{

private:
	typedef pointer_iterator<T,T*,std::random_access_iterator_tag> base_iterator_type; //!< redeclares base pointer_iterator type to make later typedefs more compact

public:
	typedef typename base_iterator_type::iterator_category iterator_category; //!< STL iterator category
	typedef typename base_iterator_type::value_type value_type; //!< type of elements pointed by the iterator
	typedef typename base_iterator_type::difference_type difference_type; //!< type to represent difference between two iterators
	typedef typename base_iterator_type::pointer pointer; //!< type to represent a pointer to an element pointed by the iterator
	typedef typename base_iterator_type::reference reference; //!< type to represent a reference to an element pointed by the iterator

public:
	///
	/// \brief Default constructor.
	///
	/// \param ptr Pointer to device memory location that holds the element to be pointed at.
	///
	HOST DEVICE contiguous_pointer_iterator( T* ptr = nullptr ) : base_iterator_type(ptr) {}

	///
	/// \brief Copy constructor.
	///
	/// \param src Another iterator whose contents are to be copied.
	///
	HOST DEVICE contiguous_pointer_iterator( const contiguous_pointer_iterator<T>& src ) : base_iterator_type(src) {}

	///
	/// \brief Copy constructor.
	///
	/// The element type and pointer type of the other iterator can be of a different types than
	/// this iterator, but they must be implicitly convertible to the type(s) in this iterator.
	/// This is currently utilized to allow an iterator pointing to non-const elements to be converted
	/// to one that does.
	///
	/// \param src Another iterator whose contents are to be copied.
	///
	template<typename T2>
	HOST DEVICE contiguous_pointer_iterator( const contiguous_pointer_iterator<T2>& src ) : base_iterator_type(src.operator->()) {}

	///
	/// \brief Gets the difference in location between the element pointed at by this iterator and another.
	///
	/// This ability is required of random access STL iterators.
	///
	/// \param other Another iterator with which to determine the difference in location.
	/// \returns the difference in location between the element pointed at by this iterator and other
	///
	HOST DEVICE inline difference_type operator-( const contiguous_pointer_iterator& other ) { return base_iterator_type::operator->() - other.operator->(); }

};

///
/// \brief Reverse iterator.
///
/// Given a BaseIterator of type pointer_iterator, this provides the same capabilities
/// as the given pointer_iterator, but in reverse order.  The same strategy as standard
/// STL reverse iterators is used: the provided base iterator is assumed to point
/// to an element one position ahead of the element that the reverse iterator wishes
/// to refer to.  All increment/decrement operations are reversed in this iterator.
///
/// Note that this single-position offset in the base iterator imposes on operations
/// where the element itself is retrieved (e.g. operator*()) the need to correct the
/// offset prior to getting the value.  This introduces some additional overhead
/// (although a smart compiler may be able to reduce this to a infinitesimal cost).
///
/// Any undocumented methods are exactly the same as their counterpart in pointer_iterator.
///
template<class BaseIterator>
class pointer_reverse_iterator : public std::iterator<typename BaseIterator::iterator_category,typename BaseIterator::value_type,typename BaseIterator::difference_type,typename BaseIterator::pointer>
{

public:
	typedef BaseIterator iterator_type; //!< type of the parent iterator being reversed
	typedef typename BaseIterator::iterator_category iterator_category; //!< STL iterator category
	typedef typename BaseIterator::value_type value_type; //!< type of elements pointed by the iterator
	typedef typename BaseIterator::difference_type difference_type; //!< type to represent difference between two iterators
	typedef typename BaseIterator::pointer pointer; //!< type to represent a pointer to an element pointed by the iterator
	typedef typename BaseIterator::reference reference; //!< type to represent a reference to an element pointed by the iterator

private:
	BaseIterator parentIterator; //!< parent iterator being operated on in reverse

public:
	HOST DEVICE pointer_reverse_iterator( BaseIterator parentIterator = BaseIterator() ) : parentIterator(parentIterator) {}

	HOST DEVICE pointer_reverse_iterator( const pointer_reverse_iterator& src ) : parentIterator(src.parentIterator) {}

	template<class ParentIterator2>
	HOST DEVICE pointer_reverse_iterator( const pointer_reverse_iterator<ParentIterator2>& src ) : parentIterator(src.base()) {}

	HOST DEVICE virtual ~pointer_reverse_iterator() {}

	///
	/// \brief Returns a copy of the base iterator.
	///
	///	The base iterator is an iterator of the same type as the one used to construct the pointer_reverse_iterator,
	/// but pointing to the element next to the one the pointer_reverse_iterator is currently pointing to (a
	/// pointer_reverse_iterator has always an offset of -1 with respect to its base iterator).
	///
	/// \returns A copy of the base iterator, which iterates in the opposite direction.
	///
	HOST DEVICE BaseIterator base() const { return parentIterator; }

	HOST DEVICE inline pointer_reverse_iterator& operator++() { --parentIterator; return *this; }
	HOST DEVICE inline pointer_reverse_iterator operator++( int ) {
		pointer_reverse_iterator tmp(*this);
		++(*this);
		// operator++(); // nvcc V6.0.1 didn't like this but above line works
		return tmp;
	}

	HOST DEVICE inline pointer_reverse_iterator& operator--() { ++parentIterator; return *this; }
	HOST DEVICE inline pointer_reverse_iterator& operator--( int ) const {
		pointer_reverse_iterator tmp(*this);
		--(*this);
		// operator--(); // nvcc V6.0.1 didn't like this but above line works
		return tmp;
	}

	HOST DEVICE inline bool operator==( const pointer_reverse_iterator& other ) const { return parentIterator == other.parentIterator; }
	HOST DEVICE inline bool operator!=( const pointer_reverse_iterator& other ) const { return !operator==(other); }

	DEVICE inline reference operator*() const {
		BaseIterator tmp(parentIterator);
		--tmp;
		return tmp.operator*();
	}
	DEVICE inline pointer operator->() const {
		BaseIterator tmp(parentIterator);
		--tmp;
		return tmp.operator->();
	}

	HOST DEVICE inline difference_type operator-( const pointer_reverse_iterator& other ) { return parentIterator - other.parentIterator; }

	HOST DEVICE inline pointer_reverse_iterator operator+( int x ) const { return pointer_reverse_iterator( parentIterator-x ); }
	HOST DEVICE inline pointer_reverse_iterator operator-( int x ) const { return pointer_reverse_iterator( parentIterator+x ); }

	HOST DEVICE inline bool operator<( const pointer_reverse_iterator& other ) const { return parentIterator < other.parentIterator; }
	HOST DEVICE inline bool operator>( const pointer_reverse_iterator& other ) const { return parentIterator > other.parentIterator; }
	HOST DEVICE inline bool operator<=( const pointer_reverse_iterator& other ) const { return operator<(other) or operator==(other); }
	HOST DEVICE inline bool operator>=( const pointer_reverse_iterator& other ) const { return operator>(other) or operator==(other); }

	HOST DEVICE inline pointer_reverse_iterator& operator+=( int x ) { parentIterator -= x; return *this; }
	HOST DEVICE inline pointer_reverse_iterator& operator-=( int x ) { parentIterator += x; return *this; }

	DEVICE reference operator[]( int x ) const { return parentIterator.operator[]( -x-1 ); }

	HOST DEVICE pointer_reverse_iterator& operator=( const pointer_reverse_iterator& other ) {
		parentIterator = other.parentIterator;
		return *this;
	}

	template<class ParentIterator2>
	HOST DEVICE pointer_reverse_iterator& operator=( const pointer_reverse_iterator<ParentIterator2>& other ) {
		parentIterator = other.parentIterator;
		return *this;
	}

};

} // namespace ecuda

#endif
