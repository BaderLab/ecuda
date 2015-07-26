#ifndef ECUDA_PTR_PADDED_PTR_HPP
#define ECUDA_PTR_PADDED_PTR_HPP

#include <iostream>

#include "../global.hpp"
#include "../type_traits.hpp"
#include "common.hpp"

namespace ecuda {

template<typename T,class PointerType=typename type_traits<T>::pointer>
class padded_ptr
{

public:
	typedef T element_type;
	typedef PointerType pointer;
	//typedef typename __pointer_type<T>::pointer pointer;
	typedef T& reference;
	typedef const T& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

private:
	pointer edge_ptr;
	size_type pitch;
	size_type width;
	pointer ptr;

public:
	__HOST__ __DEVICE__ padded_ptr( pointer edge_ptr = pointer(), size_type pitch = size_type(), size_type width = size_type(), pointer ptr = pointer() ) : edge_ptr(edge_ptr), pitch(pitch), width(width), ptr(ptr) {
		if( !ptr ) this->ptr = edge_ptr;
	}
	template<typename T2,class PointerType2>
	__HOST__ __DEVICE__ padded_ptr( const padded_ptr<T2,PointerType2>& src ) : edge_ptr(src.get_edge()), pitch(src.get_pitch()), width(src.get_width()), ptr(src.get()) {}

	__HOST__ __DEVICE__ inline pointer get_edge() const { return edge_ptr; }
	__HOST__ __DEVICE__ inline size_type get_pitch() const { return pitch; }
	__HOST__ __DEVICE__ inline size_type get_width() const { return width; }
	__HOST__ __DEVICE__ inline pointer get() const { return ptr; }

	__HOST__ __DEVICE__ inline size_type get_remaining_width() const { return width-(ptr-edge_ptr); }

	__DEVICE__ inline reference operator*() { return *ptr; }
	__DEVICE__ inline const_reference operator*() const { return *ptr; }
	__DEVICE__ inline pointer operator->() const { return ptr; }
	__DEVICE__ inline reference operator[]( std::size_t i ) { return padded_ptr(*this).operator+=(i).operator*(); }
	__DEVICE__ inline const_reference operator[]( std::size_t i ) const { return padded_ptr(*this).operator+=(i).operator*(); }

	template<class Q> // assumes sizeof(Q) is a strict multiple of sizeof(T) TODO: can probably enforce this with an enable_if
	__HOST__ __DEVICE__ inline difference_type operator-( Q ptr ) {
		typename pointer_traits<pointer>::naked_pointer start = reinterpret_cast<typename pointer_traits<pointer>::naked_pointer>( pointer_traits<Q>().undress(ptr) );
		typename pointer_traits<pointer>::naked_pointer stop = pointer_traits<pointer>().undress(ptr);
		difference_type n = (stop-start)*sizeof(T); // bytes difference
		n -= n/pitch * (pitch-width*sizeof(T)); // substract pads
		// assumption comes into play here, where n/sizeof(T) should evaluate exactly
		// i.e. ( n % sizeof(T) ) > 0 should never be true
		n /= sizeof(T);
		// delegate to underlying pointer specializations to allow it
		// a chance to apply logic
		return pointer(ptr.get()+n)-ptr;
	}

	__HOST__ __DEVICE__ inline padded_ptr& operator++() {
		++ptr;
		if( (ptr-edge_ptr) == width ) {
			// skip padding
			ptr = reinterpret_cast<pointer>(reinterpret_cast<typename pointer_traits<pointer>::char_pointer>(edge_ptr)+pitch );
			//ptr = reinterpret_cast<pointer>(reinterpret_cast<typename __cast_to_char<pointer>::type>(edge_ptr)+pitch);
			edge_ptr = ptr;
		}
		return *this;
	}

	__HOST__ __DEVICE__ inline padded_ptr& operator--() {
		--ptr;
		if( ptr < edge_ptr ) {
			// skip padding
			edge_ptr = reinterpret_cast<pointer>(reinterpret_cast<typename pointer_traits<pointer>::char_pointer>(edge_ptr)-pitch );
			//edge_ptr = reinterpret_cast<pointer>(reinterpret_cast<typename __cast_to_char<pointer>::type>(edge_ptr)-pitch);
			ptr = edge_ptr + width - 1;
		}
		return *this;
	}

	__HOST__ __DEVICE__ inline padded_ptr operator++( int ) { padded_ptr tmp(*this); ++(*this); return tmp; }
	__HOST__ __DEVICE__ inline padded_ptr operator--( int ) { padded_ptr tmp(*this); --(*this); return tmp; }

	__HOST__ __DEVICE__ inline padded_ptr& operator+=( int x ) {
		ptr += x;
		if( (ptr-edge_ptr) >= width ) {
			// skip padding(s)
			const size_type nskips = (ptr-edge_ptr) / width;
			const size_type offset = (ptr-edge_ptr) % width;
			edge_ptr = reinterpret_cast<pointer>(reinterpret_cast<typename pointer_traits<pointer>::char_pointer>(edge_ptr)+nskips*pitch );
			//edge_ptr = reinterpret_cast<pointer>(reinterpret_cast<typename __cast_to_char<pointer>::type>(edge_ptr)+nskips*pitch);
			ptr = edge_ptr + offset;
		}
		return *this;
	}

	__HOST__ __DEVICE__ inline padded_ptr& operator-=( int x ) {
		ptr -= x;
		if( ptr < edge_ptr ) {
			// skip padding(s)
			const size_type nskips = (edge_ptr-ptr) / width;
			const size_type offset = (edge_ptr-ptr) % width;
			edge_ptr = reinterpret_cast<pointer>(reinterpret_cast<typename pointer_traits<pointer>::char_pointer>(edge_ptr)-nskips*pitch );
			//edge_ptr = reinterpret_cast<pointer>(reinterpret_cast<typename __cast_to_char<pointer>::type>(edge_ptr)-nskips*pitch);
			ptr = edge_ptr + (width-offset);
		}
		return *this;
	}

	__HOST__ __DEVICE__ inline padded_ptr operator+( int x ) const {
		padded_ptr tmp( *this );
		tmp += x;
		return tmp;
	}

	__HOST__ __DEVICE__ inline padded_ptr operator-( int x ) const {
		padded_ptr tmp( *this );
		tmp -= x;
		return tmp;
	}

	template<typename T2,typename PointerType2> __HOST__ __DEVICE__ bool operator==( const padded_ptr<T2,PointerType2>& other ) const { return ptr == other.ptr; }
	template<typename T2,typename PointerType2> __HOST__ __DEVICE__ bool operator!=( const padded_ptr<T2,PointerType2>& other ) const { return ptr != other.ptr; }
	template<typename T2,typename PointerType2> __HOST__ __DEVICE__ bool operator< ( const padded_ptr<T2,PointerType2>& other ) const { return ptr <  other.ptr; }
	template<typename T2,typename PointerType2> __HOST__ __DEVICE__ bool operator> ( const padded_ptr<T2,PointerType2>& other ) const { return ptr >  other.ptr; }
	template<typename T2,typename PointerType2> __HOST__ __DEVICE__ bool operator<=( const padded_ptr<T2,PointerType2>& other ) const { return ptr <= other.ptr; }
	template<typename T2,typename PointerType2> __HOST__ __DEVICE__ bool operator>=( const padded_ptr<T2,PointerType2>& other ) const { return ptr >= other.ptr; }

	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const padded_ptr& ptr ) {
		out << "padded_ptr(edge_ptr=" << ptr.edge_ptr << ";pitch=" << ptr.pitch << ";width=" << ptr.width << ";ptr=" << ptr.ptr << ")";
		return out;
	}

};


} // namespace ecuda

#endif
