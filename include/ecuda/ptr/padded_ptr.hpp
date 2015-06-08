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
	__host__ __device__ padded_ptr( pointer edge_ptr = pointer(), size_type pitch = size_type(), size_type width = size_type(), pointer ptr = pointer() ) : edge_ptr(edge_ptr), pitch(pitch), width(width), ptr(ptr) {}
	template<typename T2,class PointerType2>
	__host__ __device__ padded_ptr( const padded_ptr<T2,PointerType2>& src ) : edge_ptr(src.get_edge()), pitch(src.get_pitch()), width(src.get_width()), ptr(src.get()) {}

	__host__ __device__ inline pointer get_edge() const { return edge_ptr; }
	__host__ __device__ inline size_type get_pitch() const { return pitch; }
	__host__ __device__ inline size_type get_width() const { return width; }
	__host__ __device__ inline pointer get() const { return ptr; }

	__device__ inline reference operator*() { return *ptr; }
	__device__ inline const_reference operator*() const { return *ptr; }
	__device__ inline pointer operator->() const { return ptr; }

	__host__ __device__ inline padded_ptr& operator++() {
		++ptr;
		if( (ptr-edge_ptr) == width ) {
			// skip padding
			ptr = reinterpret_cast<pointer>(reinterpret_cast<typename pointer_traits<pointer>::char_pointer>(edge_ptr)+pitch );
			//ptr = reinterpret_cast<pointer>(reinterpret_cast<typename __cast_to_char<pointer>::type>(edge_ptr)+pitch);
			edge_ptr = ptr;
		}
		return *this;
	}

	__host__ __device__ inline padded_ptr& operator--() {
		--ptr;
		if( ptr < edge_ptr ) {
			// skip padding
			edge_ptr = reinterpret_cast<pointer>(reinterpret_cast<typename pointer_traits<pointer>::char_pointer>(edge_ptr)-pitch );
			//edge_ptr = reinterpret_cast<pointer>(reinterpret_cast<typename __cast_to_char<pointer>::type>(edge_ptr)-pitch);
			ptr = edge_ptr + width - 1;
		}
		return *this;
	}

	__host__ __device__ inline padded_ptr operator++( int ) { padded_ptr tmp(*this); ++(*this); return tmp; }
	__host__ __device__ inline padded_ptr operator--( int ) { padded_ptr tmp(*this); --(*this); return tmp; }

	__host__ __device__ inline padded_ptr& operator+=( int x ) {
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

	__host__ __device__ inline padded_ptr& operator-=( int x ) {
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

	__host__ __device__ inline padded_ptr operator+( int x ) const {
		padded_ptr tmp( *this );
		tmp += x;
		return tmp;
	}

	__host__ __device__ inline padded_ptr operator-( int x ) const {
		padded_ptr tmp( *this );
		tmp -= x;
		return tmp;
	}

	template<typename T2,typename PointerType2> __host__ __device__ bool operator==( const padded_ptr<T2,PointerType2>& other ) const { return ptr == other.ptr; }
	template<typename T2,typename PointerType2> __host__ __device__ bool operator!=( const padded_ptr<T2,PointerType2>& other ) const { return ptr != other.ptr; }
	template<typename T2,typename PointerType2> __host__ __device__ bool operator< ( const padded_ptr<T2,PointerType2>& other ) const { return ptr <  other.ptr; }
	template<typename T2,typename PointerType2> __host__ __device__ bool operator> ( const padded_ptr<T2,PointerType2>& other ) const { return ptr >  other.ptr; }
	template<typename T2,typename PointerType2> __host__ __device__ bool operator<=( const padded_ptr<T2,PointerType2>& other ) const { return ptr <= other.ptr; }
	template<typename T2,typename PointerType2> __host__ __device__ bool operator>=( const padded_ptr<T2,PointerType2>& other ) const { return ptr >= other.ptr; }

	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const padded_ptr& ptr ) {
		out << "padded_ptr(edge_ptr=" << ptr.edge_ptr << ";pitch=" << ptr.pitch << ";width=" << ptr.width << ";ptr=" << ptr.ptr << ")";
		return out;
	}

};


} // namespace ecuda

#endif
