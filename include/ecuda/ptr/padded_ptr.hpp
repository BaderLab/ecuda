#ifndef ECUDA_PTR_PADDED_PTR_HPP
#define ECUDA_PTR_PADDED_PTR_HPP

#include <iostream>

#include "../global.hpp"
#include "../type_traits.hpp"
#include "common.hpp"

namespace ecuda {

///
/// \brief A specialized pointer to padded memory.
///
/// A specialized pointer to device memory where traversal of the data takes into account an ignorable region
/// of padding after every fixed number of sequential elements.
///
/// The specialization is used to both represent 2D memory allocations using cudaMallocPitch() and to
/// create certain views of a cube (e.g. single row or column).
///
/// Memory use can be conceptualized as:
/// \code
///   |- width --|      // in multiples of sizeof(T)
///   |---- pitch ----| // in bytes
///   +----------+----+
///   |          |xxxx|
///   |          |xxxx| x = allocated but not used
///   |          |xxxx|
///   |          |xxxx|
///   |          |xxxx|
///   |          |xxxx| ... etc. (total size of the allocation is not known internally by padded_ptr)
///   +----------+----+
/// \endcode
///
template<typename T,class PointerType=typename std::add_pointer<T>::type>
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

private:
	template<typename U,typename V> struct change_type_keep_constness;
	template<typename U,typename V> struct change_type_keep_constness<      U*,V*      > { typedef V* type; };
	template<typename U,typename V> struct change_type_keep_constness<      U*,const V*> { typedef V* type; };
	template<typename U,typename V> struct change_type_keep_constness<const U*,V*      > { typedef const V* type; };
	template<typename U,typename V> struct change_type_keep_constness<const U*,const V*> { typedef const V* type; };

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

	#ifdef __CPP11_SUPPORTED__
	///
	/// \brief Checks if this stores a non-null pointer.
	///
	/// \return true if *this stores a pointer, false otherwise.
	///
	__HOST__ __DEVICE__ explicit operator bool() const __NOEXCEPT__ {
		//return ecuda::pointer_traits<pointer>().undress( ptr ) != NULL;
		return naked_cast<typename std::add_pointer<const element_type>::type>( ptr ) != NULL;
	}
	#else
	///
	/// \brief Checks if this stores a non-null pointer.
	///
	/// \return true if *this stores a pointer, false otherwise.
	///
	__HOST__ __DEVICE__ operator bool() const __NOEXCEPT__ {
		//return ecuda::pointer_traits<pointer>().undress( ptr ) != NULL;
		return naked_cast<typename std::add_pointer<const element_type>::type>( ptr ) != NULL;
	}
	#endif

	template<class Q> // assumes sizeof(Q) is a strict multiple of sizeof(T) TODO: can probably enforce this with an enable_if
	__HOST__ __DEVICE__ inline difference_type operator-( Q other ) {
		//typename pointer_traits<pointer>::naked_pointer start = reinterpret_cast<typename pointer_traits<pointer>::naked_pointer>( pointer_traits<Q>().undress(other) );
		//typename pointer_traits<pointer>::naked_pointer stop  = pointer_traits<pointer>().undress(ptr);
		typedef typename std::add_pointer<element_type>::type naked_pointer_type;
		naked_pointer_type start = naked_cast<naked_pointer_type>( other );
		naked_pointer_type stop  = naked_cast<naked_pointer_type>( ptr );
		difference_type n = (stop-start)*sizeof(T); // bytes difference
		return n;
	}

/*
	template<class Q> // assumes sizeof(Q) is a strict multiple of sizeof(T) TODO: can probably enforce this with an enable_if
	__HOST__ __DEVICE__ inline difference_type operator-( Q p ) {
		typename pointer_traits<pointer>::naked_pointer start = reinterpret_cast<typename pointer_traits<pointer>::naked_pointer>( pointer_traits<Q>().undress(p) );
		typename pointer_traits<pointer>::naked_pointer stop = pointer_traits<pointer>().undress(ptr);
		difference_type n = (stop-start)*sizeof(T); // bytes difference
		n -= n/pitch * (pitch-width*sizeof(T)); // substract pads
		// assumption comes into play here, where n/sizeof(T) should evaluate exactly
		// i.e. ( n % sizeof(T) ) > 0 should never be true
		n /= sizeof(T);
		// delegate to underlying pointer specializations to allow it
		// a chance to apply logic
		return pointer(ptr+n)-p;
	}
*/
	__HOST__ __DEVICE__ inline padded_ptr& operator++() {
		++ptr;
		if( (ptr-edge_ptr) == width ) {
			// skip padding
			//ptr = reinterpret_cast<pointer>(reinterpret_cast<typename pointer_traits<pointer>::char_pointer>(edge_ptr)+pitch );
			ptr = pointer( naked_cast<typename std::add_pointer<element_type>::type>( naked_cast<typename change_type_keep_constness<pointer,char*>::type>(edge_ptr)+pitch ) );
			//ptr = reinterpret_cast<pointer>(reinterpret_cast<typename __cast_to_char<pointer>::type>(edge_ptr)+pitch);
			edge_ptr = ptr;
		}
		return *this;
	}

	__HOST__ __DEVICE__ inline padded_ptr& operator--() {
		--ptr;
		if( ptr < edge_ptr ) {
			// skip padding
			//edge_ptr = reinterpret_cast<pointer>(reinterpret_cast<typename pointer_traits<pointer>::char_pointer>(edge_ptr)-pitch );
			edge_ptr = pointer( naked_cast<typename std::add_pointer<element_type>::type>( naked_cast<typename change_type_keep_constness<pointer,char*>::type>(edge_ptr)-pitch ) );
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
			//edge_ptr = reinterpret_cast<pointer>(reinterpret_cast<typename pointer_traits<pointer>::char_pointer>(edge_ptr)+nskips*pitch );
			edge_ptr = pointer( naked_cast<typename std::add_pointer<element_type>::type>( naked_cast<typename change_type_keep_constness<pointer,char*>::type>(edge_ptr)+nskips*pitch ) );
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
			//edge_ptr = reinterpret_cast<pointer>(reinterpret_cast<typename pointer_traits<pointer>::char_pointer>(edge_ptr)-nskips*pitch );
			edge_ptr = pointer( naked_cast<typename std::add_pointer<element_type>::type>( naked_cast<typename change_type_keep_constness<pointer,char*>::type>(edge_ptr)-nskips*pitch ) );
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

	__HOST__ __DEVICE__ inline padded_ptr operator+( std::size_t x ) const { return operator+( static_cast<int>(x) ); }
	__HOST__ __DEVICE__ inline padded_ptr operator-( std::size_t x ) const { return operator-( static_cast<int>(x) ); }

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
