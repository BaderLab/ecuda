#ifndef ECUDA_PTR_STRIDING_PTR_HPP
#define ECUDA_PTR_STRIDING_PTR_HPP

#include <iostream>

#include "../global.hpp"
#include "../type_traits.hpp"
#include "../utility.hpp"
#include "common.hpp"

namespace ecuda {

///
/// \brief A specialized pointer to striding memory.
///
/// A specialized pointer to device memory where traversal of the data takes into a "stride", or a
/// fixed number of elements that are skipped each time the pointer is incremented.
///
/// The specialization is used to create certain views of a matrix or cube (e.g. single matrix column).
///
/// Memory use can be conceptualized as:
/// \code
///   |--- stride ----| // in multiples of sizeof(T)
///   +-+-------------+
///   | |xxxxxxxxxxxxx|
///   | |xxxxxxxxxxxxx| x = allocated but not used
///   | |xxxxxxxxxxxxx|
///   | |xxxxxxxxxxxxx|
///   | |xxxxxxxxxxxxx|
///   | |xxxxxxxxxxxxx| ... etc. (total size of the allocation is not known internally by striding_ptr)
///   +-+--------+----+
/// \endcode
///
/// For example, a pointer that will traverse the first column of a 10x5 matrix containing elements
/// of type T could be represented with striding_ptr<T>(ptr,5), where ptr points to the first element
/// of the matrix.
///
template<typename T,typename P=typename std::add_pointer<T>::type>
class striding_ptr
{

public:
	typedef T              element_type;
	typedef P              pointer;
	typedef T&             reference;
	typedef const T&       const_reference;
	typedef std::size_t    size_type;
	typedef std::ptrdiff_t difference_type;

private:
	pointer ptr;
	size_type stride; //!< amount pointer should move to reach next element

	template<typename T2,typename PointerType2> friend class striding_ptr;

public:
	__HOST__ __DEVICE__ striding_ptr( pointer ptr = pointer(), size_type stride = 0 ) : ptr(ptr), stride(stride) {}
	__HOST__ __DEVICE__ striding_ptr( const striding_ptr& src ) : ptr(src.ptr), stride(src.stride) {}

	#ifdef __CPP11_SUPPORTED__
	__HOST__ __DEVICE__ striding_ptr( striding_ptr&& src ) : ptr(ecuda::move(src.ptr)), stride(ecuda::move(src.stride)) {}
	__HOST__ __DEVICE__ striding_ptr& operator=( striding_ptr&& src ) {
		ptr = ecuda::move(src.ptr);
		stride = ecuda::move(src.stride);
		return *this;
	}
	#endif

	__HOST__ __DEVICE__ inline pointer get() const { return ptr; }

	__HOST__ __DEVICE__ inline size_type get_stride() const { return stride; }

	__DEVICE__ inline reference       operator*()                       { return *ptr; }
	__DEVICE__ inline const_reference operator*() const                 { return *ptr; }
	__DEVICE__ inline pointer         operator->() const                { return ptr; }
	__DEVICE__ inline reference       operator[]( std::size_t i )       { return striding_ptr(*this).operator+=(i).operator*(); }
	__DEVICE__ inline const_reference operator[]( std::size_t i ) const { return striding_ptr(*this).operator+=(i).operator*(); }

	__HOST__ __DEVICE__ inline striding_ptr& operator++()      { ptr += stride; return *this; }
	__HOST__ __DEVICE__ inline striding_ptr& operator--()      { ptr -= stride; return *this; }
	__HOST__ __DEVICE__ inline striding_ptr  operator++( int ) { striding_ptr tmp(*this); ++(*this); return tmp; }
	__HOST__ __DEVICE__ inline striding_ptr  operator--( int ) { striding_ptr tmp(*this); --(*this); return tmp; }

	__HOST__ __DEVICE__ inline striding_ptr& operator+=( int x ) { ptr += x*stride; return *this; }
	__HOST__ __DEVICE__ inline striding_ptr& operator-=( int x ) { ptr -= x*stride; return *this; }

	__HOST__ __DEVICE__ inline striding_ptr operator+( int x ) const { striding_ptr tmp(*this); tmp += x; return tmp; }
	__HOST__ __DEVICE__ inline striding_ptr operator-( int x ) const { striding_ptr tmp(*this); tmp -= x; return tmp; }

	template<typename T2,typename P2> __HOST__ __DEVICE__ inline bool operator==( const striding_ptr<T2,P2>& other ) const { return ptr == other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ inline bool operator!=( const striding_ptr<T2,P2>& other ) const { return ptr != other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ inline bool operator< ( const striding_ptr<T2,P2>& other ) const { return ptr <  other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ inline bool operator> ( const striding_ptr<T2,P2>& other ) const { return ptr >  other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ inline bool operator<=( const striding_ptr<T2,P2>& other ) const { return ptr <= other.ptr; }
	template<typename T2,typename P2> __HOST__ __DEVICE__ inline bool operator>=( const striding_ptr<T2,P2>& other ) const { return ptr >= other.ptr; }

	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const striding_ptr& ptr ) {
		out << "striding_ptr(ptr=" << ptr.ptr << ";stride=" << ptr.stride << ")";
		return out;
	}

};

} // namespace ecuda

#endif
