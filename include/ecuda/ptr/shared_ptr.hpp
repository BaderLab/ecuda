#ifndef ECUDA_PTR_SHARED_PTR_HPP
#define ECUDA_PTR_SHARED_PTR_HPP

#include <iostream>

#include "../global.hpp"
#include "common.hpp"
#include "../algorithm.hpp"
#include "../type_traits.hpp"

//
// Implementation is the approach used by boost::shared_ptr.
//

namespace ecuda {

namespace detail {

struct sp_counter_base {
	std::size_t owner_count;
	sp_counter_base() : owner_count(1) {}
	virtual ~sp_counter_base() {}
	virtual void dispose( void* ) = 0;
	virtual void* get_deleter() { return 0; }
};

template<typename T>
struct sp_counter_impl_p : sp_counter_base {
	sp_counter_impl_p() : sp_counter_base() {}
	virtual void dispose( void* p ) { if( p ) cudaFree( p ); }
};

template<typename T,class Deleter>
struct sp_counter_impl_pd : sp_counter_base {
	Deleter deleter;
	sp_counter_impl_pd( Deleter deleter ) : sp_counter_base(), deleter(deleter) {}
	virtual void dispose( void* p ) { if( p ) deleter(p); }
	virtual void* get_deleter() { return deleter; }
};

// this is hacky structure that takes any pointer (const or not)
// and casts it to void* so it can be used by the deleter dispose() method
template<typename T> struct __cast_void;
template<typename T> struct __cast_void<const T*> { inline void* operator()( const T* ptr ) { return reinterpret_cast<void*>( const_cast<T*>(ptr) ); } };
template<typename T> struct __cast_void { inline void* operator()( T ptr ) { return reinterpret_cast<void*>(ptr); } };

} // namespace detail

///
/// \brief A smart pointer that retains shared ownership of an object in device memory.
///
/// ecuda::shared_ptr is a smart pointer that retains shared ownership of an object in device memory
/// through a pointer. Several shared_ptr objects may own the same object. The object is destroyed
/// and its memory deallocated when either of the following happens:
///
/// - the last remaining shared_ptr owning the object is destroyed
/// - the last remaining shared_ptr owning the object is assigned another pointer via operator= or reset().
///
/// The object is destroyed using cudaFree or a custom deleter that is supplied to shared_ptr during
/// construction.
///
/// A shared_ptr can share ownership of an object while storing a pointer to another object. This feature
/// can be used to point to member objects while owning the object they belong to. The stored pointer is the
/// one accessed by get(), the dereference and the comparison operators. The managed poitner is the one
/// passed to the deleter when use count reaches zero.
///
/// A shared_ptr may also own no objects, in which case it is called empty (an empty shared_ptr may have
/// a non-null stored pointer if the aliasing constructor was used to create it).
///
/// All specializations of shared_ptr meet the requirements of CopyConstructible, CopyAssignable, and
/// LessThanComparable and are contextually convertible to bool.
///
/// All member functions (including copy constructor and copy assignment) can be called by multiple threads
/// on different instances of shared_ptr without additional synchronization even if these instances are
/// copies and share ownership of the same object. If multiple threads of execution access the same shared_ptr
/// without synchronization and any of those accesses uses a non-const member function of shared_ptr then a
/// data race will occur; the shared_ptr overloads of atomic functions can be used to prevent the data race.
///
template<typename T>
class shared_ptr
{

public:
	typedef T element_type; //!< type of the managed object
	//typedef T* pointer; //!< type of pointer to the managed object

private:
	//pointer current_ptr;
	void* current_ptr;
	detail::sp_counter_base* counter;

	template<typename U> friend class shared_ptr;

public:
	template<typename U>
	__host__ explicit shared_ptr( U* ptr = NULL ) : current_ptr(detail::__cast_void<U*>()(ptr)) {
		counter = new detail::sp_counter_impl_p<U>();
	}

	template<typename U,class Deleter>
	__host__ shared_ptr( U* ptr, Deleter deleter ) : current_ptr(detail::__cast_void<U*>()(ptr)) {
		counter = new detail::sp_counter_impl_pd<U,Deleter>( deleter );
	}

	__host__ __device__ shared_ptr( const shared_ptr& src ) : current_ptr(src.current_ptr), counter(src.counter) {
		#ifndef __CUDA_ARCH__
		++(counter->owner_count);
		#endif
	}

	template<typename U>
	__host__ __device__ shared_ptr( const shared_ptr<U>& src ) : current_ptr(src.current_ptr), counter(src.counter) {
		#ifndef __CUDA_ARCH__
		++(counter->owner_count);
		#endif
	}

	__host__ __device__ ~shared_ptr() {
		#ifndef __CUDA_ARCH__
		--(counter->owner_count);
		if( counter->owner_count == 0 ) {
			counter->dispose( current_ptr );
			delete counter;
		}
		#endif
	}

	__host__ __device__ inline void reset() __NOEXCEPT__ { shared_ptr().swap( *this ); }
	template<typename U> __host__ __device__ inline void reset( U* ptr ) __NOEXCEPT__ { shared_ptr( ptr ).swap( *this ); }
	template<typename U,class Deleter> __host__ __device__ inline void reset( U* ptr, Deleter d ) __NOEXCEPT__ { shared_ptr( ptr, d ).swap( *this ); }

	__host__ __device__ inline void swap( shared_ptr& other ) __NOEXCEPT__ {
		::ecuda::swap( current_ptr, other.current_ptr );
		::ecuda::swap( counter, other.counter );
	}

	__host__ __device__ inline T* get() const { return reinterpret_cast<T*>(current_ptr); }

	__device__ inline typename add_lvalue_reference<T>::type operator*() const __NOEXCEPT__ { return *reinterpret_cast<T*>(current_ptr); }

	__host__ __device__ inline T* operator->() const __NOEXCEPT__ { return reinterpret_cast<T*>(current_ptr); }

	__host__ __device__ inline std::size_t use_count() const __NOEXCEPT__ { return counter->owner_count; }

	__host__ __device__ inline bool unique() const __NOEXCEPT__ { return use_count() == 1; }

	#ifdef __CPP11_SUPPORTED__
	__host__ __device__ explicit operator bool() const { return get() != NULL; }
	#else
	__host__ __device__ operator bool() const { return get() != NULL; }
	#endif

	template<typename U>
	__host__ __device__ inline bool owner_before( const shared_ptr<U>& other ) const { return counter < other.counter; }

	template<typename T2> __host__ __device__ bool operator==( const shared_ptr<T2>& other ) const { return get() == other.get(); }
	template<typename T2> __host__ __device__ bool operator!=( const shared_ptr<T2>& other ) const { return get() != other.get(); }
	template<typename T2> __host__ __device__ bool operator< ( const shared_ptr<T2>& other ) const { return get() <  other.get(); }
	template<typename T2> __host__ __device__ bool operator> ( const shared_ptr<T2>& other ) const { return get() >  other.get(); }
	template<typename T2> __host__ __device__ bool operator<=( const shared_ptr<T2>& other ) const { return get() <= other.get(); }
	template<typename T2> __host__ __device__ bool operator>=( const shared_ptr<T2>& other ) const { return get() >= other.get(); }

	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const shared_ptr& ptr ) {
		out << ptr.get();
		return out;
	}

};

} // namespace ecuda

#endif
