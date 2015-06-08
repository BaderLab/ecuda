#ifndef ECUDA_PTR_SHARED_PTR_HPP
#define ECUDA_PTR_SHARED_PTR_HPP

#include <iostream>

#include "../global.hpp"
#include "common.hpp"
#include "../algorithm.hpp"

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
	virtual void dispose( void* p ) { if( p ) cudaFree( static_cast<T*>(p) ); }
};

template<typename T,class Deleter>
struct sp_counter_impl_pd : sp_counter_base {
	Deleter deleter;
	sp_counter_impl_pd( Deleter deleter ) : sp_counter_base(), deleter(deleter) {}
	virtual void dispose( void* p ) { if( p ) deleter(p); }
	virtual void* get_deleter() { return deleter; }
};

} // namespace detail

template<typename T>
class shared_ptr
{

public:
	typedef T element_type;
	typedef T* pointer;

private:
	pointer current_ptr;
	detail::sp_counter_base* counter;

	template<typename U> friend class shared_ptr;

public:
	__host__ explicit shared_ptr( T* ptr = NULL ) : current_ptr(ptr) {
		counter = new detail::sp_counter_impl_p<T>();
	}

	template<class Deleter>
	__host__ shared_ptr( T* ptr, Deleter deleter ) : current_ptr(ptr) {
		counter = new detail::sp_counter_impl_pd<T,Deleter>( deleter );
	}

	__host__ __device__ shared_ptr( const shared_ptr& src ) : current_ptr(src.current_ptr), counter(src.counter) {
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

	__host__ __device__ inline void reset( pointer ptr = pointer() ) __NOEXCEPT__ {	shared_ptr().swap( *this );	}

	__host__ __device__ inline void swap( shared_ptr& other ) __NOEXCEPT__ {
		::ecuda::swap( current_ptr, other.current_ptr );
		::ecuda::swap( counter, other.counter );
	}

	__host__ __device__ inline pointer get() const { return current_ptr; }

	__device__ inline typename add_lvalue_reference<T>::type operator*() const __NOEXCEPT__ { return *current_ptr; }

	__host__ __device__ inline pointer operator->() const __NOEXCEPT__ { return current_ptr; }

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
