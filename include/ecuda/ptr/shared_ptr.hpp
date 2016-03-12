/*
Copyright (c) 2014-2016, Scott Zuyderduyn
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
// ptr/shared_ptr.hpp
//
// A managed, reference-counted pointer to device memory.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#ifndef ECUDA_PTR_SHARED_PTR_HPP
#define ECUDA_PTR_SHARED_PTR_HPP

#include <iostream>

#include "../global.hpp"
#include "common.hpp"
#include "../algorithm.hpp"

//
// Implementation is the approach used by boost::shared_ptr.
//
// \todo STL contains means to use a custom allocator to allocate memory for the internal counter
//       but this hasn't been done here (things are overengineered enough atm!) - if a programmer
//       were to take the approach of allocating a lot of individual shared_ptr managing many
//       different objects, there is a potential performance benefit from allowing a task-optimized
//       custom allocator
//

namespace ecuda {

/// \cond DEVELOPER_DOCUMENTATION
namespace detail {

struct sp_counter_base
{
	std::size_t owner_count;
	sp_counter_base() : owner_count(1) {}
	virtual ~sp_counter_base() {}
	virtual void dispose( void* ) = 0;
	virtual void* get_deleter() { return 0; }
};

template<typename T>
struct sp_counter_impl_p : sp_counter_base
{
	sp_counter_impl_p() : sp_counter_base() {}
	virtual void dispose( void* p ) { if( p ) ecuda::default_device_delete<void>()(p); } // cudaFree( p );
};

template<typename T,class Deleter>
struct sp_counter_impl_pd : sp_counter_base
{
	Deleter deleter;
	sp_counter_impl_pd( Deleter deleter ) : sp_counter_base(), deleter(deleter) {}
	virtual void dispose( void* p ) { if( p ) deleter(p); }
	//virtual void* get_deleter() { return deleter; }
	virtual void* get_deleter() { return reinterpret_cast<void*>(&deleter); }
};

} // namespace detail
/// \endcond

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
/// The object is destroyed using ecuda::default_device_delete or a custom deleter that is supplied to shared_ptr
/// during construction.
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

private:
	void* current_ptr; //!< the stored pointer to the object itself is anonymous (sp_counter_base relies on a void* pointer for dispose)
	detail::sp_counter_base* counter; //!< pointer to shared counting/disposal structure (is NULL iff current_ptr is NULL)

	template<typename U> friend class shared_ptr;

public:

	///
	/// \brief Default constructor constructs a shared_ptr with no managed object.
	///
	__HOST__ __DEVICE__ ECUDA__CONSTEXPR shared_ptr() ECUDA__NOEXCEPT : current_ptr(NULL), counter(NULL) {}

	///
	/// \brief Constructs a shared_ptr with a pointer to the managed object.
	///
	/// U must be a complete type and ptr must be convertible to T*.
	/// Additionally, uses ecuda::default_device_delete as the deleter.
	///
	/// \param ptr a pointer to an object to manage
	///
	template<typename U>
	__HOST__ __DEVICE__ explicit shared_ptr( U* ptr ) : current_ptr(detail::void_cast<T*>()(ptr)), counter(NULL)
	{
		#ifndef __CUDA_ARCH__
		if( ptr ) counter = new detail::sp_counter_impl_p<T>();
		#endif
	}

	///
	/// \brief Constructs a shared_ptr with a pointer to the managed object.
	///
	/// U must be a complete type and ptr must be convertible to T*.
	/// Additionally, uses ecuda::default_device_delete as the deleter.
	///
	/// \param ptr a pointer to an object to manage
	/// \param deleter a deleter to use to destroy the object
	///
	template<typename U,class Deleter>
	__HOST__ __DEVICE__ shared_ptr( U* ptr, Deleter deleter ) : current_ptr(detail::void_cast<T*>()(ptr)), counter(NULL)
	{
		#ifndef __CUDA_ARCH__
		if( ptr ) counter = new detail::sp_counter_impl_pd<T,Deleter>( deleter );
		#endif
	}

	///
	/// \brief The aliasing constructor.
	///
	/// Constructs a shared_ptr which shares ownership information with src, but
	/// holds an unrelated and unmanaged pointer ptr. Even if this shared_ptr is
	/// the last of the group to go out of scope, it will call the destructor
	/// for the object originally managed by src. However, calling get() on this
	/// will always return a copy of ptr. It is the responsibility of the
	/// programmer to make sure that this ptr remains valid as long as this
	/// shared_ptr exists, such as in the typical use cases where ptr is a member
	/// of the object managed by src or is an alias (e.g. downcast) of src.get().
	///
	/// \param src another smart pointer to share ownership to or acquire the ownership from
	/// \param ptr a pointer to an object to manage
	///
	template<typename U>
	__HOST__ __DEVICE__ shared_ptr( const shared_ptr<U>& src, T* ptr ) ECUDA__NOEXCEPT : current_ptr(ptr), counter(src.counter)
	{
		#ifndef __CUDA_ARCH__
		++(counter->owner_count);
		#endif
	}

	///
	/// \brief Copy constructor.
	///
	/// Constructs a shared_ptr which shares ownership of the object managed by src.
	/// If src manages no object, *this manages no object too.
	///
	/// \param src another smart pointer to share ownership to or acquire the ownership from
	///
	__HOST__ __DEVICE__ shared_ptr( const shared_ptr& src ) ECUDA__NOEXCEPT : current_ptr(src.current_ptr), counter(src.counter)
	{
		#ifndef __CUDA_ARCH__
		if( counter ) ++(counter->owner_count);
		#endif
	}

	///
	/// \brief Copy constructor.
	///
	/// Constructs a shared_ptr which shares ownership of the object managed by src.
	/// If src manages no object, *this manages no object too. This overload won't
	/// participate in overload resolution if U* is not implicitly convertible to
	/// T*.
	///
	/// \param src another smart pointer to share ownership to or acquire the ownership from
	///
	template<typename U>
	__HOST__ __DEVICE__ shared_ptr( const shared_ptr<U>& src ) ECUDA__NOEXCEPT : current_ptr(src.current_ptr), counter(src.counter)
	{
		#ifndef __CUDA_ARCH__
		if( counter ) ++(counter->owner_count);
		#endif
	}

	#ifdef ECUDA_CPP11_AVAILABLE
	///
	/// \brief Move constructor.
	///
	/// Move-constructs a shared_ptr from src. After the construction, *this contains
	/// a copy of the previous state of src, src is empty.
	///
	/// \param src another smart pointer to share ownership to or acquire the ownership from
	///
	__HOST__ __DEVICE__ shared_ptr( shared_ptr&& src ) ECUDA__NOEXCEPT : current_ptr(std::move(src.current_ptr)), counter(std::move(src.counter))
	{
		//src.current_ptr = NULL;
		//src.counter = NULL;
	}

	///
	/// \brief Move constructor.
	///
	/// Move-constructs a shared_ptr from src. After the construction, *this contains
	/// a copy of the previous state of src, src is empty. This overload doesn't
	/// participate in overload resolution if U* is not implicitly convertible to T*.
	///
	/// \param src another smart pointer to share ownership to or acquire the ownership from
	///
	template<typename U>
	__HOST__ __DEVICE__ shared_ptr( shared_ptr<U>&& src ) ECUDA__NOEXCEPT : current_ptr(std::move(src.current_ptr)), counter(std::move(src.counter))
	{
		//src.current_ptr = NULL;
		//src.counter = NULL;
	}

	///
	/// \brief Takes over management of object owned by a unique_ptr.
	///
	/// Constructs a shared_ptr which manages the object currently managed by src.
	/// The deleter associated to src is stored for future deletion of the
	/// managed object. src manages no object after the call.
	///
	/// \param src another smart pointer to share ownership to or acquire the ownership from
	///
	template<typename U,class Deleter>
	__HOST__ __DEVICE__ shared_ptr( ecuda::unique_ptr<U,Deleter>&& src ) : current_ptr(detail::void_cast<T*>()(src.release()))
	{
	   counter = current_ptr ? new detail::sp_counter_impl_pd<T,Deleter>( src.get_deleter() ) : NULL;
	}
	#endif

	///
	/// \brief Destructor.
	///
	/// If *this owns an object and it is the last shared_ptr owning it, the object
	/// is destroyed through the owned deleter. After the destruction, the smart
	/// pointers that shared ownership with *this, if any, will report a use_count()
	/// that is one less than the previous value.
	///
	__HOST__ __DEVICE__ ~shared_ptr()
	{
		#ifndef __CUDA_ARCH__
		if( counter ) {
			--(counter->owner_count);
			if( counter->owner_count == 0 ) {
				counter->dispose( current_ptr );
				delete counter;
				counter = NULL;
			}
		}
		#endif
	}

	///
	/// \brief Replaces the managed object.
	///
	/// Shares ownership of the object managed by src. If src manages no object, *this
	/// manages no object too.
	///
	/// \param src another smart pointer to share ownership to or acquire the ownership from
	///
	__HOST__ __DEVICE__ inline shared_ptr& operator=( const shared_ptr& src ) ECUDA__NOEXCEPT
	{
		shared_ptr(src).swap(*this);
		return *this;
	}

	///
	/// \brief Replaces the managed object.
	///
	/// Shares ownership of the object managed by src. If src manages no object, *this
	/// manages no object too. This overload doesn't participate in overload resolution
	/// if U* is not implicitly convertible to T*.
	///
	/// \param src another smart pointer to share ownership to or acquire the ownership from
	///
	template<typename U> __HOST__ __DEVICE__ inline shared_ptr& operator=( const shared_ptr<U>& src ) ECUDA__NOEXCEPT { shared_ptr(src).swap(*this); return *this; }

	#ifdef ECUDA_CPP11_AVAILABLE
	///
	/// \brief Replaces the managed object.
	///
	/// Move-assigns a shared_ptr from src. After the assignment, *this contains a copy
	/// of the previous state of src, src is empty.
	///
	/// \param src another smart pointer to share ownership to or acquire the ownership from
	///
	__HOST__ __DEVICE__ inline shared_ptr& operator=( shared_ptr&& src ) ECUDA__NOEXCEPT { shared_ptr(move(src)).swap(*this); return *this; }

	///
	/// \brief Replaces the managed object.
	///
	/// Move-assigns a shared_ptr from src. After the assignment, *this contains a copy
	/// of the previous state of src, src is empty. This overload doesn't participate
	/// in overload resoution if U* is not implicitly convertible to T*.
	///
	/// \param src another smart pointer to share ownership to or acquire the ownership from
	///
	template<typename U> __HOST__ __DEVICE__ inline shared_ptr& operator=( shared_ptr<U>&& src ) ECUDA__NOEXCEPT { shared_ptr(move(src)).swap(*this); return *this; }

	///
	/// \brief Transfers management of object owned by a unique_ptr.
	///
	/// Transfers the ownership of the object managed by src to *this. The deleter
	/// associated to src is stored for future deletion of the managed object.
	/// src manages no object after the call.
	///
	/// \param src another smart pointer to share ownership to or acquire the ownership from
	///
	template<typename U,class Deleter> __HOST__ __DEVICE__ inline shared_ptr& operator=( unique_ptr<U,Deleter>&& src ) ECUDA__NOEXCEPT { shared_ptr(move(src)).swap(*this); return *this; }
	#endif

	///
	/// \brief Releases ownership of the managed object.
	///
	/// After the call, *this manages no object.
	///
	__HOST__ __DEVICE__ inline void reset() ECUDA__NOEXCEPT { shared_ptr().swap( *this ); }

	///
	/// \brief Replaces the managed object with another.
	///
	/// Replaces the managed object with an object pointed to by ptr. U must be a complete
	/// type and implicitly convertible to T. Additionally, uses ecuda::default_device_delete
	/// as the deleter.
	///
	/// \param ptr pointer to an object to acquire ownership of
	///
	template<typename U> __HOST__ __DEVICE__ inline void reset( U* ptr ) { shared_ptr( ptr ).swap( *this ); }

	///
	/// \brief Replaces the managed object with another.
	///
	/// Replaces the managed object with an object pointed to by ptr. U must be a complete
	/// type and implicitly convertible to T. Additionally, uses the specified deleter as
	/// the deleter.
	///
	/// \param ptr pointer to an object to acquire ownership of
	/// \param d deleter to store for deletion of the object
	///
	template<typename U,class Deleter> __HOST__ __DEVICE__ inline void reset( U* ptr, Deleter d ) { shared_ptr( ptr, d ).swap( *this ); }

	///
	/// \brief Exchanges the contents of *this and other.
	///
	/// \param other smart pointer to exchange the contents with
	///
	__HOST__ __DEVICE__ inline void swap( shared_ptr& other ) ECUDA__NOEXCEPT
	{
		::ecuda::swap( current_ptr, other.current_ptr );
		::ecuda::swap( counter, other.counter );
	}

	///
	/// \brief Returns a pointer to the managed object.
	///
	/// Returns a null pointer if no object is being managed.
	///
	/// \return a pointer to the managed object
	///
	__HOST__ __DEVICE__ inline T* get() const ECUDA__NOEXCEPT { return reinterpret_cast<T*>(current_ptr); }

	///
	/// \brief Dereferences pointer to the managed object.
	///
	/// This is only callable from the device, since objects in device memory are only
	/// accessible from device code.
	///
	/// \return reference to the managed object
	///
	__DEVICE__ inline typename ecuda::add_lvalue_reference<T>::type operator*() const ECUDA__NOEXCEPT { return *reinterpret_cast<T*>(current_ptr); }

	///
	/// \brief Dereferences pointer to the managed object.
	///
	/// \return pointer to the managed object
	///
	__HOST__ __DEVICE__ inline T* operator->() const ECUDA__NOEXCEPT { return reinterpret_cast<T*>(current_ptr); }

	///
	/// \brief Returns the number of smart pointers managing the current object.
	///
	/// Returns the number of different shared_ptr instances (this included) managing
	/// the current object. If there is no managed object, 0 is returned.
	///
	/// \return the number of shared_ptr instances managing the current object or 0 if there is no managed object
	///
	__HOST__ __DEVICE__ inline std::size_t use_count() const ECUDA__NOEXCEPT { return counter ? counter->owner_count : 0; }

	///
	/// \brief Checks if this is the only shared_ptr instance managing the current object.
	///
	/// i.e. whether use_count() == 1.
	///
	/// \return true if *this is the only shared_ptr instance managing the current object, false otherwise
	///
	__HOST__ __DEVICE__ inline bool unique() const ECUDA__NOEXCEPT { return use_count() == 1; }

	#ifdef ECUDA_CPP11_AVAILABLE
	///
	/// \brief Checks if this stores a non-null pointer.
	///
	/// \return true if *this stores a pointer, false otherwise.
	///
	__HOST__ __DEVICE__ explicit operator bool() const ECUDA__NOEXCEPT { return get() != NULL; }
	#else
	///
	/// \brief Checks if this stores a non-null pointer.
	///
	/// \return true if *this stores a pointer, false otherwise.
	///
	__HOST__ __DEVICE__ operator bool() const ECUDA__NOEXCEPT { return get() != NULL; }
	#endif

	///
	/// Checks whether this shared_ptr precedes other in implementation defined owner-based
	/// (as opposed to value-based) order. The order is such that two smart pointers compare
	/// equivalent only if they are both empty or if they both own the same object, even
	/// if the values of the pointers obtained by get() are different (e.g. because they
	/// point at different subobjects within the same object).
	///
	/// The ordering is used to make shared pointers usable as keys in associative
	/// containers, typically through ecuda::owner_less.
	///
	/// \return true if *this precedes other, false otherwise.
	///
	template<typename U>
	__HOST__ __DEVICE__ inline bool owner_before( const shared_ptr<U>& other ) const { return counter < other.counter; }

	template<typename T2> __HOST__ __DEVICE__ bool operator==( const shared_ptr<T2>& other ) const ECUDA__NOEXCEPT { return get() == other.get(); }
	template<typename T2> __HOST__ __DEVICE__ bool operator!=( const shared_ptr<T2>& other ) const ECUDA__NOEXCEPT { return get() != other.get(); }
	template<typename T2> __HOST__ __DEVICE__ bool operator< ( const shared_ptr<T2>& other ) const ECUDA__NOEXCEPT { return get() <  other.get(); }
	template<typename T2> __HOST__ __DEVICE__ bool operator> ( const shared_ptr<T2>& other ) const ECUDA__NOEXCEPT { return get() >  other.get(); }
	template<typename T2> __HOST__ __DEVICE__ bool operator<=( const shared_ptr<T2>& other ) const ECUDA__NOEXCEPT { return get() <= other.get(); }
	template<typename T2> __HOST__ __DEVICE__ bool operator>=( const shared_ptr<T2>& other ) const ECUDA__NOEXCEPT { return get() >= other.get(); }

	///
	/// \brief Inserts a shared_ptr into a std::basic_ostream.
	///
	/// Equivalent to out << ptr.get().
	///
	/// \param out a std::basic_ostream to insert ptr into
	/// \param ptr the data to be inserted into os
	/// \return out
	///
	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const shared_ptr& ptr )
	{
		out << ptr.get();
		return out;
	}

};

} // namespace ecuda

#endif
