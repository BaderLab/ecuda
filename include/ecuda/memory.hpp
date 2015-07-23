#ifndef ECUDA_DEVICE_PTR_HPP
#define ECUDA_DEVICE_PTR_HPP

#include "global.hpp"

///
/// \cond DEVELOPER_DOCUMENTATION
///
/// \section ecuda_pointers Pointer specializations
///
/// All but the most low-level memory access calls inside the ecuda API
/// are done using five pointer specialization classes: naked_ptr,
/// shared_ptr, padded_ptr, striding_ptr, and unique_ptr.
///
/// shared_ptr and unique_ptr are functionally identical to their C++11
/// STL counterparts (i.e. std::shared_ptr), but are written to utilize
/// CUDA-allocated device memory (most notably by providing deallocation
/// through cudaFree).
///
/// tbc...
///
/// \endcond
///

#include "ptr/naked_ptr.hpp"
#include "ptr/padded_ptr.hpp"
#include "ptr/shared_ptr.hpp"
#include "ptr/striding_ptr.hpp"
#include "ptr/unique_ptr.hpp"

namespace ecuda {

template<typename T> struct owner_less;

///
/// This function object provides owner-based (as opposed to value-based) mixed-type
/// ordering of ecuda::shared_ptr. The order is such that two smart pointers compare
/// equivalent only if they are both empty or if they both manage the same object,
/// even if the values of the raw pointers obtained by get() are different (e.g.
/// because they point at different subobjects within the same object).
///
/// This class template is the preferred comparison predicate when building
/// associative containers with ecuda::shared_ptr as keys, that is:
/// std::map< ecuda::shared_ptr<T>, U, ecuda::owner_less<ecuda::shared_ptr<T> > >.
///
///
template<typename T>
struct owner_less< shared_ptr<T> > {
	typedef bool result_type;
	typedef shared_ptr<T> first_argument_type;
	typedef shared_ptr<T> second_argument_type;
	///
	/// \brief Compares lhs and rhs using owner-based semantics.
	///
	/// \param lhs,rhs shared-ownership pointers to compare
	/// \return true if lhs is less than rhs as determined by the owner-based ordering
	///
	__host__ __device__ inline bool operator()( const shared_ptr<T>& lhs, const shared_ptr<T>& rhs ) const { return lhs.owner_before(rhs); }
};

} // namespace ecuda


#endif
