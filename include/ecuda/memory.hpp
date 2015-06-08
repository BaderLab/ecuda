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

#endif
