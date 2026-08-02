//----------------------------------------------------------------------------
// cuda_error.hpp
//
// Custom exception to throw CUDA API error codes.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_CUDA_ERROR_HPP
#define ECUDA_CUDA_ERROR_HPP

#include <stdexcept>
#include <string>

#include "impl/host_emulation.hpp" // gets data structure definitions when compiling host-only without nvcc

namespace ecuda {

///
/// \brief Exception for CUDA API cudaError_t errors.
///
/// See http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for a list
/// of possible error codes and their meaning.
///
class cuda_error : public std::runtime_error
{

private:
    cudaError_t errorType;

public:
    ///
    /// \brief Constructor.
    /// \param errorType error code
    ///
    explicit cuda_error(cudaError_t errorType)
      : std::runtime_error("")
      , errorType(errorType)
    {
    }

    ///
    /// \brief Constructor.
    /// \param errorType error code
    /// \param what_arg string describing the error
    ///
    explicit cuda_error(cudaError_t errorType, const std::string& what_arg)
      : std::runtime_error(what_arg)
      , errorType(errorType)
    {
    }

    ///
    /// \brief Constructor.
    /// \param errorType error code
    /// \param what_arg string describing the error
    ///
    explicit cuda_error(cudaError_t errorType, const char* what_arg)
      : std::runtime_error(what_arg)
      , errorType(errorType)
    {
    }

    ///
    /// \brief Gets the error code.
    /// \return the CUDA error code
    ///
    inline cudaError_t get_error_code() const { return errorType; }
};

} // namespace ecuda

#endif
