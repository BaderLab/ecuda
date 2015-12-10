/*
Copyright (c) 2014-2015, Scott Zuyderduyn
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
	explicit cuda_error( cudaError_t errorType ) : std::runtime_error(""), errorType(errorType) {}

	///
	/// \brief Constructor.
	/// \param errorType error code
	/// \param what_arg string describing the error
	///
	explicit cuda_error( cudaError_t errorType, const std::string& what_arg ) : std::runtime_error( what_arg ), errorType(errorType) {}

	///
	/// \brief Constructor.
	/// \param errorType error code
	/// \param what_arg string describing the error
	///
	explicit cuda_error( cudaError_t errorType, const char* what_arg ) : std::runtime_error( what_arg ), errorType(errorType) {}

	///
	/// \brief Gets the error code.
	/// \return the CUDA error code
	///
	inline cudaError_t get_error_code() const { return errorType; }

};

} // namespace ecuda

#endif
