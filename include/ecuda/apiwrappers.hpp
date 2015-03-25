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
// apiwrappers.hpp
//
// Wrappers around CUDA API functions.
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_APIWRAPPERS_HPP
#define ECUDA_APIWRAPPERS_HPP

#include "global.hpp"

namespace ecuda {

template<typename T>
inline cudaError_t cudaMemcpy( T* dest, const T* src, const std::size_t count, cudaMemcpyKind kind ) {
	return cudaMemcpy( reinterpret_cast<void*>(dest), reinterpret_cast<const void*>(src), sizeof(T)*count, kind );
}

template<typename T>
inline cudaError_t cudaMemcpy2D( T* dest, const std::size_t dpitch, const T* src, const std::size_t spitch, const std::size_t width, const std::size_t height, cudaMemcpyKind kind ) {
	return cudaMemcpy2D( reinterpret_cast<void*>(dest), dpitch, reinterpret_cast<const void*>(src), spitch, width*sizeof(T), height, kind );
}

} // namespace ecuda

#endif
