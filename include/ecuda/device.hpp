/*
Copyright (c) 2014, Scott Zuyderduyn
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
// device.hpp
// Holds information about a CUDA-capable device(s).
//
// Author: Scott D. Zuyderduyn, Ph.D. (scott.zuyderduyn@utoronto.ca)
//----------------------------------------------------------------------------

#pragma once
#ifndef ECUDA_DEVICE_HPP
#define ECUDA_DEVICE_HPP

#include <sstream>
#include <string>
#include "global.hpp"

namespace ecuda {

class device {

private:
	int deviceNumber;
	cudaDeviceProp deviceProperties;
	int driverVersion, runtimeVersion;

public:
	static int getDeviceCount() {
		int deviceCount = 0;
		CUDA_CALL( cudaGetDeviceCount(&deviceCount) );
		return deviceCount;
	}

public:
	device( const int deviceNumber ) : deviceNumber(deviceNumber) {
		CUDA_CALL( cudaGetDeviceProperties(&deviceProperties,deviceNumber) );
		CUDA_CALL( cudaDriverGetVersion(&driverVersion) );
		CUDA_CALL( cudaRuntimeGetVersion(&runtimeVersion) );
	}
	~device() {}

	inline int get_device_number() const { return deviceNumber; }
	inline int get_driver_version() const { return driverVersion; }
	inline int get_runtime_version() const { return runtimeVersion; }
	inline const cudaDeviceProp& get_properties() const { return deviceProperties; }

	std::string get_driver_version_string() const {
		std::ostringstream oss;
		oss << (driverVersion/1000);
		oss << ".";
		oss << ((driverVersion%100)/10);
		return oss.str();
	}

	std::string get_runtime_version_string() const {
		std::ostringstream oss;
		oss << (runtimeVersion/1000);
		oss << ".";
		oss << ((runtimeVersion%100)/10);
		return oss.str();
	}

};

} // namespace ecuda

#endif
