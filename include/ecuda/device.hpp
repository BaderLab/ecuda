//----------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
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
