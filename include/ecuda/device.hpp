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

///
/// \brief Encapsulates CUDA API device information functions.
///
class device {

private:
	int deviceNumber;
	cudaDeviceProp deviceProperties;
	int driverVersion, runtimeVersion;

public:
<<<<<<< HEAD
	static int getDeviceCount() {
=======
	static int getDeviceCount()
	{
>>>>>>> ecuda2/master
		int deviceCount = 0;
		CUDA_CALL( cudaGetDeviceCount(&deviceCount) );
		return deviceCount;
	}

public:
<<<<<<< HEAD
	device( const int deviceNumber ) : deviceNumber(deviceNumber) {
=======
	device( const int deviceNumber ) : deviceNumber(deviceNumber)
	{
>>>>>>> ecuda2/master
		CUDA_CALL( cudaGetDeviceProperties(&deviceProperties,deviceNumber) );
		CUDA_CALL( cudaDriverGetVersion(&driverVersion) );
		CUDA_CALL( cudaRuntimeGetVersion(&runtimeVersion) );
	}
	~device() {}

	inline int get_device_number() const { return deviceNumber; }
	inline int get_driver_version() const { return driverVersion; }
	inline int get_runtime_version() const { return runtimeVersion; }
	inline const cudaDeviceProp& get_properties() const { return deviceProperties; }

<<<<<<< HEAD
	std::string get_driver_version_string() const {
=======
	std::string get_driver_version_string() const
	{
>>>>>>> ecuda2/master
		std::ostringstream oss;
		oss << (driverVersion/1000);
		oss << ".";
		oss << ((driverVersion%100)/10);
		return oss.str();
	}

<<<<<<< HEAD
	std::string get_runtime_version_string() const {
=======
	std::string get_runtime_version_string() const
	{
>>>>>>> ecuda2/master
		std::ostringstream oss;
		oss << (runtimeVersion/1000);
		oss << ".";
		oss << ((runtimeVersion%100)/10);
		return oss.str();
	}

<<<<<<< HEAD
	void print_summary( std::ostream& output = std::cout ) {
=======
	void print_summary( std::ostream& output = std::cout )
	{
		/*
		output << "Device " << get_device_number() << ": \"" << deviceProperties.name << "\"" << std::endl;
		output << "  CUDA Driver Version / Runtime Version          " << get_driver_version_string() << std::endl;
		output << "  CUDA Capability Major/Minor version number:    " << deviceProperties.major << "."<< deviceProperties.minor << std::endl;
		output << "  Total amount of global memory:                 " << (deviceProperties.totalGlobalMem/1048576.0f) << " (" << deviceProperties.totalGlobalMem << " bytes)" << std::endl;
		output << "  (" << std::setw(2) << deviceProperties.multiProcessorCount << ") Multiprocessors, (" << std::setw(3) <<
		*/
		#ifdef __CUDACC__
>>>>>>> ecuda2/master
		output << "name=" << deviceProperties.name << std::endl;
		output << "totalGlobalMem=" << deviceProperties.totalGlobalMem << std::endl;
		output << "sharedMemPerBlock=" << deviceProperties.sharedMemPerBlock << std::endl;
		output << "regsPerBlock=" << deviceProperties.regsPerBlock << std::endl;
		output << "warpSize=" << deviceProperties.warpSize << std::endl;
		output << "memPitch=" << deviceProperties.memPitch << std::endl;
		output << "maxThreadsPerBlock=" << deviceProperties.maxThreadsPerBlock << std::endl;
		output << "maxThreadsDim=" << deviceProperties.maxThreadsDim[0] << "," << deviceProperties.maxThreadsDim[1] << "," << deviceProperties.maxThreadsDim[2] << std::endl;
		output << "maxGridSize=" << deviceProperties.maxGridSize[0] << "," << deviceProperties.maxGridSize[1] << "," << deviceProperties.maxGridSize[2] << std::endl;
		output << "clockRate=" << deviceProperties.clockRate << std::endl;
		output << "totalConstMem=" << deviceProperties.totalConstMem << std::endl;
		output << "major=" << deviceProperties.major << std::endl;
		output << "minor=" << deviceProperties.minor << std::endl;
		output << "textureAlignment=" << deviceProperties.textureAlignment << std::endl;
		output << "texturePitchAlignment=" << deviceProperties.texturePitchAlignment << std::endl;
		output << "deviceOverlap=" << deviceProperties.deviceOverlap << std::endl;
		output << "multiProcessorCount=" << deviceProperties.multiProcessorCount << std::endl;
		output << "kernelExecTimeoutEnabled=" << deviceProperties.kernelExecTimeoutEnabled << std::endl;
		output << "integrated=" << deviceProperties.integrated << std::endl;
		output << "canMapHostMemory=" << deviceProperties.canMapHostMemory << std::endl;
		output << "computeMode=" << deviceProperties.computeMode << std::endl;
		output << "maxTexture1D=" << deviceProperties.maxTexture1D << std::endl;
		output << "maxTexture1DLinear=" << deviceProperties.maxTexture1DLinear << std::endl;
		output << "maxTexture2D=" << deviceProperties.maxTexture2D[0] << "," << deviceProperties.maxTexture2D[1] << std::endl;
		output << "maxTexture2DLinear=" << deviceProperties.maxTexture2DLinear[0] << "," << deviceProperties.maxTexture2DLinear[1] << "," << deviceProperties.maxTexture2DLinear[2] << std::endl;
		output << "maxTexture2DGather=" << deviceProperties.maxTexture2DGather[0] << "," << deviceProperties.maxTexture2DGather[1] << std::endl;
		output << "maxTexture3D=" << deviceProperties.maxTexture3D[0] << "," << deviceProperties.maxTexture3D[1] << "," << deviceProperties.maxTexture3D[2] << std::endl;
		output << "maxTextureCubemap=" << deviceProperties.maxTextureCubemap << std::endl;
		output << "maxTexture1DLayered=" << deviceProperties.maxTexture1DLayered[0] << "," << deviceProperties.maxTexture1DLayered[1] << std::endl;
		output << "maxTexture2DLayered=" << deviceProperties.maxTexture2DLayered[0] << "," << deviceProperties.maxTexture2DLayered[1] << "," << deviceProperties.maxTexture2DLayered[2] << std::endl;
		output << "maxTextureCubemapLayered=" << deviceProperties.maxTextureCubemapLayered[0] << "," << deviceProperties.maxTextureCubemapLayered[1] << std::endl;
		output << "maxSurface1D=" << deviceProperties.maxSurface1D << std::endl;
		output << "maxSurface2D=" << deviceProperties.maxSurface2D[0] << "," << deviceProperties.maxSurface2D[1] << std::endl;
		output << "maxSurface3D=" << deviceProperties.maxSurface3D[0] << "," << deviceProperties.maxSurface3D[1] << "," << deviceProperties.maxSurface3D[2] << std::endl;
		output << "maxSurface1DLayered=" << deviceProperties.maxSurface1DLayered[0] << "," << deviceProperties.maxSurface1DLayered[1] << std::endl;
		output << "maxSurface2DLayered=" << deviceProperties.maxSurface2DLayered[0] << "," << deviceProperties.maxSurface2DLayered[1] << "," << deviceProperties.maxSurface2DLayered[2] << std::endl;
		output << "maxSurfaceCubemap=" << deviceProperties.maxSurfaceCubemap << std::endl;
		output << "maxSurfaceCubemapLayered=" << deviceProperties.maxSurfaceCubemapLayered[0] << "," << deviceProperties.maxSurfaceCubemapLayered[1] << std::endl;
		output << "surfaceAlignment=" << deviceProperties.surfaceAlignment << std::endl;
		output << "concurrentKernels=" << deviceProperties.concurrentKernels << std::endl;
		output << "ECCEnabled=" << deviceProperties.ECCEnabled << std::endl;
		output << "pciBusID=" << deviceProperties.pciBusID << std::endl;
		output << "pciDeviceID=" << deviceProperties.pciDeviceID << std::endl;
		output << "pciDomainID=" << deviceProperties.pciDomainID << std::endl;
		output << "tccDriver=" << deviceProperties.tccDriver << std::endl;
		output << "asyncEngineCount=" << deviceProperties.asyncEngineCount << std::endl;
		output << "unifiedAddressing=" << deviceProperties.unifiedAddressing << std::endl;
		output << "memoryClockRate=" << deviceProperties.memoryClockRate << std::endl;
		output << "memoryBusWidth=" << deviceProperties.memoryBusWidth << std::endl;
		output << "l2CacheSize=" << deviceProperties.l2CacheSize << std::endl;
		output << "maxThreadsPerMultiProcessor=" << deviceProperties.maxThreadsPerMultiProcessor << std::endl;
<<<<<<< HEAD
=======
		#else
		output << "Not using device, in ecuda host emulation mode." << std::endl;
		#endif

>>>>>>> ecuda2/master
	}

};

} // namespace ecuda

#endif
