#ifdef WIN32
#pragma warning(disable:4005)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#undef _HAS_ITERATOR_DEBUGGING
#pragma warning(default:4005)
#include <Windows.h>

//#define _ITERATOR_DEBUG_LEVEL 0
#endif

#include <iomanip>
#include <iostream>

#include "../include/ecuda/device.hpp"

std::string create_memory_string( unsigned long x );
std::string create_frequency_string( const unsigned x );

int ConvertSMVer2Cores( int major, int minor );

int main( int argc, char* argv[] )
{

	int deviceCount = ecuda::device::get_device_count();

	if( deviceCount == 1 )
		std::cout << "There is " << deviceCount << " device supporting ecuda." << std::endl;
	else
		std::cout << "There are " << deviceCount << " devices supporting ecuda." << std::endl;
	std::cout << std::endl;

	for( int i = 0; i < deviceCount; ++i ) {

		ecuda::device device( i );
		const cudaDeviceProp& prop = device.get_properties();

		int cudaCores = -1;
		try {
			// this solution for getting the number of CUDA cores from:
			// https://devtalk.nvidia.com/default/topic/470848/what-39-s-the-proper-way-to-detect-sp-cuda-cores-count-per-sm-/
			cudaCores = ConvertSMVer2Cores( prop.major, prop.minor ) * prop.multiProcessorCount;
		} catch( std::runtime_error& ) {
			std::cerr << "WARNING: number of cores for the hardware's SM version is not defined in program source code." << std::endl;
		}

		std::cout << "========================================================================" << std::endl;
		std::cout << "::Device " << i << " is a: " << prop.name << std::endl;
		std::cout << "------------------------------------------------------------------------" << std::endl;
		std::cout << "Version    :: CUDA Driver: " << device.get_driver_version_string() << " CUDA Runtime: " << device.get_runtime_version_string() << " Compute Capability: " << prop.major << "." << prop.minor << std::endl;
		std::cout << "Memory     :: Global: " << create_memory_string(prop.totalGlobalMem) << " Constant: " << create_memory_string(prop.totalConstMem) << std::endl;
		std::cout << "              Shared Per Block: " << create_memory_string(prop.sharedMemPerBlock) << " L2 Cache: " << create_memory_string(prop.l2CacheSize) << std::endl;
		std::cout << "              Bus Width: " << create_memory_string(prop.memoryBusWidth) << std::endl;
		std::cout << "Number     :: Multiprocessors: " << prop.multiProcessorCount << " Warp Size: " << prop.warpSize << " CUDA Cores: " << cudaCores << std::endl;
		std::cout << "              Maximum Threads Per Block: " << prop.maxThreadsPerBlock << " Asynchronous Engines: " << prop.asyncEngineCount << std::endl;
		std::cout << "Dimension  :: Block: [" << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << "] Grid: [" << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << "]" << std::endl;
		std::cout << "Texture    :: Alignment: " << create_memory_string(prop.textureAlignment) << " Pitch Alignment: " << create_memory_string(prop.texturePitchAlignment) << std::endl;
		std::cout << "Surface    :: Alignment: " << create_memory_string(prop.surfaceAlignment) << std::endl;
		std::cout << "Other      :: Registers Per Block: " << prop.regsPerBlock << " Maximum Memory Pitch: " << create_memory_string(prop.memPitch) << std::endl;
		std::cout << "              Concurrent kernels: " << prop.concurrentKernels << std::endl;
		std::cout << "              Maximum Threads Per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "Clock Rate :: GPU: " << create_frequency_string(prop.clockRate*1000) << " Memory: " << create_frequency_string(prop.memoryClockRate) << std::endl;
		std::cout << "Features   :: Concurrent copy and execution            [" << (prop.deviceOverlap?'Y':'N')            << "]" << std::endl;
		std::cout << "              Run time limit on kernels                [" << (prop.kernelExecTimeoutEnabled?'Y':'N') << "]" << std::endl;
		std::cout << "              Integrated                               [" << (prop.integrated?'Y':'N')               << "]" << std::endl;
		std::cout << "              Host page-locked memory                  [" << (prop.canMapHostMemory?'Y':'N')         << "]" << std::endl;
		std::cout << "              ECC enabled                              [" << (prop.ECCEnabled?'Y':'N')               << "]" << std::endl;
		std::cout << "              Shares a unified address space with host [" << (prop.unifiedAddressing?'Y':'N')        << "]" << std::endl;
		std::cout << "              Tesla device using TCC driver            [" << (prop.tccDriver?'Y':'N')                << "]" << std::endl;
		std::cout << "PCI        :: Domain: " << prop.pciDomainID << " Bus: " << prop.pciBusID << " Device: " << prop.pciDeviceID << std::endl;
		std::cout << "------------------------------------------------------------------------" << std::endl;
		std::cout << "Compute mode:" << std::endl;
		std::cout << "  Default     meaning: multiple threads can use cudaSetDevice()  [" << (prop.computeMode==cudaComputeModeDefault?'X':' ')    << "]" << std::endl;
		std::cout << "  Exclusive   meaning: only one thread can use cudaSetDevice()   [" << (prop.computeMode==cudaComputeModeExclusive?'X':' ')  << "]" << std::endl;
		std::cout << "  Prohibited  meaning: no threads can use cudaSetDevice()        [" << (prop.computeMode==cudaComputeModeProhibited?'X':' ') << "]" << std::endl;
		std::cout << "========================================================================" << std::endl;
		std::cout << std::endl;

	}

	return EXIT_SUCCESS;

}

///
/// \brief Performs a unit conversion and creates a pretty string.
///
/// If the provided value is an exact multiple of per_unit then no
/// decimal places will be displayed, regardless of the value of digits.
/// (e.g. digits=2 x=2000 per_unit=1000 unitSymbol="k" gives: 2k
///       digits=2 x=2500 per_unit=1000 unitSymbol="k" gives: 2.50k)
///
/// If x < per_unit then the function returns false so that the
/// user can retry with a smaller unit.
///
/// \param out stream to output formatted string to
/// \param digits number of decimal places
/// \param x the value to format
/// \param per_unit the value per unit (e.g. 2^30=1Gb)
/// \param unitSymbol the symbol for the unit (e.g. "Gb")
/// \return true if the string was successfully created
///
bool try_creating_unit_string( std::ostream& out, unsigned digits, unsigned long x, unsigned long per_unit, const std::string& unitSymbol = std::string() )
{
	unsigned long units = x / per_unit;
	if( !units ) return false;
	std::stringstream ss;
	if( x % per_unit ) {
		const double y = x / static_cast<double>(per_unit);
		ss << std::setprecision(digits) << std::fixed << y;
	} else {
		ss << units;
	}
	ss << unitSymbol;
	out << ss.str();
	return true;
}

std::string create_memory_string( unsigned long x )
{
	std::stringstream ss;
	if( try_creating_unit_string( ss, 1, x, 1073741824, "Gb" ) ) return ss.str();
	if( try_creating_unit_string( ss, 1, x, 1048576   , "Mb" ) ) return ss.str();
	if( try_creating_unit_string( ss, 1, x, 1024      , "kb" ) ) return ss.str();
	ss << x << "b";
	return ss.str();
}

std::string create_frequency_string( const unsigned x )
{
	std::stringstream ss;
	if( try_creating_unit_string( ss, 2, x, 1000000000, "GHz" ) ) return ss.str();
	if( try_creating_unit_string( ss, 2, x, 1000000,    "MHz" ) ) return ss.str();
	if( try_creating_unit_string( ss, 2, x, 1000,       "kHz" ) ) return ss.str();
	ss << x << "Hz";
	return ss.str();
}

int ConvertSMVer2Cores( int major, int minor )
{
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x20, 32  }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48  }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Fermi Generation (SM 3.0) GK10x class
		{ 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
		{   -1, -1  }
	};

	int index = 0;

	while( nGpuArchCoresPerSM[index].SM != -1 ) {
		if( nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) return nGpuArchCoresPerSM[index].Cores;
		++index;
	}

	std::stringstream ss;
	ss << "MapSMtoCores for SM " << major << "." << minor << " is undefined.";
	throw std::runtime_error( ss.str() );
}


/*
UNREPORTED PROPERTIES:
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
*/

