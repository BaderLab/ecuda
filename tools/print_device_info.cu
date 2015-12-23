#include <getopt.h>
#include <iomanip>
#include <iostream>
#include "../include/ecuda/device.hpp"

std::string create_memory_string( unsigned long x );
std::string create_frequency_string( const unsigned x );

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

		std::cout << "========================================================================" << std::endl;
		std::cout << "::Device " << i << " is a: " << prop.name << std::endl;
		std::cout << "------------------------------------------------------------------------" << std::endl;
		std::cout << "Version    :: CUDA Driver: " << device.get_driver_version_string() << " CUDA Runtime: " << device.get_runtime_version_string() << " Compute Capability: " << prop.major << "." << prop.minor << std::endl;
		std::cout << "Memory     :: Global: " << create_memory_string(prop.totalGlobalMem) << " Constant: " << create_memory_string(prop.totalConstMem) << std::endl;
		std::cout << "              Shared Per Block: " << create_memory_string(prop.sharedMemPerBlock) << " L2 Cache: " << create_memory_string(prop.l2CacheSize) << std::endl;
		std::cout << "Number     :: Multiprocessors: " << prop.multiProcessorCount << " Warp Size: " << prop.warpSize << " (=Cores: " << (prop.warpSize*prop.multiProcessorCount) << ")" << std::endl;
		std::cout << "              Maximum Threads Per Block: " << prop.maxThreadsPerBlock << " Asynchronous Engines: " << prop.asyncEngineCount << std::endl;
		std::cout << "Dimension  :: Block: [" << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << "] Grid: [" << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << "]" << std::endl;
		std::cout << "Texture    :: Alignment: " << create_memory_string(prop.textureAlignment) << " Pitch Alignment: " << create_memory_string(prop.texturePitchAlignment) << std::endl;
		std::cout << "Surface    :: Alignment: " << create_memory_string(prop.surfaceAlignment) << std::endl;
		std::cout << "Other      :: Registers Per Block: " << prop.regsPerBlock << " Maximum Memory Pitch: " << create_memory_string(prop.memPitch) << std::endl;
		std::cout << "Clock Rate :: GPU: " << create_frequency_string(prop.clockRate*1000) << " Memory: " << create_frequency_string(prop.memoryClockRate) << std::endl;
		std::cout << "Features   :: Concurrent copy and execution            [" << (prop.deviceOverlap?'Y':'N')            << "]" << std::endl;
		std::cout << "              Run time limit on kernels                [" << (prop.kernelExecTimeoutEnabled?'Y':'N') << "]" << std::endl;
		std::cout << "              Integrated                               [" << (prop.integrated?'Y':'N')               << "]" << std::endl;
		std::cout << "              Host page-locked memory                  [" << (prop.canMapHostMemory?'Y':'N')         << "]" << std::endl;
		std::cout << "              ECC enabled                              [" << (prop.ECCEnabled?'Y':'N')               << "]" << std::endl;
		std::cout << "              Shares a unified address space with host [" << (prop.unifiedAddressing?'Y':'N')        << "]" << std::endl;
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

unsigned exp10( unsigned n )
{
	unsigned y = 1;
	for( unsigned i = 0; i < n; ++i ) y *= 10;
	return y;
}

unsigned log2( unsigned n )
{
	unsigned y = 1;
	for( unsigned i = 0; i < n; ++i ) y *= 2;
	return y;
}

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
	if( try_creating_unit_string( ss, 1, x, 1024      , "Kb" ) ) return ss.str();
	if( try_creating_unit_string( ss, 1, x, 1073741824, "Gb" ) ) return ss.str();
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

/*


Device <<i>> | <<name>>
=======================================================================
Versions :: CUDA Driver: get_driver_version_string() CUDA Runtime: get_runtime_version_string() Compute Capability: <<major>>.<<minor>>
Memory   :: Global: create_memory_string(<<totalGlobalMem>> Constant: create_memory_string(<<totalConstMem>>) Shared Per Block: create_memory_string(<<sharedMemPerBlock>>) L2 Cache: create_memory_string(<<l2CacheSize>>)
Number   :: Multiprocessors: <<multiProcessorCount>> Warp Size: <<warpSize>> (=Cores: (<<warpSize>>*<<multiProcessorCount>>)) Maximum Threads Per Block: <<maxThreadsPerBlock>> Asynchronous Engines: <<asyncEngineCount>>
Dimension:: Block: [<<maxThreadsDim[0]>> x <<maxThreadsDim[1]>> x <<maxThreadsDim[2]>>] Grid: [<<maxGridSize[0]>> x <<maxGridSize[1]>> x <<maxGridSize[2]>>]
Texture  :: Alignment: create_memory_string(<<textureAlignment>>) Pitch Alignment: create_memory_string(<<texturePitchAlignment>>)
Surface  :: Alignment: create_memory_string(<<surfaceAlignment>>)
Other    :: Registers Per Block: <<regsPerBlock>> Maximum Memory Pitch: <<memPitch>>
Clock    :: GPU: create_frequency_string(<<clockRate*1000>>>) Memory: create_frequency_string(<<memoryClockRate>>)
Features :: Concurrent copy and execution [<<deviceOverlap?'Y':'N'>>]
            Run time limit on kernels     [<<kernelExecTimeoutEnabled?'Y':'N'>>]
            Integrated                    [<<integrated?'Y':'N'>>]
            Host page-locked memory       [<<canMapHostMemory?'Y':'N'>>]
            ECC enabled                   [<<ECCEnabled?'Y':'N'>>]
            Shares a unified address space with host [<<unifiedAddressing?'Y':'N'>>]
=======================================================================
Compute mode:
  Default     meaning: multiple threads can use cudaSetDevice()  [(<<computeMode>>==cudaComputeModeDefault?'X':' ')]
  Exclusive   meaning: only one thread can use cudaSetDevice()   [(<<computeMode>>==cudaComputeModeExclusive?'X':' ')]
  Prohibited  meaning: no threads can use cudaSetDevice()        [(<<computeMode>>==cudaComputeModeProhibited?'X':' ')]


Default (multiple host threads can use this device simultaneously)

 */


/*
output << "Device " << get_device_number() << ": \"" << deviceProperties.name << "\"" << std::endl;
output << "  CUDA Driver Version / Runtime Version          " << get_driver_version_string() << std::endl;
output << "  CUDA Capability Major/Minor version number:    " << deviceProperties.major << "."<< deviceProperties.minor << std::endl;
output << "  Total amount of global memory:                 " << (deviceProperties.totalGlobalMem/1048576.0f) << " (" << deviceProperties.totalGlobalMem << " bytes)" << std::endl;
output << "  (" << std::setw(2) << deviceProperties.multiProcessorCount << ") Multiprocessors, (" << std::setw(3) <<
*/
/*
#ifdef __CUDACC__
output << "name=" << deviceProperties.name << std::endl;
output << "major=" << deviceProperties.major << std::endl;
output << "minor=" << deviceProperties.minor << std::endl;

output << "totalGlobalMem=" << deviceProperties.totalGlobalMem << std::endl;
output << "totalConstMem=" << deviceProperties.totalConstMem << std::endl;
output << "sharedMemPerBlock=" << deviceProperties.sharedMemPerBlock << std::endl;
output << "l2CacheSize=" << deviceProperties.l2CacheSize << std::endl;

output << "multiProcessorCount=" << deviceProperties.multiProcessorCount << std::endl;
output << "warpSize=" << deviceProperties.warpSize << std::endl;
output << "maxThreadsPerBlock=" << deviceProperties.maxThreadsPerBlock << std::endl;
output << "asyncEngineCount=" << deviceProperties.asyncEngineCount << std::endl;

output << "maxThreadsDim=" << deviceProperties.maxThreadsDim[0] << "," << deviceProperties.maxThreadsDim[1] << "," << deviceProperties.maxThreadsDim[2] << std::endl;
output << "maxGridSize=" << deviceProperties.maxGridSize[0] << "," << deviceProperties.maxGridSize[1] << "," << deviceProperties.maxGridSize[2] << std::endl;

output << "textureAlignment=" << deviceProperties.textureAlignment << std::endl;
output << "texturePitchAlignment=" << deviceProperties.texturePitchAlignment << std::endl;

output << "surfaceAlignment=" << deviceProperties.surfaceAlignment << std::endl;

output << "regsPerBlock=" << deviceProperties.regsPerBlock << std::endl;
output << "memPitch=" << deviceProperties.memPitch << std::endl;

output << "clockRate=" << deviceProperties.clockRate << std::endl;
output << "memoryClockRate=" << deviceProperties.memoryClockRate << std::endl;

output << "deviceOverlap=" << deviceProperties.deviceOverlap << std::endl;
output << "kernelExecTimeoutEnabled=" << deviceProperties.kernelExecTimeoutEnabled << std::endl;
output << "integrated=" << deviceProperties.integrated << std::endl;
output << "canMapHostMemory=" << deviceProperties.canMapHostMemory << std::endl;
output << "ECCEnabled=" << deviceProperties.ECCEnabled << std::endl;
output << "unifiedAddressing=" << deviceProperties.unifiedAddressing << std::endl;

output << "computeMode=" << deviceProperties.computeMode << std::endl;




output << "concurrentKernels=" << deviceProperties.concurrentKernels << std::endl;
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
output << "pciBusID=" << deviceProperties.pciBusID << std::endl;
output << "pciDeviceID=" << deviceProperties.pciDeviceID << std::endl;
output << "pciDomainID=" << deviceProperties.pciDomainID << std::endl;
output << "tccDriver=" << deviceProperties.tccDriver << std::endl;
output << "memoryBusWidth=" << deviceProperties.memoryBusWidth << std::endl;
output << "maxThreadsPerMultiProcessor=" << deviceProperties.maxThreadsPerMultiProcessor << std::endl;
#else
output << "Not using device, in ecuda host emulation mode." << std::endl;
#endif
*/

