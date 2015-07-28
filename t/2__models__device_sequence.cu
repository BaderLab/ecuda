#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/models.hpp"
#include "../include/ecuda/type_traits.hpp"

#define THREADS 100

template<typename T,typename P>
__global__ void fill_with_consecutive_values( ecuda::__device_sequence<T,P> sequence ) {
	const std::size_t threadNum = blockIdx.x*blockDim.x+threadIdx.x;
	if( !threadNum ) {
		std::size_t index = 0;
		for( typename ecuda::__device_sequence<T,P>::iterator iter = sequence.begin(); iter != sequence.end(); ++iter, ++index ) *iter = static_cast<T>(index);
	}
}

int main( int argc, char* argv[] ) {

	// allocate memory
	ecuda::device_allocator<double> deviceAllocator;
	typedef typename ecuda::device_allocator<double>::pointer pointer_type;
	pointer_type ptr = deviceAllocator.allocate(1000);

	// create sequence
	ecuda::__device_sequence<double,pointer_type> deviceSequence( ptr, 1000 );

	// fill sequence with consecutive values
	fill_with_consecutive_values<double><<<1,1>>>( deviceSequence );
	CUDA_CHECK_ERRORS();
	CUDA_CALL( cudaDeviceSynchronize() );

	// copy sequence to host
	std::vector<double> hostSequence( deviceSequence.size() );
//	ecuda::copy( deviceSequence.begin(), deviceSequence.end(), hostSequence.begin() );

	{
		std::vector<double> correctSequence( deviceSequence.size() );
		for( std::size_t i = 0; i < correctSequence.size(); ++i ) correctSequence[i] = static_cast<double>(i);
		std::cerr << "EQUAL = " << ( hostSequence == correctSequence ? "true" : "false" ) << std::endl;
	}

	// deallocate the memory
	deviceAllocator.deallocate( ptr, 1000 );

	return 0;

}
