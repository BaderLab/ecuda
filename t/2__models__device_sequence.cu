#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/models.hpp"
#include "../include/ecuda/type_traits.hpp"

#define THREADS 100

//#define CHECK_IF_DEVICE_SOURCE_ASSERTION_CAUSES_FAILURE
//#define CHECK_IF_DEVICE_DESTINATION_ASSERTION_CAUSES_FAILURE

template<typename T,typename P>
__global__
void fill_with_consecutive_values( ecuda::impl::device_sequence<T,P> sequence ) {
	const std::size_t threadNum = blockIdx.x*blockDim.x+threadIdx.x;
	if( !threadNum ) {
		std::size_t index = 0;
		for( typename ecuda::impl::device_sequence<T,P>::iterator iter = sequence.begin(); iter != sequence.end(); ++iter, ++index ) *iter = static_cast<T>(index);
	}
}

template<typename T,typename P>
__global__
void reverse_copy( const ecuda::impl::device_sequence<T,P> src, ecuda::impl::device_sequence<T,P> dest ) {
	const std::size_t threadNum = blockIdx.x*blockDim.x+threadIdx.x;
	if( !threadNum ) ecuda::copy( src.begin(), src.end(), dest.rbegin() );
}

int main( int argc, char* argv[] ) {

	// allocate memory
	ecuda::device_allocator<double> deviceAllocator;
	typedef typename ecuda::device_allocator<double>::pointer pointer_type;
	pointer_type ptr = deviceAllocator.allocate(1000);

	// create sequence
	ecuda::impl::device_sequence<double,pointer_type> deviceSequence( ptr, 1000 );

	// fill sequence with consecutive values
	CUDA_CALL_KERNEL_AND_WAIT( fill_with_consecutive_values<double><<<1,1>>>( deviceSequence ) );

	// copy sequence and reverse
	pointer_type ptr2 = deviceAllocator.allocate(1000);
	ecuda::impl::device_sequence<double,pointer_type> deviceSequence2( ptr2, 1000 );
	CUDA_CALL_KERNEL_AND_WAIT( reverse_copy<double><<<1,1>>>( deviceSequence, deviceSequence2 ) );

	// copy sequence to host
	std::vector<double> hostSequence( deviceSequence.size() );
	#ifdef CHECK_IF_DEVICE_SOURCE_ASSERTION_CAUSES_FAILURE
	ecuda::copy( deviceSequence.begin(), deviceSequence.end(), hostSequence.begin() ); // should fail compile-time assertion since device memory is not declared contiguous
	#endif
	#ifdef CHECK_IF_DEVICE_DESTINATION_ASSERTION_CAUSES_FAILURE
	ecuda::copy( hostSequence.begin(), hostSequence.end(), deviceSequence.begin() ); // should fail compile-time assertion since device memory is not declared contiguous
	#endif

	{
		std::vector<double> correctSequence( deviceSequence.size() );
		for( std::size_t i = 0; i < correctSequence.size(); ++i ) correctSequence[i] = static_cast<double>(i);
		std::cerr << "EQUAL = " << ( hostSequence == correctSequence ? "true" : "false" ) << std::endl;
	}

	// deallocate the memory
	deviceAllocator.deallocate( ptr, 1000 );

	return 0;

}
