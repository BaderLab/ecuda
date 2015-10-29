#include "../include/ecuda/event.hpp"
#include "../include/ecuda/models.hpp"

#define THREADS 100
#define N 100000
#define ROUNDS 100

template<typename T,typename P>
__global__
void fill_with_consecutive_values( ecuda::impl::device_sequence<T,P> sequence ) {
	const std::size_t index = blockIdx.y*blockDim.x+threadIdx.x;
	if( index < sequence.size() ) sequence[index] = index;
}

template<typename T>
__global__
void fill_with_consecutive_values( T* ptr, const std::size_t n ) {
	const std::size_t index = blockIdx.x*blockDim.x+threadIdx.x;
	if( index < n ) *(ptr+index) = index;
}

void perform_tasks_with_ecuda();
void perform_tasks_old_school();

int main( int argc, char* argv[] ) {

	{
		ecuda::event start, stop;
		start.record();
		perform_tasks_old_school();
		stop.record();
		std::cout << "EXECUTION TIME (CUDA API): " << (stop-start) << "ms" << std::endl;
	}

	{
		ecuda::event start, stop;
		start.record();
		perform_tasks_with_ecuda();
		stop.record();
		std::cout << "EXECUTION TIME (ECUDA)   : " << (stop-start) << "ms" << std::endl;
	}

	return 0;

}

/// ecuda::event start, stop;
///
/// // ... specify thread grid/blocks
///
/// start.record();
/// kernelFunction<<<grid,block>>>( ... ); // call the kernel
/// stop.record();
/// stop.synchronize(); // wait until kernel finishes executing
///
/// std::cout << "EXECUTION TIME: " << ( stop - start ) << "ms" << std::endl;

void perform_tasks_with_ecuda() {

	ecuda::device_allocator<double> deviceAllocator;
	typedef typename ecuda::device_allocator<double>::pointer video_memory_pointer;
	video_memory_pointer ptr = deviceAllocator.allocate(N);

	// create sequence
	ecuda::impl::device_sequence<double,video_memory_pointer> deviceSequence( ptr, N );

	// fill with values many times
	for( unsigned i = 0; i < ROUNDS; ++i ) {
		dim3 grid( 1, (N+THREADS-1)/THREADS ), threads( THREADS, 1 );
		CUDA_CALL_KERNEL_AND_WAIT( fill_with_consecutive_values<double><<<grid,threads>>>( deviceSequence ) );
	}

	deviceAllocator.deallocate( ptr, N );

}

void perform_tasks_old_school() {

	double* ptr;
	cudaMalloc( reinterpret_cast<void**>(&ptr), N*sizeof(double) );

	for( unsigned i = 0; i < ROUNDS; ++i ) {
		dim3 grid( 1, (N+THREADS-1)/THREADS ), threads( THREADS, 1 );
		fill_with_consecutive_values<double><<<grid,threads>>>( ptr, N );
		cudaDeviceSynchronize();
	}

	cudaFree( ptr );

}
