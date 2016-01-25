#include <iostream>
#include <vector>

#include "../include/ecuda/event.hpp"

#include "config.hpp"
#ifndef BENCHMARK_THREADS
#define BENCHMARK_THREADS 480
#endif

typedef double value_type;

template<typename T,std::size_t N>
__global__
void copyArray( const T* src, T* dest )
{
	const int t = blockIdx.x*blockDim.x+threadIdx.x;
	if( t < N ) dest[t] = src[t];
}

int main( int argc, char* argv[] )
{

	const std::size_t N = 1000000;

	ecuda::event start, stop;
	start.record();

	std::vector<value_type> hostSequence1( N );
	{
		std::size_t n = 0;
		for( typename std::vector<value_type>::iterator iter = hostSequence1.begin(); iter != hostSequence1.end(); ++iter, ++n ) *iter = static_cast<value_type>(n);
	}

	value_type *deviceSequence1, *deviceSequence2;
	CUDA_CALL( cudaMalloc( &deviceSequence1, N*sizeof(value_type) ) );
	CUDA_CALL( cudaMalloc( &deviceSequence2, N*sizeof(value_type) ) );

	CUDA_CALL( cudaMemcpy( deviceSequence1, &hostSequence1.front(), N*sizeof(value_type), cudaMemcpyHostToDevice ) );

	dim3 grid( 1, (N+BENCHMARK_THREADS-1)/BENCHMARK_THREADS ), threads( BENCHMARK_THREADS, 1 );
	CUDA_CALL_KERNEL_AND_WAIT( copyArray<value_type,N><<<grid,threads>>>( deviceSequence1, deviceSequence2 ) );

	std::vector<value_type> hostSequence2( N );
	CUDA_CALL( cudaMemcpy( &hostSequence2.front(), deviceSequence2, N*sizeof(value_type), cudaMemcpyDeviceToHost ) );

	const bool isEqual = std::equal( hostSequence2.begin(), hostSequence2.end(), hostSequence1.begin() );

	cudaFree( deviceSequence1 );
	cudaFree( deviceSequence2 );

	stop.record();
	stop.synchronize();

	if( isEqual )
		std::cout << "Test successful." << std::endl;
	else
		std::cout << "Test failed." << std::endl;

	std::cout << "Execution Time: " << (stop-start) << "ms" << std::endl;

	return EXIT_SUCCESS;

}

