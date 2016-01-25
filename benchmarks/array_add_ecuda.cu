#include <iostream>
#include <vector>

#include "../include/ecuda/event.hpp"
#include "../include/ecuda/array.hpp"

#include "config.hpp"
#ifndef BENCHMARK_THREADS
#define BENCHMARK_THREADS 480
#endif

typedef double value_type;

template<typename T,std::size_t N>
__global__
void copyArray( typename ecuda::array<T,N>::const_kernel_argument src, typename ecuda::array<T,N>::kernel_argument dest )
{
	const int t = threadIdx.x + blockIdx.x * blockDim.x;
	if( t < src.size() ) dest[t] = src[t];
}

int main( int argc, char* argv[] )
{

	const std::size_t N = 1000000;

	ecuda::event start, stop;
	start.record();

	std::vector<value_type> hostSequence( N );
	{
		std::size_t n = 0;
		for( typename std::vector<value_type>::iterator iter = hostSequence.begin(); iter != hostSequence.end(); ++iter, ++n ) *iter = static_cast<value_type>(n);
	}

	ecuda::array<value_type,N> deviceSequence1, deviceSequence2;

	ecuda::copy( hostSequence.begin(), hostSequence.end(), deviceSequence1.begin() );

	CUDA_CALL_KERNEL_AND_WAIT( copyArray<value_type,N><<<BENCHMARK_THREADS,((N+BENCHMARK_THREADS-1)/BENCHMARK_THREADS)>>>( deviceSequence1, deviceSequence2 ) );

	const bool isEqual = ecuda::equal( deviceSequence2.begin(), deviceSequence2.end(), hostSequence.begin() );

	stop.record();
	stop.synchronize();

	if( isEqual )
		std::cout << "Test successful." << std::endl;
	else
		std::cout << "Test failed." << std::endl;

	std::cout << "Execution Time: " << (stop-start) << "ms" << std::endl;

	return EXIT_SUCCESS;

}

