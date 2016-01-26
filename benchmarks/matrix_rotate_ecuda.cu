#include <algorithm>
#include <iostream>
#include <vector>

#include "../include/ecuda/event.hpp"
#include "../include/ecuda/matrix.hpp"

#include "config.hpp"
#ifndef BENCHMARK_THREADS
#define BENCHMARK_THREADS 480
#endif

struct Coordinate
{
	int x, y;
	Coordinate( int x = 0, int y = 0 ) : x(x), y(y) {}
	bool operator!=( const Coordinate& other ) const { return x != other.x || y != other.y; }
	bool operator==( const Coordinate& other ) const { return x == other.x && y == other.y; }
};

typedef Coordinate value_type;

template<typename T>
__global__
void rotateMatrix( typename ecuda::matrix<T>::const_kernel_argument src, typename ecuda::matrix<T>::kernel_argument dest )
{
	const int tx = blockIdx.x*blockDim.x+threadIdx.x;
	const int ty = blockIdx.y*blockDim.y+threadIdx.y;
	if( tx < src.number_rows() && ty < src.number_columns() ) dest(ty,tx) = src(tx,ty);
}

int main( int argc, char* argv[] )
{

	const std::size_t N = 10000;

	std::vector<value_type> hostSequence1( N*N );
	for( std::size_t i = 0; i < N; ++i ) {
		for( std::size_t j = 0; j < N; ++j ) hostSequence1[i*N+j] = Coordinate(i,j);
	}
	std::random_shuffle( hostSequence1.begin(), hostSequence1.end() );

	ecuda::event start1, stop1;
	start1.record();

	ecuda::matrix<value_type> deviceMatrix1( N, N );
	ecuda::matrix<value_type> deviceMatrix2( N, N );

	stop1.record();
	stop1.synchronize();
	std::cout << "Initialization Time: " << (stop1-start1) << "ms" << std::endl;

	ecuda::copy( hostSequence1.begin(), hostSequence1.end(), deviceMatrix1.begin() );

	ecuda::event start2, stop2;
	start2.record();
	dim3 grid( N, (N+BENCHMARK_THREADS-1)/BENCHMARK_THREADS ), threads( 1, BENCHMARK_THREADS );
	CUDA_CALL_KERNEL_AND_WAIT( rotateMatrix<value_type><<<grid,threads>>>( deviceMatrix1, deviceMatrix2 ) );
	stop2.record();
	stop2.synchronize();
	std::cout << "Kernel Time: " << (stop2-start2) << "ms" << std::endl;

	ecuda::event start3, stop3;
	start3.record();

	std::vector<value_type> hostSequence2( N*N );
	ecuda::copy( deviceMatrix2.begin(), deviceMatrix2.end(), hostSequence2.begin() );
	bool isEqual = true;
	for( std::size_t i = 0; i < N; ++i ) {
		for( std::size_t j = 0; j < N; ++j ) {
			if( hostSequence1[i*N+j] == hostSequence2[j*N+i] ) continue;
			isEqual = false;
		}
	}

	stop3.record();
	stop3.synchronize();
	std::cout << "Comparison Time: " << (stop3-start3) << "ms" << std::endl;

	if( isEqual )
		std::cout << "Test successful." << std::endl;
	else
		std::cout << "Test failed." << std::endl;

	return EXIT_SUCCESS;

}

