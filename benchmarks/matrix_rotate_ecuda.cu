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
	bool operator!=( const Coordinate& other ) const { return x != other.x or y != other.y; }
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

	const std::size_t N = 1000;

	ecuda::event start, stop;
	start.record();

	std::vector<value_type> hostSequence( N*N );
	for( std::size_t i = 0; i < N; ++i ) {
		for( std::size_t j = 0; j < N; ++j ) hostSequence[i*N+j] = Coordinate(i,j);
	}

	ecuda::matrix<value_type> deviceMatrix1( N, N );
	ecuda::matrix<value_type> deviceMatrix2( N, N );

	ecuda::copy( hostSequence.begin(), hostSequence.end(), deviceMatrix1.begin() );

	dim3 grid( (N+BENCHMARK_THREADS-1)/BENCHMARK_THREADS, N ), threads( BENCHMARK_THREADS, 1 );
	CUDA_CALL_KERNEL_AND_WAIT( rotateMatrix<value_type><<<grid,threads>>>( deviceMatrix1, deviceMatrix2 ) );

	bool isEqual = true;
	std::vector< value_type, ecuda::host_allocator<Coordinate> > hostColumn( N );
	for( std::size_t i = 0; i < N; ++i ) {
		typename ecuda::matrix<value_type>::const_row_type deviceRow = deviceMatrix2[i];
		ecuda::copy( deviceRow.begin(), deviceRow.end(), hostColumn.begin() );
		for( std::size_t j = 0; j < N; ++j ) {
			if( hostSequence[j*N+i] != hostColumn[j] ) { isEqual = false; break; }
		}
	}

	stop.record();
	stop.synchronize();

	if( isEqual )
		std::cout << "Test successful." << std::endl;
	else
		std::cout << "Test failed." << std::endl;

	std::cout << "Execution Time: " << (stop-start) << "ms" << std::endl;

	return EXIT_SUCCESS;

}

