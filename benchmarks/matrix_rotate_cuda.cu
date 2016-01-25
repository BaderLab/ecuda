#include <iostream>
#include <vector>

#include "../include/ecuda/event.hpp"

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
void rotateMatrix( const T* src, T* dest, const size_t spitch, const size_t dpitch, const std::size_t nr, const std::size_t nc )
{
	const int tx = blockIdx.x*blockDim.x+threadIdx.x;
	const int ty = blockIdx.y*blockDim.y+threadIdx.y;
	if( tx < nr && ty < nc ) {
		const char* p = reinterpret_cast<const char*>(src);
		p += tx * spitch;
		p += ty*sizeof(T);
		char* q = reinterpret_cast<char*>(dest);
		q += ty * dpitch;
		q += tx*sizeof(T);
		*reinterpret_cast<T*>(q) = *reinterpret_cast<const T*>(p);
	}
}

int main( int argc, char* argv[] )
{

	const std::size_t N = 1000;

	ecuda::event start, stop;
	start.record();

	std::vector<value_type> hostSequence( N*N );
	for( std::size_t i = 0; i < N; ++i ) {
		for( std::size_t j = 0; j < N; ++j ) hostSequence[i*N+j] = value_type(i,j);
	}

	value_type *deviceMatrix1, *deviceMatrix2;
	size_t pitch1, pitch2;
	CUDA_CALL( cudaMallocPitch( &deviceMatrix1, &pitch1, N, N ) );
	CUDA_CALL( cudaMallocPitch( &deviceMatrix2, &pitch2, N, N ) );

	CUDA_CALL( cudaMemcpy2D( deviceMatrix1, pitch1, &hostSequence.front(), N*sizeof(value_type), N, N, cudaMemcpyHostToDevice ) );

	dim3 grid( (N+BENCHMARK_THREADS-1)/BENCHMARK_THREADS, N ), threads( BENCHMARK_THREADS, 1 );
	CUDA_CALL_KERNEL_AND_WAIT( rotateMatrix<value_type><<<grid,threads>>>( deviceMatrix1, deviceMatrix2, pitch1, pitch2, N, N ) );

	bool isEqual = true;
	value_type *hostColumn;
	CUDA_CALL( cudaMallocHost( &hostColumn, N*sizeof(value_type), cudaHostAllocDefault ) );

	for( std::size_t i = 0; i < N; ++i ) {
		CUDA_CALL( cudaMemcpy( hostColumn, reinterpret_cast<const char*>(deviceMatrix2)+pitch2*i, N*sizeof(value_type), cudaMemcpyDeviceToHost ) );
		for( std::size_t j = 0; j < N; ++j ) {
			if( hostSequence[j*N+i] != hostColumn[j] ) { isEqual = false; break; }
		}
	}

	cudaFreeHost( hostColumn );
	cudaFree( deviceMatrix2 );
	cudaFree( deviceMatrix1 );

	stop.record();
	stop.synchronize();

	if( isEqual )
		std::cout << "Test successful." << std::endl;
	else
		std::cout << "Test failed." << std::endl;

	std::cout << "Execution Time: " << (stop-start) << "ms" << std::endl;

	return EXIT_SUCCESS;

}

