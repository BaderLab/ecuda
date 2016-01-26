#include <algorithm>
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
	bool operator==( const Coordinate& other ) const { return x == other.x && y == other.y; }
	friend std::ostream& operator<<( std::ostream& out, const Coordinate& coord )
	{
		out << "(" << coord.x << "," << coord.y << ")";
		return out;
	}
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

	const std::size_t N = 10000;

	std::vector<value_type> hostSequence1( N*N );
	for( std::size_t i = 0; i < N; ++i ) {
		for( std::size_t j = 0; j < N; ++j ) hostSequence1[i*N+j] = value_type(i,j);
	}
	std::random_shuffle( hostSequence1.begin(), hostSequence1.end() );

	ecuda::event start1, stop1;
	start1.record();

	value_type *deviceMatrix1, *deviceMatrix2;
	size_t pitch1, pitch2;
	CUDA_CALL( cudaMallocPitch( &deviceMatrix1, &pitch1, N*sizeof(value_type), N ) );
	CUDA_CALL( cudaMemset2D( deviceMatrix1, pitch1, 0, N*sizeof(value_type), N ) );
	CUDA_CALL( cudaMallocPitch( &deviceMatrix2, &pitch2, N*sizeof(value_type), N ) );
	CUDA_CALL( cudaMemset2D( deviceMatrix2, pitch2, 0, N*sizeof(value_type), N ) );

	CUDA_CALL( cudaMemcpy2D( deviceMatrix1, pitch1, &hostSequence1.front(), N*sizeof(value_type), N*sizeof(value_type), N, cudaMemcpyHostToDevice ) );

	stop1.record();
	stop1.synchronize();
	std::cout << "Initialization Time: " << (stop1-start1) << "ms" << std::endl;

	ecuda::event start2, stop2;
	start2.record();
	dim3 grid( N, (N+BENCHMARK_THREADS-1)/BENCHMARK_THREADS ), threads( 1, BENCHMARK_THREADS );
	CUDA_CALL_KERNEL_AND_WAIT( rotateMatrix<value_type><<<grid,threads>>>( deviceMatrix1, deviceMatrix2, pitch1, pitch2, N, N ) );
	stop2.record();
	stop2.synchronize();
	std::cout << "Kernel Run Time: " << (stop2-start2) << "ms" << std::endl;

	ecuda::event start3, stop3;
	start3.record();

	std::vector<value_type> hostSequence2( N*N );
	CUDA_CALL( cudaMemcpy2D( &hostSequence2.front(), N*sizeof(value_type), deviceMatrix2, pitch2, N*sizeof(value_type), N, cudaMemcpyDeviceToHost ) );
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

	cudaFree( deviceMatrix2 );
	cudaFree( deviceMatrix1 );

	if( isEqual )
		std::cout << "Test successful." << std::endl;
	else
		std::cout << "Test failed." << std::endl;

	return EXIT_SUCCESS;

}

