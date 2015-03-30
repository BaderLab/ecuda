#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>
//#include <estd/matrix.hpp>
#include "../include/ecuda/matrix.hpp"
#include "../include/ecuda/event.hpp"

template<typename T>
__global__ void matrixMultiply(
	const T* A,
	std::size_t pitchA,
	const T* B,
	std::size_t pitchB,
	std::size_t n, std::size_t m, std::size_t p,
	T* AB,
	std::size_t pitchAB
);

template<typename T> __global__ void matrixMultiply( const ecuda::matrix<T> A, const ecuda::matrix<T> B, ecuda::matrix<T> AB );

float cudaMatrixMultiply( const int numThreads, const std::size_t n = 100, const std::size_t m = 100, const std::size_t p = 100 );
float ecudaMatrixMultiply( const int numThreads, const std::size_t n = 100, const std::size_t m = 100, const std::size_t p = 100 );

int main( int argc, char* argv[] ) {

	const std::size_t THREADS = 480;
	const std::size_t n = 1000;
	const std::size_t m = 1000;
	const std::size_t p = 1000;

	std::cout << "STANDARD: " << std::fixed <<  cudaMatrixMultiply( THREADS, n, m, p ) << " ms" << std::endl;
	std::cout << "ECUDA   : " << std::fixed << ecudaMatrixMultiply( THREADS, n, m, p ) << " ms" << std::endl;

	return EXIT_SUCCESS;

}

template<typename T>
__global__ void matrixMultiply(
	const T* A,
	std::size_t pitchA,
	const T* B,
	std::size_t pitchB,
	std::size_t n, std::size_t m, std::size_t p,
	T* AB,
	std::size_t pitchAB
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x; // row
	const int y = blockIdx.y*blockDim.y+threadIdx.y; // column
	if( x < n and y < p ) {
		T result = 0;
		for( std::size_t i = 0; i < m; ++i ) {
			const T A_ik = *(reinterpret_cast<const T*>( reinterpret_cast<const char*>(A)+(pitchA*i) )+x);
			const T B_kj = *(reinterpret_cast<const T*>( reinterpret_cast<const char*>(B)+(pitchB*y) )+i);
			result += A_ik * B_kj;
		}
		*(reinterpret_cast<T*>( reinterpret_cast<char*>(AB)+(pitchAB*x) )+y) = result;
	}
}

template<typename T>
__global__ void matrixMultiply(
	const ecuda::matrix<T> A,
	const ecuda::matrix<T> B,
	ecuda::matrix<T> AB
)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x; // row
	const int y = blockIdx.y*blockDim.y+threadIdx.y; // column
	if( x < A.number_rows() and y < B.number_columns() ) {
		T result = 0;
		for( std::size_t i = 0; i < A.number_columns(); ++i ) result += A[x][i] * B[i][y];
		AB[x][y] = result;
	}
}

float cudaMatrixMultiply( const int numThreads, const std::size_t n, const std::size_t m, const std::size_t p ) {

	ecuda::event start, stop;
	start.record();

	double *A, *B, *AB;
	std::size_t pitchA, pitchB, pitchAB;

	cudaMallocPitch( &A, &pitchA, n, m );
	cudaMallocPitch( &B, &pitchB, m, p );
	cudaMallocPitch( &AB, &pitchAB, n, p );

	dim3 grid( n, (p+numThreads-1)/numThreads ), threads( 1, numThreads );
	matrixMultiply<<<grid,threads>>>( A, pitchA, B, pitchB, n, m, p, AB, pitchAB );

	cudaFree( A );
	cudaFree( B );
	cudaFree( AB );

	stop.record();
	stop.synchronize();

	return ( stop - start );

}

float ecudaMatrixMultiply( const int numThreads, const std::size_t n, const std::size_t m, const std::size_t p ) {

	ecuda::event start, stop;
	start.record();

	ecuda::matrix<double> A( n, m );
	ecuda::matrix<double> B( m, p );
	ecuda::matrix<double> AB( n, p );

	dim3 grid( n, (p+numThreads-1)/numThreads ), threads( 1, numThreads );
	matrixMultiply<<<grid,threads>>>( A, B, AB );

	stop.record();
	stop.synchronize();

	return ( stop - start );

}
