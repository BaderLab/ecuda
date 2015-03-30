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

template<typename T> __global__ void matrixTranspose( T* matrix, const std::size_t pitch, const std::size_t n, const std::size_t m );
template<typename T> __global__ void matrixTranspose( ecuda::matrix<T> matrix );


float cpuMatrixMultiply( const std::size_t n, const std::size_t m, const std::size_t p );
float cudaMatrixMultiply( const int numThreads, const std::size_t n = 100, const std::size_t m = 100, const std::size_t p = 100 );
float ecudaMatrixMultiply( const int numThreads, const std::size_t n = 100, const std::size_t m = 100, const std::size_t p = 100 );

float cpuMatrixTranspose( const std::size_t n );
float cudaMatrixTranspose( const int numberThreads, const std::size_t n = 100 );
float ecudaMatrixTranspose( const int numberThreads, const std::size_t n = 100 );

int main( int argc, char* argv[] ) {

	const std::size_t THREADS = 480;
	{
		const std::size_t n = 1000;
		const std::size_t m = 1000;
		const std::size_t p = 1000;
		std::cout << "MATRIX MULTIPLICATION CPU:      " << std::fixed <<   cpuMatrixMultiply(          n, m, p ) << " ms" << std::endl;
		std::cout << "MATRIX MULTIPLICATION STANDARD: " << std::fixed <<  cudaMatrixMultiply( THREADS, n, m, p ) << " ms" << std::endl;
		std::cout << "MATRIX MULTIPLICATION ECUDA   : " << std::fixed << ecudaMatrixMultiply( THREADS, n, m, p ) << " ms" << std::endl;
	}
	{
		const std::size_t n = 1000;
		std::cout << "MATRIX TRANSPOSE CPU:      " << std::fixed <<   cpuMatrixTranspose(          n ) << " ms" << std::endl;
		std::cout << "MATRIX TRANSPOSE STANDARD: " << std::fixed <<  cudaMatrixTranspose( THREADS, n ) << " ms" << std::endl;
		std::cout << "MATRIX TRANSPOSE ECUDA   : " << std::fixed << ecudaMatrixTranspose( THREADS, n ) << " ms" << std::endl;
	}

	return EXIT_SUCCESS;

}

template<typename T>
__global__ void matrixTranspose( ecuda::matrix<T> matrix ) {
	const int x = blockIdx.x*blockDim.x+threadIdx.x; // row
	const int y = blockIdx.y*blockDim.y+threadIdx.y; // column
	if( x < matrix.number_rows() and y < matrix.number_columns() and x < y ) ecuda::swap( matrix[y][x], matrix[x][y] );
}

template<typename T>
__global__ void matrixTranspose( T* matrix, const std::size_t pitch, const std::size_t r, const std::size_t c ) {
	const int x = blockIdx.x*blockDim.x+threadIdx.x; // row
	const int y = blockIdx.y*blockDim.y+threadIdx.y; // column
	if( x < r and y < c and x < y ) {
		T& valXY = *(reinterpret_cast<T*>( reinterpret_cast<char*>(matrix)+(pitch*x) )+y);
		T& valYX = *(reinterpret_cast<T*>( reinterpret_cast<char*>(matrix)+(pitch*y) )+x);
		T tmp = valXY;
		valXY = valYX;
		valYX = tmp;
	}
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

	cudaMallocPitch( &A, &pitchA, m*sizeof(double), n );
	cudaMallocPitch( &B, &pitchB, p*sizeof(double), m );
	cudaMallocPitch( &AB, &pitchAB, p*sizeof(double), n );

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

float cpuMatrixMultiply( const std::size_t n, const std::size_t m, const std::size_t p ) {

	ecuda::event start, stop;
	start.record();

	std::vector<double> A( n*m );
	std::vector<double> B( m*p );
	std::vector<double> AB( n*p );

	for( std::size_t i = 0; i < n; ++i ) {
		for( std::size_t j = 0; j < p; ++j ) {
			double sum = 0.0;
			for( std::size_t k = 0; k < m; ++k ) sum += A[i*m+k] * B[k*p+j];
			AB[i*p+j] = sum;
		}
	}

	stop.record();

	return ( stop - start );

}

float cudaMatrixTranspose( const int numThreads, const std::size_t n ) {

	ecuda::event start, stop;
	start.record();

	double *matrix;
	std::size_t pitch;
	cudaMallocPitch( &matrix, &pitch, n*sizeof(double), n );

	dim3 grid( (n+numThreads-1)/numThreads, (n+numThreads-1)/numThreads ), threads( numThreads, numThreads );
	matrixTranspose<<<grid,threads>>>( matrix, pitch, n, n );

	stop.record();
	stop.synchronize();

	cudaFree( matrix );

	return ( stop - start );

}

float ecudaMatrixTranspose( const int numThreads, const std::size_t n ) {

	ecuda::event start, stop;
	start.record();

	ecuda::matrix<double> matrix( n, n );

	dim3 grid( (n+numThreads-1)/numThreads, (n+numThreads-1)/numThreads ), threads( numThreads, numThreads );
	matrixTranspose<<<grid,threads>>>( matrix );

	stop.record();
	stop.synchronize();

	return ( stop - start );

}

float cpuMatrixTranspose( const std::size_t n ) {

	ecuda::event start, stop;
	start.record();

	std::vector<double> v( n*n );
	for( std::size_t i = 0; i < n; ++i ) {
		for( std::size_t j = i+1; j < n; ++j ) std::swap( v[i*n+j], v[j*n+i] );
	}

	stop.record();

	return ( stop - start );

}
