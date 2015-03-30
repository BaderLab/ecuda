#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>
//#include <estd/matrix.hpp>
#include "../include/ecuda/matrix.hpp"
#include "../include/ecuda/event.hpp"

template<typename T> __global__ void matrixTranspose( T* matrix, const std::size_t pitch, const std::size_t n );
template<typename T> __global__ void matrixTranspose( ecuda::matrix<T> matrix );


float cpuMatrixTranspose( const std::size_t n );
float cudaMatrixTranspose( const int numberThreads, const std::size_t n = 100 );
float ecudaMatrixTranspose( const int numberThreads, const std::size_t n = 100 );

int main( int argc, char* argv[] ) {

	const std::size_t THREADS = 480;
	const std::size_t n = 10000;
	std::cout << "MATRIX TRANSPOSE CPU  : " << std::fixed <<   cpuMatrixTranspose(          n ) << " ms" << std::endl;
	std::cout << "MATRIX TRANSPOSE CUDA : " << std::fixed <<  cudaMatrixTranspose( THREADS, n ) << " ms" << std::endl;
	std::cout << "MATRIX TRANSPOSE ECUDA: " << std::fixed << ecudaMatrixTranspose( THREADS, n ) << " ms" << std::endl;

	return EXIT_SUCCESS;

}

template<typename T>
__global__ void matrixTranspose( ecuda::matrix<T> matrix ) {
	const int x = blockIdx.x*blockDim.x+threadIdx.x; // row
	const int y = blockIdx.y*blockDim.y+threadIdx.y; // column
	if( x < matrix.number_rows() and y < matrix.number_columns() and x < y ) {
		//T& valXY = matrix.at( x, y );
		//T& valYX = matrix.at( y, x );
		//T& valXY = *(reinterpret_cast<T*>( reinterpret_cast<char*>(matrix.data())+(matrix.get_pitch()*x) )+y);
		//T& valYX = *(reinterpret_cast<T*>( reinterpret_cast<char*>(matrix.data())+(matrix.get_pitch()*y) )+x);
		T& valXY = matrix[x][y];
		T& valYX = matrix[y][x];
		T tmp = valXY;
		valXY = valYX;
		valYX = tmp;
	}
	//ecuda::swap( matrix[x][y], matrix[y][x] );
}

template<typename T>
__global__ void matrixTranspose( T* matrix, const std::size_t pitch, const std::size_t n ) {
	const int x = blockIdx.x*blockDim.x+threadIdx.x; // row
	const int y = blockIdx.y*blockDim.y+threadIdx.y; // column
	if( x < n and y < n and x < y ) {
		T& valXY = *(reinterpret_cast<T*>( reinterpret_cast<char*>(matrix)+(pitch*x) )+y);
		T& valYX = *(reinterpret_cast<T*>( reinterpret_cast<char*>(matrix)+(pitch*y) )+x);
		T tmp = valXY;
		valXY = valYX;
		valYX = tmp;
	}
}

float cudaMatrixTranspose( const int numThreads, const std::size_t n ) {

	ecuda::event start, stop;

	double *matrix;
	std::size_t pitch;
	cudaMallocPitch( &matrix, &pitch, n*sizeof(double), n );

	dim3 grid( n, (n+numThreads-1)/numThreads ), threads( 1, numThreads );
	start.record();
	matrixTranspose<<<grid,threads>>>( matrix, pitch, n );
	stop.record();

	CUDA_CHECK_ERRORS();
	stop.synchronize();

	cudaFree( matrix );

	return ( stop - start );

}

float ecudaMatrixTranspose( const int numThreads, const std::size_t n ) {

	ecuda::event start, stop;

	ecuda::matrix<double> matrix( n, n );

	dim3 grid( n, (n+numThreads-1)/numThreads ), threads( 1, numThreads );
	start.record();
	matrixTranspose<<<grid,threads>>>( matrix );
	stop.record();

	CUDA_CHECK_ERRORS();
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
