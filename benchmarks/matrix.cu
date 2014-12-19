#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>
#include <estd/matrix.hpp>
#include "../include/ecuda/matrix.hpp"
#include "../include/ecuda/event.hpp"

template<typename T>
struct coord_t { T x, y; };

template<typename T>
__device__
coord_t<T> doSomething( const coord_t<T>& value ) {
	coord_t<T> result;
	result.x = value.x;
	result.y = value.y;
	for( std::size_t i = 0; i < 10000; ++i ) { result.x += 0.0001; result.y += 0.0001; }
	return result;
}

template<typename T>
__global__
void manipulateMatrix( ecuda::matrix< coord_t<T> > input ) {
	const int x = blockIdx.x*blockDim.x+threadIdx.x / input.column_size();
	const int y = blockIdx.x*blockDim.x+threadIdx.x % input.column_size();
	if( x < input.row_size() and y < input.column_size() ) {
		input[x][y] = doSomething<T>( input[x][y] );
	}
}

template<typename T>
__global__
void manipulateMatrix( coord_t<T>* input, std::size_t n, std::size_t m ) {
	const int x = blockIdx.x*blockDim.x+threadIdx.x / m;
	const int y = blockIdx.x*blockDim.x+threadIdx.x % m;
	if( x < n and y < m ) {
		input[x*m+y] = doSomething<T>( input[x*m+y] );
	}
}

int main( int argc, char* argv[] ) {

	const std::size_t N = 1000;
	const std::size_t THREADS = 800;

	estd::matrix< coord_t<double>, std::size_t, std::size_t > hostMatrix( N, N );
	for( std::size_t i = 0; i < N; ++i ) {
		for( std::size_t j = 0; j < N; ++j ) {
			coord_t<double>& coord = hostMatrix[i][j];
			coord.x = static_cast<double>(i);
			coord.y = static_cast<double>(j);
		}
	}

	ecuda::matrix< coord_t<double> > deviceMatrix( N, N );
	deviceMatrix << hostMatrix;

	coord_t<double>* rawData = NULL;
	CUDA_CALL( cudaMalloc( reinterpret_cast<void**>(&rawData), N*N*sizeof(coord_t<double>) ) );
	CUDA_CALL( cudaMemcpy( reinterpret_cast<void*>(rawData), reinterpret_cast<const void*>(hostMatrix.data()), N*N*sizeof(double), cudaMemcpyHostToDevice ) );

	dim3 grid( (N*N+THREADS-1)/THREADS ), threads( 1, THREADS );

	{
		ecuda::event start, stop;
		start.record();
		manipulateMatrix<double><<<grid,threads>>>( deviceMatrix );
		CUDA_CALL( cudaDeviceSynchronize() );
		CUDA_CHECK_ERRORS();
		stop.record();
		stop.synchronize();
		std::cout << "TIME (ecuda): " << std::fixed << (stop-start) << std::endl;
		estd::matrix< coord_t<double> > results( N, N );
		deviceMatrix >> results;
		std::cout << "[0,0]=" << std::fixed << results[0][0].x << "," << results[0][0].y << std::endl;
		std::cout << "[" << (N-1) << "," << (N-1) << "]=" << std::fixed << results[N-1][N-1].x << "," << results[N-1][N-1].y << std::endl;
	}

	{
		ecuda::event start, stop;
		start.record();
		manipulateMatrix<double><<<grid,threads>>>( rawData, N, N );
		CUDA_CALL( cudaDeviceSynchronize() );
		CUDA_CHECK_ERRORS();
		stop.record();
		stop.synchronize();
		std::cout << "TIME (raw):  " << std::fixed << (stop-start) << std::endl;
		std::vector< coord_t<double> > results( N*N );
		CUDA_CALL( cudaMemcpy( &results.front(), rawData, N*N*sizeof(double), cudaMemcpyDeviceToHost ) );
		std::cout << "[0,0]=" << std::fixed << results[0].x << "," << results[0].y << std::endl;
		std::cout << "[" << (N-1) << "," << (N-1) << "]=" << std::fixed << results.back().x << "," << results.back().y << std::endl;
	}

	return EXIT_SUCCESS;

}
