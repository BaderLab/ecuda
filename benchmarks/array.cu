#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>
#include "../include/ecuda/array.hpp"

template<typename T>
__device__
T doSomething( const T& value ) {
	double result = static_cast<double>(0);
	for( std::size_t i = 0; i < 10000; ++i ) {
		result += 0.0001;
//		return log(static_cast<double>(value));
	}
	return static_cast<T>(result);
}

template<typename T>
__global__
void squareArray( ecuda::array<T> input ) {
	const int index = blockIdx.x*blockDim.x+threadIdx.x;
	if( index < input.size() ) {
		T& value = input[index];
		value = doSomething(value);
		//value = log(static_cast<double>(value));
	}
}

template<typename T>
__global__
void squareArray( T* input, std::size_t n ) {
	const int index = blockIdx.x*blockDim.x+threadIdx.x;
	if( index < n ) {
		T& value = input[index];
		value = doSomething(value);
		//value = log(static_cast<double>(value));
	}
}

int main( int argc, char* argv[] ) {

	const std::size_t N = 10000000;
	const std::size_t THREADS = 800;

	std::vector<double> hostData( N );
	for( std::size_t i = 0; i < N; ++i ) hostData[i] = i+1.0;

	ecuda::array<double> deviceData( N );
	deviceData << hostData;

	double* rawData = NULL;
	CUDA_CALL( cudaMalloc( reinterpret_cast<void**>(&rawData), N*sizeof(double) ) );
	CUDA_CALL( cudaMemcpy( reinterpret_cast<void*>(rawData), reinterpret_cast<const void*>(&hostData.front()), N*sizeof(double), cudaMemcpyHostToDevice ) );

	dim3 grid( (N+THREADS-1)/THREADS ), threads( THREADS );

//	const std::size_t ROUNDS = 100;
	{
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
//		std::time_t start, end;
//		std::time(&start);
//		for( std::size_t i = 0; i < ROUNDS; ++i ) {
		cudaEventRecord( start, 0 );
		squareArray<double><<<grid,threads>>>( deviceData );
		CUDA_CALL( cudaDeviceSynchronize() );
		CUDA_CHECK_ERRORS();
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &time, start, stop );
		cudaEventDestroy( start );
		cudaEventDestroy( stop );
//		}
//		std::time(&end);
		std::cout << "TIME (ecuda): " << std::fixed << time << std::endl;
//		std::cout << "TIME (ecuda): " << std::fixed << difftime( end, start ) << std::endl;
		std::vector<double> results( N );
		deviceData >> results;
		for( std::size_t i = 0; i < 10; ++i ) std::cout << "[" << i << "]=" << std::fixed << results[i] << std::endl;
		std::cout << "[" << (N-1) << "]=" << std::fixed << results.back() << std::endl;
	}

	{
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
//		std::time_t start, end;
//		std::time(&start);
//		for( std::size_t i = 0; i < ROUNDS; ++i ) {
		cudaEventRecord( start, 0 );
		squareArray<double><<<grid,threads>>>( rawData, N );
		CUDA_CALL( cudaDeviceSynchronize() );
		CUDA_CHECK_ERRORS();
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &time, start, stop );
		cudaEventDestroy( start );
		cudaEventDestroy( stop );
//		}
//		std::time(&end);
		std::cout << "TIME (raw):  " << std::fixed << time << std::endl;
//		std::cout << "TIME (raw): " << std::fixed << difftime( end, start ) << std::endl;
		std::vector<double> results( N );
		CUDA_CALL( cudaMemcpy( &results.front(), rawData, N*sizeof(double), cudaMemcpyDeviceToHost ) );
		for( std::size_t i = 0; i < 10; ++i ) std::cout << "[" << i << "]=" << std::fixed << results[i] << std::endl;
		std::cout << "[" << (N-1) << "]=" << std::fixed << results.back() << std::endl;
	}

	return EXIT_SUCCESS;

}
