#include <iostream>
#include <cstdio>
#include "../include/ecuda/array.hpp"

__global__ void squareVector( const ecuda::array<float> input, ecuda::array<float> output ) {
	const int index = threadIdx.x;
	output[index] = input[index]*input[index];
}

__global__ void sumVector( const ecuda::array<float> input, ecuda::array<float> output ) {
	const int index = threadIdx.x;
	float sum = 0.0;
	ecuda::array<float>::const_iterator current = input.begin();
	const ecuda::array<float>::const_iterator end = input.end();
	while( current != end ) {
		sum += *current;
		++current;
	}
//	for( ecuda::array<float>::const_iterator iter = input.begin(); iter != input.end(); ++iter ) sum += *iter;
	output[index] = sum;
}

int main( int argc, char* argv[] ) {

	// prepare host vector
	const size_t n = 100;
	std::vector<float> hostVector( n );
	for( size_t i = 0; i < n; ++i ) hostVector[i] = i+1;

	// allocate some device arrays
	ecuda::array<float> deviceArray1( n, 3 ); // should have all 3
	ecuda::array<float> deviceArray2( deviceArray1 ); // should be a copy of deviceArray1
	const ecuda::array<float> deviceArray3( hostVector ); // should be a copy of the host vector

	ecuda::array<float> deviceArray4( n );
	dim3 dimBlock( n, 1 ), dimGrid( 1, 1 );
	squareVector<<<dimGrid,dimBlock>>>( deviceArray3, deviceArray4 );
	CUDA_CHECK_ERRORS
	CUDA_CALL( cudaDeviceSynchronize() );

	// copy array to host
	deviceArray4 >> hostVector;
	// print contents (should be 1^2,2^2,3^2,...)
	for( size_t i = 0; i < n; ++i ) std::cout << "test1.hostVector[" << i << "]=" << hostVector[i] << std::endl;

	sumVector<<<dimGrid,dimBlock>>>( deviceArray3, deviceArray4 );
	CUDA_CHECK_ERRORS
	CUDA_CALL( cudaDeviceSynchronize() );

	// copy array to host
	deviceArray4 >> hostVector;
	// print contents (should be sum(1:1000)=5050)
	for( size_t i = 0; i < n; ++i ) std::cout << "test2.hostVector[" << i << "]=" << hostVector[i] << std::endl;

	return EXIT_SUCCESS;

}
