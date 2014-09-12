#include <iostream>
#include <cstdio>
#include "../include/array.hpp"

__global__ void testKernel( const ecuda::array<float>& input, ecuda::array<float>& output )
{
	const int index = threadIdx.x;
	printf( "index=%i\n", index );
	output[index] = input[index]*index;
}


int main( int argc, char* argv[] ) {

	std::vector<float> hostVectorInput( 100 );
	for( size_t i = 0; i < 100; ++i ) hostVectorInput[i] = i+1;
	ecuda::array<float> deviceVectorInput( &hostVectorInput.front(), hostVectorInput.size() );
	ecuda::array<float> deviceVectorOutput( hostVectorInput.size() );

	dim3 dimBlock( 100, 1 ), dimGrid( 1, 1 );
	testKernel<<<dimGrid,dimBlock>>>( deviceVectorInput, deviceVectorOutput );
	CUDA_CHECK_ERRORS

	std::vector<float> hostVectorOutput;
	deviceVectorOutput >> hostVectorOutput;

	for( size_t i = 0; i < hostVectorOutput.size(); ++i ) std::cout << "[" << i << "]=" << hostVectorOutput[i] << std::endl;

	return EXIT_SUCCESS;

}
