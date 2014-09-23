#include <iostream>
#include <cstdio>
#include "../include/ecuda/stack.hpp"
#include "../include/ecuda/vector.hpp"

__global__ void testKernel( ecuda::stack<float> input, ecuda::vector<float> output )
//__global__ void testKernel( const ecuda::array<float>::DevicePayload in, ecuda::array<float>::DevicePayload out )
{
	const int index = threadIdx.x;
	while( !input.empty() ) { /*output.push_back( input.top() );*/ input.pop(); }
	//output[index] = input[index]*static_cast<float>(index);
        //printf( "index=%i value_before=%.2f value_after=%.2f\n", index, input[index], output[index] );
}


int main( int argc, char* argv[] ) {
std::cerr << "step1" << std::endl;
	std::vector<float> hostVectorInput( 100 );
std::cerr << "step2" << std::endl;
	for( size_t i = 0; i < 100; ++i ) hostVectorInput[i] = i+1;
	//ecuda::array<float> deviceVectorInput( &hostVectorInput.front(), hostVectorInput.size() );
std::cerr << "step3" << std::endl;
	ecuda::stack<float> deviceVectorInput; //( hostVectorInput );
std::cerr << "step4" << std::endl;
	ecuda::vector<float> deviceVectorOutput( hostVectorInput.size() );
std::cerr << "step5" << std::endl;

	dim3 dimBlock( 100, 1 ), dimGrid( 1, 1 );
	//testKernel<<<dimGrid,dimBlock>>>( deviceVectorInput.passToDevice(), deviceVectorOutput.passToDevice() );
std::cerr << "step6" << std::endl;
	testKernel<<<dimGrid,dimBlock>>>( deviceVectorInput, deviceVectorOutput );
std::cerr << "step7" << std::endl;
	CUDA_CHECK_ERRORS
std::cerr << "step8" << std::endl;
	CUDA_CALL( cudaDeviceSynchronize() );
std::cerr << "COMPLETE" << std::endl;

	std::vector<float> hostVectorOutput;
std::cerr << "step9" << std::endl;
	//deviceVectorOutput >> hostVectorOutput;
std::cerr << "step10" << std::endl;
	for( size_t i = 0; i < hostVectorOutput.size(); ++i ) std::cout << "[" << i << "]=" << hostVectorOutput[i] << std::endl;

	return EXIT_SUCCESS;

}
