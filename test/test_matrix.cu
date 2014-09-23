#include <iostream>
#include <cstdio>
#include <estd/matrix.hpp>
#include "../include/ecuda/matrix.hpp"

__global__ void testKernel( const ecuda::matrix<float> input, ecuda::matrix<float> output )
//__global__ void testKernel( const float* inputMemory, const size_t pitch, ecuda::matrix<float> output )
{
	const int index = threadIdx.x;
	//const ecuda::array<float> input( in );
	//ecuda::array<float> output( out );
	//printf( "index=%i value_before=%.2f\n", index, input[index] );
	const int rowIndex = index/input.column_size();
	const int columnIndex = index % input.column_size();
	output[rowIndex][columnIndex] = input[rowIndex][columnIndex] / static_cast<float>(10.0);
	const float tmp = *(input.data()+(rowIndex*input.get_pitch()/sizeof(float)+columnIndex));
	printf( "index=%i row=%i col=%i tmp=%.2f value_before=%.2f value_after=%.2f\n", index, rowIndex, columnIndex, tmp, input[rowIndex][columnIndex], output[rowIndex][columnIndex] );
	//printf( "value_after=%.2f\n", output[index] );
}

__global__ void testKernel2( const ecuda::matrix<float>::row_type row, ecuda::matrix<float> output )
{
	const int index = threadIdx.x;
	const int rowIndex = index/row.size();
	const int columnIndex = index % row.size();
	output[rowIndex][columnIndex] = row[rowIndex] / static_cast<float>(10.0);
	//const float tmp = *(input.data()+(rowIndex*input.get_pitch()/sizeof(float)+columnIndex));
	//printf( "index=%i row=%i col=%i tmp=%.2f value_before=%.2f value_after=%.2f\n", index, rowIndex, columnIndex, tmp, input[rowIndex][columnIndex], output[rowIndex][columnIndex] );
}

int main( int argc, char* argv[] ) {
std::cerr << "step1" << std::endl;
	estd::matrix<float> hostMatrixInput( 10, 10 );
std::cerr << "step2" << std::endl;
	for( size_t i = 0; i < 10; ++i ) {
		for( size_t j = 0; j < 10; ++j ) {
			hostMatrixInput[i][j] = static_cast<float>((i+1)*(j+1));
		}
	}
std::cerr << "step3" << std::endl;
	ecuda::matrix<float> deviceMatrixInput( hostMatrixInput );
	for( size_t i = 0; i < 10; ++i ) {
		for( size_t j = 0; j < 10; ++j ) {
			std::cerr << " " << hostMatrixInput[i][j];
		}
		std::cerr << std::endl;
	}
std::cerr << "step4" << std::endl;
	ecuda::matrix<float> deviceMatrixOutput( 10, 10 );
std::cerr << "step5" << std::endl;

	dim3 dimBlock( 100, 1 ), dimGrid( 1, 1 );
	//testKernel<<<dimGrid,dimBlock>>>( deviceVectorInput.passToDevice(), deviceVectorOutput.passToDevice() );
std::cerr << "step6" << std::endl;
	//testKernel<<<dimGrid,dimBlock>>>( deviceMatrixInput, deviceMatrixOutput );
	testKernel2<<<dimGrid,dimBlock>>>( deviceMatrixInput[0], deviceMatrixOutput );
std::cerr << "step7" << std::endl;
	CUDA_CHECK_ERRORS
std::cerr << "step8" << std::endl;
	CUDA_CALL( cudaDeviceSynchronize() );
std::cerr << "COMPLETE" << std::endl;

	estd::matrix<float> hostMatrixOutput;
std::cerr << "step9" << std::endl;
	deviceMatrixOutput >> hostMatrixOutput;
std::cerr << "step10" << std::endl;
	for( size_t i = 0; i < hostMatrixOutput.row_size(); ++i ) {
		for( size_t j = 0; j < hostMatrixOutput.column_size(); ++j ) {
			std::cout << " " << hostMatrixOutput[i][j];
		}
		std::cout << std::endl;
	}

	return EXIT_SUCCESS;

}
