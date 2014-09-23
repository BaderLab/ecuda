#include <iostream>
#include <cstdio>
#include <estd/matrix.hpp>
#include "../include/ecuda/matrix.hpp"

__global__ void transpose( const ecuda::matrix<float> inputMatrix, ecuda::matrix<float> outputMatrix ) {

	const size_t index = threadIdx.x;
	ecuda::matrix<float>::const_row_type row = inputMatrix[index];
	ecuda::matrix<float>::column_type column = outputMatrix[index];
	ecuda::matrix<float>::column_type::iterator columnIterator = column.begin();
	for( ecuda::matrix<float>::const_row_type::const_iterator iter = row.begin(); iter != row.end(); ++iter, ++columnIterator ) *columnIterator = *iter;

}

int main( int argc, char* argv[] ) {

	estd::matrix<float> hostMatrix( 10, 10 );
	for( size_t i = 0; i < 10; ++i ) {
		for( size_t j = 0; j < 10; ++j ) {
			hostMatrix[i][j] = static_cast<float>((i+1)*(j+1));
		}
	}

	const ecuda::matrix<float> deviceMatrix1( hostMatrix );
	for( size_t i = 0; i < 10; ++i ) {
		for( size_t j = 0; j < 10; ++j ) {
			std::cerr << " " << hostMatrix[i][j];
		}
		std::cerr << std::endl;
	}

	ecuda::matrix<float> deviceMatrix2( 10, 10 );

	dim3 dimBlock( 10, 1 ), dimGrid( 1, 1 );
	transpose<<<dimGrid,dimBlock>>>( deviceMatrix1, deviceMatrix2 );
	CUDA_CHECK_ERRORS
	CUDA_CALL( cudaDeviceSynchronize() );

	deviceMatrix2 >> hostMatrix;
	for( size_t i = 0; i < hostMatrix.row_size(); ++i ) {
		for( size_t j = 0; j < hostMatrix.column_size(); ++j ) {
			std::cout << " " << hostMatrix[i][j];
		}
		std::cout << std::endl;
	}

	return EXIT_SUCCESS;

}
