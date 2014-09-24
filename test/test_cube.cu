#include <iostream>
#include <cstdio>
#include <estd/cube.hpp>
#include "../include/ecuda/cube.hpp"

__global__ void scale( const ecuda::cube<float> inputCube, ecuda::cube<float> outputCube, const float factor ) {

	const size_t index = threadIdx.x;
printf( "entering thread=%i\n", index );
	ecuda::cube<float>::const_matrix_type inputSlice = inputCube[index];
	ecuda::cube<float>::matrix_type outputSlice = outputCube[index];

	const ecuda::cube<float>::const_matrix_type::size_type nr = inputSlice.row_size();
	for( ecuda::cube<float>::const_matrix_type::size_type i = 0; i < nr; ++i ) {
		ecuda::cube<float>::const_matrix_type::const_row_type inputRow = inputSlice[i];
		ecuda::cube<float>::matrix_type::row_type outputRow = outputSlice[i];
		ecuda::cube<float>::matrix_type::row_type::iterator outputIterator = outputRow.begin();
		// change 3rd column only
		*(outputIterator+3) = *(inputRow.begin()+3) * factor;
//		for( ecuda::cube<float>::const_matrix_type::const_row_type::const_iterator iter = inputRow.begin(); iter != inputRow.end(); ++iter, ++outputIterator ) {
//			*outputIterator = *iter * factor;
//printf( "index=%i i=%i input=%0.2f output=%0.2f\n", index, i, *iter, *outputIterator );
//		}
	}

}

int main( int argc, char* argv[] ) {

	unsigned counter = 0;
	estd::cube<float> hostCube( 10, 10, 10 );
	for( size_t i = 0; i < 10; ++i ) {
		for( size_t j = 0; j < 10; ++j ) {
			for( size_t k = 0; k < 10; ++k ) {
				hostCube[i][j][k] = static_cast<float>(++counter);
			}
		}
	}
	for( size_t i = 0; i < 10; ++i ) {
		for( size_t j = 0; j < 10; ++j ) {
			std::cout << "[" << i << "]";
			for( size_t k = 0; k < 10; ++k ) {
				std::cout << " " << hostCube[i][j][k];
			}
			std::cout << std::endl;
		}
	}
std::cerr << "cp1" << std::endl;
	const ecuda::cube<float> deviceCube1( hostCube );
std::cerr << "cp2" << std::endl;
	ecuda::cube<float> deviceCube2( 10, 10, 10 );
std::cerr << "cp3" << std::endl;

	dim3 dimBlock( 10, 1 ), dimGrid( 1, 1 );
	scale<<<dimGrid,dimBlock>>>( deviceCube1, deviceCube2, 3.0 );
	CUDA_CHECK_ERRORS
	CUDA_CALL( cudaDeviceSynchronize() );

std::cerr << "cp4" << std::endl;

	deviceCube2 >> hostCube;
std::cerr << "cp5" << std::endl;
	for( size_t i = 0; i < 10; ++i ) {
		for( size_t j = 0; j < 10; ++j ) {
			std::cout << "[" << i << "]";
			for( size_t k = 0; k < 10; ++k ) {
				std::cout << " " << hostCube[i][j][k];
			}
			std::cout << std::endl;
		}
	}

	return EXIT_SUCCESS;

}

