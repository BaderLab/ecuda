#include <iostream>
#include <list>
//#include <initializer_list>
#include <vector>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/cube.hpp"
#include "../include/ecuda/models.hpp"

template<typename T>
__global__ void testIterators( const ecuda::cube<T> src, ecuda::cube<T> dest ) {
	typename ecuda::cube<T>::iterator result = dest.begin();
	//typename ecuda::matrix<T>::const_iterator result2 = result;
	for( typename ecuda::cube<T>::const_iterator iter = src.begin(); iter != src.end(); ++iter, ++result ) *result = *iter;
}

template<typename T>
__global__ void testIterators2( const ecuda::cube<T> src, ecuda::cube<T> dest ) {
	for( typename ecuda::cube<T>::size_type i = 0; i < src.number_rows(); ++i ) {
		for( typename ecuda::cube<T>::size_type j = 0; j < src.number_columns(); ++j ) {
			typename ecuda::cube<T>::const_depth_type srcDepth = src.get_depth(i,j);
			typename ecuda::cube<T>::depth_type destDepth = dest.get_depth(i,j);
			ecuda::copy( srcDepth.begin(), srcDepth.end(), destDepth.begin() );
		}
	}
}

int main( int argc, char* argv[] ) {

	std::vector<int> hostVector( 2*3*4 );
	for( unsigned i = 0; i < 2*3*4; ++i ) hostVector[i] = i;

	ecuda::cube<int> deviceCube( 2, 3, 4 );
	ecuda::copy( hostVector.begin(), hostVector.end(), deviceCube.begin() );

	std::cout << "deviceCube.number_rows()=" << deviceCube.number_rows() << std::endl;
	std::cout << "deviceCube.number_columns()=" << deviceCube.number_columns() << std::endl;
	std::cout << "deviceCube.number_depths()=" << deviceCube.number_depths() << std::endl;

	hostVector.assign( hostVector.size(), 0 );
	deviceCube >> hostVector;

	for( std::size_t i = 0; i < hostVector.size(); ++i ) std::cout << "[" << i << "]=" << hostVector[i] << std::endl;

//	{
//		ecuda::cube<int> deviceCube2( 10, 10, 10 );
//		testIterators2<<<1,1>>>( deviceCube, deviceCube2 );
//		CUDA_CHECK_ERRORS();
//		CUDA_CALL( cudaDeviceSynchronize() );
//	}

	return EXIT_SUCCESS;

}

