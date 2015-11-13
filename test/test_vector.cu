#include <iostream>
#include <list>
#include <vector>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/vector.hpp"

template<typename T,std::size_t N>
__global__ void testIterators( const typename ecuda::array<T,N>::kernel_argument src, typename ecuda::array<T,N>::kernel_argument dest ) {
	typename ecuda::array<T,N>::iterator result = dest.begin();
	for( typename ecuda::array<T,N>::const_iterator iter = src.begin(); iter != src.end(); ++iter, ++result ) *result = *iter;
}

int main( int argc, char* argv[] ) {

	std::vector<int> hostVector( 100 );
	for( unsigned i = 0; i < 100; ++i ) hostVector[i] = i;

	ecuda::array<int,100> deviceArray;
	ecuda::copy( hostVector.begin(), hostVector.end(), deviceArray.begin() );

	{
		ecuda::array<int,100> deviceArray2;
		testIterators<int,100><<<1,1>>>( deviceArray, deviceArray2 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
	}


	return EXIT_SUCCESS;

}

