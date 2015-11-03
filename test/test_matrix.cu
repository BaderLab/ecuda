#include <iomanip>
#include <iostream>
#include <list>
//#include <initializer_list>
#include <vector>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/matrix.hpp"
//#include "../include/ecuda/models.hpp"
#include "../include/ecuda/vector.hpp"

#ifndef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
template<typename T>
__global__ void testIterators( const typename ecuda::matrix<T>::kernel src, typename ecuda::matrix<T>::kernel dest ) {
	typename ecuda::matrix<T>::iterator result = dest.begin();
	//typename ecuda::matrix<T>::const_iterator result2 = result;
	for( typename ecuda::matrix<T>::const_iterator iter = src.begin(); iter != src.end(); ++iter, ++result ) *result = *iter;
}

template<typename T>
__global__ void testIterators2( const ecuda::matrix<T> src, ecuda::matrix<T> dest ) {
	for( typename ecuda::matrix<T>::size_type i = 0; i < src.number_columns(); ++i ) {
		typename ecuda::matrix<T>::const_column_type srcColumn = src.get_column(i);
		typename ecuda::matrix<T>::column_type destColumn = dest.get_column(i);
		ecuda::copy( srcColumn.begin(), srcColumn.end(), destColumn.begin() );
	}
}
#endif

int main( int argc, char* argv[] ) {

	const std::size_t nRows = 5;
	const std::size_t nCols = 21;

	std::vector<int> hostVector( nRows*nCols );
	for( unsigned i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;

	ecuda::matrix<int> deviceMatrix( nRows, nCols );
	// below needs to be made to work
	ecuda::copy( hostVector.begin(), hostVector.end(), deviceMatrix.begin() ); // TODO: confirm the pseudo-contiguous nature of deviceMatrix.begin() is being accounted for
	#ifndef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
	{
		ecuda::matrix<int> deviceMatrix2( nRows, nCols );
		testIterators2<<<1,1>>>( deviceMatrix, deviceMatrix2 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );

		// need alternative to below
		//ecuda::copy( deviceMatrix.get_column(0).begin(), deviceMatrix.get_column(0).end(), deviceMatrix2.get_column(1).begin() );

		std::cout << "EQUAL " << ( deviceMatrix == deviceMatrix2 ? "true" : "false" ) << std::endl;
		std::cout << "LESS THAN " << ( deviceMatrix < deviceMatrix2 ? "true" : "false" ) << std::endl;
	}
	#endif

	{
		std::vector<int> tmp( nRows*nCols );
		ecuda::copy( deviceMatrix.begin(), deviceMatrix.end(), tmp.begin() ); // TODO: confirm the pseudo-contiguous nature of deviceMatrix.begin() is being accounted for
		unsigned mrkr = 0;
		for( unsigned i = 0; i < nRows; ++i ) {
			std::cout << "ROW[" << i << "] ="; for( unsigned j = 0; j < nCols; ++j, ++mrkr ) std::cout << " " << std::setw(3) << tmp[mrkr]; std::cout << std::endl;
		}
	}

	ecuda::matrix_transpose( deviceMatrix );

	{
		//ecuda::matrix<int> deviceMatrix2( 2, 2 );
		//deviceMatrix2.assign( { 1, 2, 3, 4 } );
	}

	return EXIT_SUCCESS;

}

