#define NDEBUG
#include <cassert>

#include <iostream>
#include <cstdio>
#include <estd/matrix.hpp>
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/matrix.hpp"

template<typename T>
struct coord_t { T x, y; };

typedef coord_t<double> Coordinate;

typedef unsigned char uint8_t;

__global__
void testAt( ecuda::matrix<Coordinate> matrix, ecuda::matrix<uint8_t> result ) {
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x / matrix.column_size();
	const std::size_t y = blockIdx.x*blockDim.x+threadIdx.x % matrix.column_size();
	if( x < matrix.row_size() and y < matrix.column_size() ) {
		if( matrix.at(x,y).x == x and matrix.at(x,y).y == y ) result[x][y] = 1;
	}
}

__global__
void testAtConst( const ecuda::matrix<Coordinate> matrix, ecuda::matrix<uint8_t> result ) {
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x / matrix.column_size();
	const std::size_t y = blockIdx.x*blockDim.x+threadIdx.x % matrix.column_size();
	if( x < matrix.row_size() and y < matrix.column_size() ) {
		if( matrix.at(x,y).x == x and matrix.at(x,y).y == y ) result[x][y] = 1;
	}
}

__global__
void testAtIndex( ecuda::matrix<Coordinate> matrix, ecuda::matrix<uint8_t> result ) {
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x / matrix.column_size();
	const std::size_t y = blockIdx.x*blockDim.x+threadIdx.x % matrix.column_size();
	if( x < matrix.row_size() and y < matrix.column_size() ) {
		if( matrix.at(x*matrix.column_size()+y).x == x and matrix.at(x*matrix.column_size()+y).y == y ) result[x][y] = 1;
	}
}

__global__
void testAtIndexConst( const ecuda::matrix<Coordinate> matrix, ecuda::matrix<uint8_t> result ) {
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x / matrix.column_size();
	const std::size_t y = blockIdx.x*blockDim.x+threadIdx.x % matrix.column_size();
	if( x < matrix.row_size() and y < matrix.column_size() ) {
		if( matrix.at(x*matrix.column_size()+y).x == x and matrix.at(x*matrix.column_size()+y).y == y ) result[x][y] = 1;
	}
}

__global__
void testGetRow( ecuda::matrix<Coordinate> matrix, ecuda::matrix<uint8_t> result ) {
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x;
	if( x < matrix.row_size() ) {
		ecuda::matrix<Coordinate>::row_type row = matrix[x];
		ecuda::matrix<Coordinate>::row_type::iterator iter = row.begin();
		std::size_t y = 0;
		for( ; iter != row.end(); ++iter, ++y )	if( iter->x == x and iter->y == y ) result[x][y] = 1;
	}
}

__global__
void testGetColumn( ecuda::matrix<Coordinate> matrix, ecuda::matrix<uint8_t> result ) {
	const std::size_t y = blockIdx.x*blockDim.x+threadIdx.x;
	if( y < matrix.column_size() ) {
		ecuda::matrix<Coordinate>::column_type column = matrix.get_column(y);
		ecuda::matrix<Coordinate>::column_type::iterator iter = column.begin();
		std::size_t x = 0;
		for( ; iter != column.end(); ++iter, ++x )	if( iter->x == x and iter->y == y ) result[x][y] = 1;
	}
}


int main( int argc, char* argv[] ) {

	const int THREADS = 500;

	// test matrix will be 10 x 20
	// each cell will have a coordinate struct
	// with the 0-based x,y location

	const std::size_t n = 10;
	const std::size_t m = 20;
	estd::matrix<Coordinate> hostMatrix( n, m );
	for( std::size_t i = 0; i < n; ++i ) {
		for( std::size_t j = 0; j < m; ++j ) {
			hostMatrix[i][j].x = i;
			hostMatrix[i][j].y = j;
		}
	}

	// test GPU memory copying
	{
		std::cerr << "Testing host=>device=>host copy..." << std::endl;
		ecuda::matrix<Coordinate> deviceMatrix( n, m );
		deviceMatrix << hostMatrix;
		estd::matrix<Coordinate> hostMatrix2( n, m );
		deviceMatrix >> hostMatrix2;
		assert( hostMatrix == hostMatrix2 );
	}

	{
		std::cerr << "Testing host=>device=>device=>host copy..." << std::endl;
		ecuda::matrix<Coordinate> deviceMatrix( n, m );
		deviceMatrix << hostMatrix;
		ecuda::matrix<Coordinate> deviceMatrix2( deviceMatrix );
		estd::matrix<Coordinate> hostMatrix2( n, m );
		deviceMatrix2 >> hostMatrix2;
		assert( hostMatrix == hostMatrix2 );
	}

	{
		std::cerr << "Testing device object instantiation from host..." << std::endl;
		ecuda::matrix<Coordinate> deviceMatrix( hostMatrix );
		estd::matrix<Coordinate> hostMatrix2( n, m );
		deviceMatrix >> hostMatrix2;
		assert( hostMatrix == hostMatrix2 );
	}

	{
		std::cerr << "Testing assign()..." << std::endl;
		ecuda::matrix<Coordinate> deviceMatrix;
		deviceMatrix.assign( hostMatrix.begin(), hostMatrix.end() );
		estd::matrix<Coordinate> hostMatrix2( n, m );
		deviceMatrix >> hostMatrix2;
		assert( hostMatrix == hostMatrix2 );
	}

	{
		std::cerr << "Testing at(x,y)..." << std::endl;
		dim3 grid( (n*m+THREADS-1)/THREADS ), threads( THREADS );
		ecuda::matrix<Coordinate> deviceMatrix( n, m );
		deviceMatrix << hostMatrix;
		ecuda::matrix<uint8_t> resultMatrix( n, m );
		testAt<<<grid,threads>>>( deviceMatrix, resultMatrix );
		CUDA_CALL( cudaThreadSynchronize() );
		CUDA_CHECK_ERRORS();
		estd::matrix<uint8_t> hostResultMatrix( n, m );
		resultMatrix >> hostResultMatrix;
		assert( hostResultMatrix.row_size() == n );
		assert( hostResultMatrix.column_size() == m );
		for( std::size_t i = 0; i < n; ++i )
			for( std::size_t j = 0; j < m; ++j )
				assert( hostResultMatrix[i][j] );
	}

	{
		std::cerr << "Testing const at(x,y)..." << std::endl;
		dim3 grid( (n*m+THREADS-1)/THREADS ), threads( THREADS );
		const ecuda::matrix<Coordinate> deviceMatrix( hostMatrix );
		ecuda::matrix<uint8_t> resultMatrix( n, m );
		testAtConst<<<grid,threads>>>( deviceMatrix, resultMatrix );
		CUDA_CALL( cudaThreadSynchronize() );
		CUDA_CHECK_ERRORS();
		estd::matrix<uint8_t> hostResultMatrix( n, m );
		resultMatrix >> hostResultMatrix;
		assert( hostResultMatrix.row_size() == n );
		assert( hostResultMatrix.column_size() == m );
		for( std::size_t i = 0; i < n; ++i )
			for( std::size_t j = 0; j < m; ++j )
				assert( hostResultMatrix[i][j] );
	}

	{
		std::cerr << "Testing at(index)..." << std::endl;
		dim3 grid( (n*m+THREADS-1)/THREADS ), threads( THREADS );
		ecuda::matrix<Coordinate> deviceMatrix( n, m );
		deviceMatrix << hostMatrix;
		ecuda::matrix<uint8_t> resultMatrix( n, m );
		testAtIndex<<<grid,threads>>>( deviceMatrix, resultMatrix );
		CUDA_CALL( cudaThreadSynchronize() );
		CUDA_CHECK_ERRORS();
		estd::matrix<uint8_t> hostResultMatrix( n, m );
		resultMatrix >> hostResultMatrix;
		assert( hostResultMatrix.row_size() == n );
		assert( hostResultMatrix.column_size() == m );
		for( std::size_t i = 0; i < n; ++i )
			for( std::size_t j = 0; j < m; ++j )
				assert( hostResultMatrix[i][j] );
	}

	{
		std::cerr << "Testing const at(index)..." << std::endl;
		dim3 grid( (n*m+THREADS-1)/THREADS ), threads( THREADS );
		const ecuda::matrix<Coordinate> deviceMatrix( hostMatrix );
		ecuda::matrix<uint8_t> resultMatrix( n, m );
		testAtIndexConst<<<grid,threads>>>( deviceMatrix, resultMatrix );
		CUDA_CALL( cudaThreadSynchronize() );
		CUDA_CHECK_ERRORS();
		estd::matrix<uint8_t> hostResultMatrix( n, m );
		resultMatrix >> hostResultMatrix;
		assert( hostResultMatrix.row_size() == n );
		assert( hostResultMatrix.column_size() == m );
		for( std::size_t i = 0; i < n; ++i )
			for( std::size_t j = 0; j < m; ++j )
				assert( hostResultMatrix[i][j] );
	}

	{
		std::cerr << "Testing attributes..." << std::endl;
		const ecuda::matrix<Coordinate> deviceMatrix( hostMatrix );
		assert( deviceMatrix.size() == n*m );
		assert( deviceMatrix.row_size() == n );
		assert( deviceMatrix.column_size() == m );
		assert( deviceMatrix.get_pitch() >= m*sizeof(Coordinate) );
		assert( deviceMatrix.data() );
	}

	{
		std::cerr << "Testing get_row(index)..." << std::endl;
		dim3 grid( (n+THREADS-1)/THREADS ), threads( THREADS );
		ecuda::matrix<Coordinate> deviceMatrix( n, m );
		deviceMatrix << hostMatrix;
		ecuda::matrix<uint8_t> resultMatrix( n, m );
		testGetRow<<<grid,threads>>>( deviceMatrix, resultMatrix );
		CUDA_CALL( cudaThreadSynchronize() );
		CUDA_CHECK_ERRORS();
		estd::matrix<uint8_t> hostResultMatrix( n, m );
		resultMatrix >> hostResultMatrix;
		assert( hostResultMatrix.row_size() == n );
		assert( hostResultMatrix.column_size() == m );
		for( std::size_t i = 0; i < n; ++i )
			for( std::size_t j = 0; j < m; ++j )
				assert( hostResultMatrix[i][j] );
	}

	{
		std::cerr << "Testing get_column(index)..." << std::endl;
		dim3 grid( (n+THREADS-1)/THREADS ), threads( THREADS );
		ecuda::matrix<Coordinate> deviceMatrix( n, m );
		deviceMatrix << hostMatrix;
		ecuda::matrix<uint8_t> resultMatrix( n, m );
		testGetColumn<<<grid,threads>>>( deviceMatrix, resultMatrix );
		CUDA_CALL( cudaThreadSynchronize() );
		CUDA_CHECK_ERRORS();
		estd::matrix<uint8_t> hostResultMatrix( n, m );
		resultMatrix >> hostResultMatrix;
		assert( hostResultMatrix.row_size() == n );
		assert( hostResultMatrix.column_size() == m );
		for( std::size_t i = 0; i < n; ++i )
			for( std::size_t j = 0; j < m; ++j )
				assert( hostResultMatrix[i][j] );
	}

	return EXIT_SUCCESS;

}
