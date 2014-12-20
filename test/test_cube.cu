#define NDEBUG
#include <cassert>

#include <iostream>
#include <cstdio>
#include <estd/cube.hpp>
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/cube.hpp"

template<typename T>
struct coord_t { T x, y, z; };

typedef coord_t<double> Coordinate;

typedef unsigned char uint8_t;

__global__
void testAt( ecuda::cube<Coordinate> cube, ecuda::cube<uint8_t> result ) {
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x / (cube.column_size()*cube.depth_size());
	const std::size_t y = ( blockIdx.x*blockDim.x+threadIdx.x % (cube.column_size()*cube.depth_size()) ) / cube.depth_size();
	const std::size_t z = ( blockIdx.x*blockDim.x+threadIdx.x % (cube.column_size()*cube.depth_size()) ) % cube.depth_size();
	if( x < cube.row_size() and y < cube.column_size() and z < cube.depth_size() ) {
		if( cube.at(x,y,z).x == x and cube.at(x,y,z).y == y and cube.at(x,y,z).z == z ) result[x][y][z] = 1;
	}
}

__global__
void testAtConst( const ecuda::cube<Coordinate> cube, ecuda::cube<uint8_t> result ) {
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x / (cube.column_size()*cube.depth_size());
	const std::size_t y = ( blockIdx.x*blockDim.x+threadIdx.x % (cube.column_size()*cube.depth_size()) ) / cube.depth_size();
	const std::size_t z = ( blockIdx.x*blockDim.x+threadIdx.x % (cube.column_size()*cube.depth_size()) ) % cube.depth_size();
	if( x < cube.row_size() and y < cube.column_size() and z < cube.depth_size() ) {
		if( cube.at(x,y,z).x == x and cube.at(x,y,z).y == y and cube.at(x,y,z).z == z ) result[x][y][z] = 1;
	}
}

__global__
void testAtIndex( ecuda::cube<Coordinate> cube, ecuda::cube<uint8_t> result ) {
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x / (cube.column_size()*cube.depth_size());
	const std::size_t y = ( blockIdx.x*blockDim.x+threadIdx.x % (cube.column_size()*cube.depth_size()) ) / cube.depth_size();
	const std::size_t z = ( blockIdx.x*blockDim.x+threadIdx.x % (cube.column_size()*cube.depth_size()) ) % cube.depth_size();
	if( x < cube.row_size() and y < cube.column_size() and z < cube.depth_size() ) {
		const std::size_t index = x*cube.column_size()*cube.depth_size() + y*cube.depth_size() + z;
		if( cube.at(index).x == x and cube.at(index).y == y and cube.at(index).z == z ) result[x][y][z] = 1;
	}
}

__global__
void testAtIndexConst( const ecuda::cube<Coordinate> cube, ecuda::cube<uint8_t> result ) {
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x / (cube.column_size()*cube.depth_size());
	const std::size_t y = ( blockIdx.x*blockDim.x+threadIdx.x % (cube.column_size()*cube.depth_size()) ) / cube.depth_size();
	const std::size_t z = ( blockIdx.x*blockDim.x+threadIdx.x % (cube.column_size()*cube.depth_size()) ) % cube.depth_size();
	if( x < cube.row_size() and y < cube.column_size() and z < cube.depth_size() ) {
		const std::size_t index = x*cube.column_size()*cube.depth_size() + y*cube.depth_size() + z;
		if( cube.at(index).x == x and cube.at(index).y == y and cube.at(index).z == z ) result[x][y][z] = 1;
	}
}

__global__
void testGetRow( ecuda::cube<Coordinate> cube, ecuda::cube<uint8_t> result ) {
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x;
	if( x < cube.row_size() ) {
		ecuda::cube<Coordinate>::matrix_type slice = cube.get_row(x);
		for( std::size_t i = 0; i < cube.column_size(); ++i ) {
			for( std::size_t j = 0; j < cube.depth_size(); ++j ) {
				if( slice[i][j].x == x and slice[i][j].y == i and slice[i][j].z == j ) result[x][i][j] = 1;
			}
		}
	}
}

/*
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
*/


int main( int argc, char* argv[] ) {

	const int THREADS = 500;

	// test cube will be 10 x 20 x 15
	// each cell will have a coordinate struct
	// with the 0-based x,y,z location

	const std::size_t n = 10;
	const std::size_t m = 20;
	const std::size_t o = 15;
	estd::cube<Coordinate> hostCube( n, m, o );
	for( std::size_t i = 0; i < n; ++i ) {
		for( std::size_t j = 0; j < m; ++j ) {
			for( std::size_t k = 0; k < o; ++k ) {
				hostCube[i][j][k].x = i;
				hostCube[i][j][k].y = j;
				hostCube[i][j][k].z = k;
			}
		}
	}

	// test GPU memory copying
	{
		std::cerr << "Testing host=>device=>host copy..." << std::endl;
		ecuda::cube<Coordinate> deviceCube( n, m, o );
		deviceCube << hostCube;
		estd::cube<Coordinate> hostCube2( n, m, o );
		deviceCube >> hostCube2;
		assert( hostCube == hostCube2 );
	}

	{
		std::cerr << "Testing host=>device=>device=>host copy..." << std::endl;
		ecuda::cube<Coordinate> deviceCube( n, m, o );
		deviceCube << hostCube;
		ecuda::cube<Coordinate> deviceCube2( deviceCube );
		estd::cube<Coordinate> hostCube2( n, m, o );
		deviceCube2 >> hostCube2;
		assert( hostCube == hostCube2 );
	}

	{
		std::cerr << "Testing device object instantiation from host..." << std::endl;
		ecuda::cube<Coordinate> deviceCube( hostCube );
		estd::cube<Coordinate> hostCube2( n, m, o );
		deviceCube >> hostCube2;
		assert( hostCube == hostCube2 );
	}

	/*
	{
		std::cerr << "Testing assign()..." << std::endl;
		ecuda::cube<Coordinate> deviceCube;
		deviceMatrix.assign( hostMatrix.begin(), hostMatrix.end() );
		estd::matrix<Coordinate> hostMatrix2( n, m );
		deviceMatrix >> hostMatrix2;
		assert( hostMatrix == hostMatrix2 );
	}
	*/

	{
		std::cerr << "Testing at(x,y,z)..." << std::endl;
		dim3 grid( (n*m*o+THREADS-1)/THREADS ), threads( THREADS );
		ecuda::cube<Coordinate> deviceCube( n, m, o );
		deviceCube << hostCube;
		ecuda::cube<uint8_t> resultCube( n, m, o );
		testAt<<<grid,threads>>>( deviceCube, resultCube );
		CUDA_CALL( cudaThreadSynchronize() );
		CUDA_CHECK_ERRORS();
		estd::cube<uint8_t> hostResultCube( n, m, o );
		resultCube >> hostResultCube;
		assert( hostResultCube.row_size() == n );
		assert( hostResultCube.column_size() == m );
		assert( hostResultCube.depth_size() == o );
		for( std::size_t i = 0; i < n; ++i )
			for( std::size_t j = 0; j < m; ++j )
				for( std::size_t k = 0; k < o; ++k )
					assert( hostResultCube[i][j][k] );
	}

	{
		std::cerr << "Testing const at(x,y,z)..." << std::endl;
		dim3 grid( (n*m*o+THREADS-1)/THREADS ), threads( THREADS );
		const ecuda::cube<Coordinate> deviceCube( hostCube );
		ecuda::cube<uint8_t> resultCube( n, m, o );
		testAtConst<<<grid,threads>>>( deviceCube, resultCube );
		CUDA_CALL( cudaThreadSynchronize() );
		CUDA_CHECK_ERRORS();
		estd::cube<uint8_t> hostResultCube( n, m, o );
		resultCube >> hostResultCube;
		assert( hostResultCube.row_size() == n );
		assert( hostResultCube.column_size() == m );
		assert( hostResultCube.depth_size() == o );
		for( std::size_t i = 0; i < n; ++i )
			for( std::size_t j = 0; j < m; ++j )
				for( std::size_t k = 0; k < o; ++k )
					assert( hostResultCube[i][j][k] );
	}

	{
		std::cerr << "Testing at(index)..." << std::endl;
		dim3 grid( (n*m*o+THREADS-1)/THREADS ), threads( THREADS );
		ecuda::cube<Coordinate> deviceCube( n, m, o );
		deviceCube << hostCube;
		ecuda::cube<uint8_t> resultCube( n, m, o );
		testAtIndex<<<grid,threads>>>( deviceCube, resultCube );
		CUDA_CALL( cudaThreadSynchronize() );
		CUDA_CHECK_ERRORS();
		estd::cube<uint8_t> hostResultCube( n, m, o );
		resultCube >> hostResultCube;
		assert( hostResultCube.row_size() == n );
		assert( hostResultCube.column_size() == m );
		assert( hostResultCube.depth_size() == o );
		for( std::size_t i = 0; i < n; ++i )
			for( std::size_t j = 0; j < m; ++j )
				for( std::size_t k = 0; k < o; ++k )
					assert( hostResultCube[i][j][k] );
	}

	{
		std::cerr << "Testing const at(index)..." << std::endl;
		dim3 grid( (n*m*o+THREADS-1)/THREADS ), threads( THREADS );
		const ecuda::cube<Coordinate> deviceCube( hostCube );
		ecuda::cube<uint8_t> resultCube( n, m, o );
		testAtIndex<<<grid,threads>>>( deviceCube, resultCube );
		CUDA_CALL( cudaThreadSynchronize() );
		CUDA_CHECK_ERRORS();
		estd::cube<uint8_t> hostResultCube( n, m, o );
		resultCube >> hostResultCube;
		assert( hostResultCube.row_size() == n );
		assert( hostResultCube.column_size() == m );
		assert( hostResultCube.depth_size() == o );
		for( std::size_t i = 0; i < n; ++i )
			for( std::size_t j = 0; j < m; ++j )
				for( std::size_t k = 0; k < o; ++k )
					assert( hostResultCube[i][j][k] );
	}

	{
		std::cerr << "Testing attributes..." << std::endl;
		const ecuda::cube<Coordinate> deviceCube( hostCube );
		assert( deviceCube.size() == n*m );
		assert( deviceCube.row_size() == n );
		assert( deviceCube.column_size() == m );
		assert( deviceCube.depth_size() == o );
		assert( deviceCube.get_pitch() >= n*o*sizeof(Coordinate) );
		assert( deviceCube.data() );
	}

	{
		std::cerr << "Testing get_row(index)..." << std::endl;
		dim3 grid( (n+THREADS-1)/THREADS ), threads( THREADS );
		ecuda::cube<Coordinate> deviceCube( n, m, o );
		deviceCube << hostCube;
		ecuda::cube<uint8_t> resultCube( n, m, o );
		testGetRow<<<grid,threads>>>( deviceCube, resultCube );
		CUDA_CALL( cudaThreadSynchronize() );
		CUDA_CHECK_ERRORS();
		estd::cube<uint8_t> hostResultCube( n, m, o );
		resultCube >> hostResultCube;
		assert( hostResultCube.row_size() == n );
		assert( hostResultCube.column_size() == m );
		assert( hostResultCube.depth_size() == o );
		for( std::size_t i = 0; i < n; ++i )
			for( std::size_t j = 0; j < m; ++j )
				for( std::size_t k = 0; k < o; ++k )
					assert( hostResultCube[i][j][k] );
	}

	/*
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
	*/

	return EXIT_SUCCESS;

}

