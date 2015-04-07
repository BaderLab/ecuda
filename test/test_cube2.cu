#include <iostream>
#include <cstdio>
#include <vector>
#include <estd/matrix.hpp>
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/vector.hpp"
#include "../include/ecuda/matrix.hpp"

//#include "../include/ecuda/cube.hpp"

/*
struct Coordinate {
	int x, y, z;
	HOST DEVICE Coordinate( const int x = 0, const int y = 0, const int z = 0 ) : x(x), y(y), z(z) {}
	friend std::ostream& operator<<( std::ostream& out, const Coordinate& coord ) {
		out << "[" << coord.x << "," << coord.y << "," << coord.z << "]";
		return out;
	}
};
*/

struct Coordinate {
	int x, y;
	HOST DEVICE Coordinate( const int x = 0, const int y = 0 ) : x(x), y(y) {}
	friend std::ostream& operator<<( std::ostream& out, const Coordinate& coord ) {
		out << "[" << coord.x << "," << coord.y << "]";
		return out;
	}
};

__global__ void linearize( const ecuda::matrix<Coordinate> matrix, ecuda::vector<Coordinate> vector ) {
	std::size_t index = 0;
	for( ecuda::matrix<Coordinate>::const_iterator iter = matrix.begin(); iter != matrix.end(); ++iter, ++index ) vector[index] = *iter;
}
/*
__global__ void linearize( const ecuda::cube<Coordinate> cube, ecuda::vector<Coordinate> vector ) {
	std::size_t index = 0;
	for( ecuda::cube<Coordinate>::const_iterator iter = cube.begin(); iter != cube.end(); ++iter, ++index ) vector[index] = *iter;
}
*/

int main( int argc, char* argv[] ) {

	std::vector<Coordinate> hostVector( 200 );
	{
		std::size_t index = 0;
		for( std::size_t i = 0; i < 5; ++i ) {
			for( std::size_t j = 0; j < 2; ++j ) {
				for( std::size_t k = 0; k < 20; ++k ) {
					//hostVector[index++] = Coordinate(i,j,k);
					hostVector[index++] = Coordinate(i*2+j,k);
				}
			}
		}
	}

//	ecuda::cube<Coordinate> deviceCube( 5, 2, 20 );
//	deviceCube.assign( hostVector.begin(), hostVector.end() );

	ecuda::matrix<Coordinate> deviceMatrix( 10, 20 );
	deviceMatrix.assign( hostVector.begin(), hostVector.end() );

	ecuda::vector<Coordinate> deviceVector1( 200 );
	linearize<<<1,1>>>( deviceMatrix, deviceVector1 );
	CUDA_CHECK_ERRORS();
	CUDA_CALL( cudaDeviceSynchronize() );
	std::vector<Coordinate> hostVector2( 200 );
	deviceVector1 >> hostVector2;
	for( std::size_t i = 0; i < hostVector2.size(); ++i ) {
		std::cout << "LINEAR MATRIX " << hostVector2[i] << std::endl;
	}

//	ecuda::vector<Coordinate> deviceVector2( 200 );
//	linearize<<<1,1>>>( deviceCube, deviceVector2 );
//	CUDA_CHECK_ERRORS();
//	CUDA_CALL( cudaDeviceSynchronize() );
//	std::vector<Coordinate> hostVector3( 200 );
//	deviceVector2 >> hostVector3;
//	for( std::size_t i = 0; i < hostVector3.size(); ++i ) {
//		std::cout << "LINEAR CUBE " << hostVector3[i] << std::endl;
//	}


}
