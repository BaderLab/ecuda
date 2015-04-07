#include <iostream>
#include <vector>

#include "../include/ecuda/vector.hpp"
#include "../include/ecuda/cube.hpp"

struct Coordinate {
	int x, y, z;
	Coordinate( const int x = 0, const int y = 0, const int z = 0 ) : x(x), y(y), z(z) {}
	friend std::ostream& operator<<( std::ostream& out, const Coordinate& coord ) {
		out << "[" << coord.x << "," << coord.y << "," << coord.z << "]";
		return out;
	}
};

__global__ void linearize( const ecuda::cube<Coordinate> cube, ecuda::vector<Coordinate> vector ) {
	std::size_t index = 0;
	for( ecuda::cube<Coordinate>::const_iterator iter = cube.begin(); iter != cube.end(); ++iter, ++index ) vector[index] = *iter;
}


int main( int argc, char* argv[] ) {

	std::vector<Coordinate> hostVector( 200 );
	{
		std::size_t index = 0;
		for( std::size_t i = 0; i < 5; ++i ) {
			for( std::size_t j = 0; j < 2; ++j ) {
				for( std::size_t k = 0; k < 20; ++k ) {
					hostVector[index++] = Coordinate(i,j,k);
				}
			}
		}
	}

	ecuda::cube<Coordinate> deviceCube( 5, 2, 20 );
	deviceCube.assign( hostVector.begin(), hostVector.end() );

	ecuda::vector<Coordinate> deviceVector( 200 );
	linearize<<<1,1>>>( deviceCube, deviceVector );
	CUDA_CHECK_ERRORS();
	CUDA_CALL( cudaDeviceSynchronize() );

	std::vector<Coordinate> hostVector2( 200 );
	deviceVector >> hostVector2;

	for( std::size_t i = 0; i < hostVector2.size(); ++i ) {
		std::cout << "LINEAR " << hostVector2[i] << std::endl;
	}


}
