#include <iostream>
#include <cstdio>
#include <vector>
#include <estd/matrix.hpp>
//#include "../include/ecuda/array.hpp"
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
template<typename T>
struct coord_t {
	T x, y;
	HOST DEVICE coord_t( const T& x = T(), const T& y = T() ) : x(x), y(y) {}
	bool operator==( const coord_t& other ) const { return x == other.x and y == other.y; }
	bool operator!=( const coord_t& other ) const { return !operator==(other); }
	friend std::ostream& operator<<( std::ostream& out, const coord_t& coord ) {
		out << "[" << coord.x << "," << coord.y << "]";
		return out;
	}
};

typedef coord_t<double> Coordinate;
/*
struct Coordinate {
	int x, y;
	HOST DEVICE Coordinate( const int x = 0, const int y = 0 ) : x(x), y(y) {}
	friend std::ostream& operator<<( std::ostream& out, const Coordinate& coord ) {
		out << "[" << coord.x << "," << coord.y << "]";
		return out;
	}
};
*/

/*
__global__ void linearize( const ecuda::matrix<Coordinate> matrix, ecuda::vector<Coordinate> vector ) {
	std::size_t index = 0;
	for( ecuda::matrix<Coordinate>::const_iterator iter = matrix.begin(); iter != matrix.end(); ++iter, ++index ) vector[index] = *iter;
}
*/

/*
__global__ void linearize( const ecuda::cube<Coordinate> cube, ecuda::vector<Coordinate> vector ) {
	std::size_t index = 0;
	for( ecuda::cube<Coordinate>::const_iterator iter = cube.begin(); iter != cube.end(); ++iter, ++index ) vector[index] = *iter;
}
*/

template<typename T> __global__
void kernel_linearize(
	const ecuda::matrix<T> matrix, ecuda::vector<T> vector
)
{
	std::size_t index = 0;
	for( typename ecuda::matrix<T>::const_iterator iter = matrix.begin(); iter != matrix.end(); ++iter, ++index ) vector[index] = *iter;
}

int main( int argc, char* argv[] ) {

	{
		std::vector<Coordinate> hostVector( 10*20 );
		unsigned index = 0;
		for( unsigned i = 0; i < 10; ++i ) {
			for( unsigned j = 0; j < 20; ++j, ++index ) {
				hostVector[index] = Coordinate(i,j);
			}
		}
		/*
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
		*/

	//	ecuda::cube<Coordinate> deviceCube( 5, 2, 20 );
	//	deviceCube.assign( hostVector.begin(), hostVector.end() );

		ecuda::matrix<Coordinate> deviceMatrix( 10, 20 );
		deviceMatrix.assign( hostVector.begin(), hostVector.end() );

		ecuda::vector<Coordinate> deviceVector1( 200 );
		kernel_linearize<<<1,1>>>( deviceMatrix, deviceVector1 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::vector<Coordinate> hostVector2( 200 );
		deviceVector1 >> hostVector2;
		for( std::size_t i = 0; i < hostVector2.size(); ++i ) {
			std::cout << "LINEAR MATRIX " << hostVector2[i] << std::endl;
		}
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

	{
		std::vector<Coordinate> hostVector( 10*20 );
		unsigned index = 0;
		for( unsigned i = 0; i < 10; ++i ) {
			for( unsigned j = 0; j < 20; ++j, ++index ) {
				hostVector[index] = Coordinate(i,j);
			}
		}
		ecuda::matrix<Coordinate> deviceMatrix( 10, 20 );
		deviceMatrix.assign( hostVector.begin(), hostVector.end() );
		ecuda::vector<Coordinate> deviceVector( 10*20 );
		kernel_linearize<<<1,1>>>( deviceMatrix, deviceVector );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );

		hostVector.clear();
		hostVector.resize( 200 );
		//hostVector.assign( 200, Coordinate() );
		deviceMatrix >> hostVector;
		for( std::size_t i = 0; i < hostVector.size(); ++i ) {
			std::cout << "LINEAR " << hostVector[i] << std::endl;
		}

	}

	return EXIT_SUCCESS;

}
