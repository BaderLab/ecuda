#include <iostream>
#include <cstdio>
#include <estd/cube.hpp>
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/cube.hpp"

template<typename T>
struct coord_t {
	T x, y, z;
	coord_t( const T& x = T(), const T& y = T(), const T& z = T() ) : x(x), y(y), z(z) {}
	bool operator==( const coord_t& other ) const { return x == other.x and y == other.y and z == other.z; }
	bool operator!=( const coord_t& other ) const { return !operator==(other); }
	friend std::ostream& operator<<( std::ostream& out, const coord_t& coord ) {
		out << "[" << coord.x << "," << coord.y << "," << coord.z << "]";
		return out;
	}
};

typedef coord_t<double> Coordinate;

typedef unsigned char uint8_t;

template<typename T,std::size_t U> __global__
void fetchRow( const ecuda::cube<T> cube, ecuda::array<T,U> array ) {
	T val = *cube.get_allocator().address( cube.data(), 2, 3, cube.get_pitch() );
                //const_pointer np = allocator.address( deviceMemory.get(), columnIndex, depthIndex, pitch );
	printf( "start=[%.05f %.05f %.05f]\n", val.x, val.y, val.z );
	typename ecuda::cube<T>::const_row_type row = cube.get_row( 2, 3 );
	for( typename ecuda::cube<T>::const_row_type::size_type i = 0; i < row.size(); ++i ) array[i] = row[i];
}

int main( int argc, char* argv[] ) {

	estd::cube<Coordinate> hostCube( 3, 4, 5 );
	for( estd::cube<Coordinate>::size_type i = 0; i < hostCube.row_size(); ++i ) {
		for( estd::cube<Coordinate>::size_type j = 0; j < hostCube.column_size(); ++j ) {
			for( estd::cube<Coordinate>::size_type k = 0; k < hostCube.depth_size(); ++k ) {
				hostCube[i][j][k] = Coordinate(i,j,k);
			}
		}
	}

	ecuda::cube<Coordinate> deviceCube( 3, 4, 5 );
	deviceCube << hostCube;

	std::cout << "(1,2,3)=" << hostCube[1][2][3] << std::endl;
	deviceCube >> hostCube;
	std::cout << "(1,2,3)=" << hostCube[1][2][3] << std::endl;

	std::cout << "sizeof(Coordinate)=" << sizeof(Coordinate) << std::endl;

	ecuda::array<Coordinate,3> deviceRow;
	fetchRow<<<1,1>>>( deviceCube, deviceRow );
	CUDA_CHECK_ERRORS();
	CUDA_CALL( cudaDeviceSynchronize() );
	std::vector<Coordinate> hostRow;
	deviceRow >> hostRow;
	std::cout << "ROW";
	for( std::vector<Coordinate>::size_type i = 0; i < hostRow.size(); ++i ) std::cout << hostRow[i];
	std::cout << std::endl;

	return EXIT_SUCCESS;

}

