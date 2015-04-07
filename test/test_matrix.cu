//#define NDEBUG
//#include <cassert>

#include <iostream>
#include <cstdio>
#include <vector>
#include <estd/matrix.hpp>
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/matrix.hpp"

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

typedef unsigned char uint8_t;

int main( int argc, char* argv[] ) {

	std::cout << "Testing ecuda::matrix..." << std::endl;

	std::cout << "Testing ctors..." << std::endl;
	{
		std::cout << "  Constructing default empty matrix..." << std::endl;
		ecuda::matrix<Coordinate> emptyDeviceMatrix;
		std::cout << "    Matrix evaluates as empty: " << ( emptyDeviceMatrix.empty() ? "YES" : "NO" ) << std::endl;
		std::cout << "    Underlying pointer evaluates as NULL: " << ( (Coordinate*)emptyDeviceMatrix.data() == NULL ? "YES" : "NO" ) << std::endl;
	}
	{
		std::cout << "  Constructing matrix of size 10x20 with default value=[66,66]..." << std::endl;
		ecuda::matrix<Coordinate> deviceMatrix( 10, 20, Coordinate(66,66) );
		std::cout << "    Matrix evaluates as 10x20: " << ( deviceMatrix.number_rows() == 10 and deviceMatrix.number_columns() == 20 ? "YES" : "NO" ) << std::endl;
		std::cout << "    Underlying pointer evaluates as non-NULL: " << ( (Coordinate*)deviceMatrix.data() ? "YES" : "NO" ) << std::endl;
	}

	return EXIT_SUCCESS;

}
