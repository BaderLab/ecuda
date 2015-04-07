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
		std::cout << "    Underlying pointer evaluates as NULL: " << ( emptyDeviceMatrix.data() ) << std::endl;
	}


	return EXIT_SUCCESS;

}
