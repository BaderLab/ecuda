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

template<typename T,typename U,typename V>
std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const ecuda::matrix<T>& matrix ) {
	out << "MATRIX[" << matrix.number_rows() << " x " << matrix.number_columns() << "]" << std::endl;
	std::vector<T> hostVector( matrix.size() );
	matrix >> hostVector;
	for( typename ecuda::matrix<T>::size_type i = 0; i < matrix.number_rows(); ++i ) {
		for( typename ecuda::matrix<T>::size_type j = 0; j < matrix.number_columns(); ++j ) {
			out << " " << hostVector[i*matrix.number_columns()+j];
		}
		out << std::endl;
	}
	return out;
}

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
	{
		std::cout << "  Constructing matrix of size 10x20 with source data from std::vector..." << std::endl;
		std::vector<Coordinate> hostCoordinates; hostCoordinates.reserve( 10*20 );
		for( unsigned i = 0; i < 10; ++i )
			for( unsigned j = 0; j < 20; ++j )
				hostCoordinates.push_back( Coordinate(i,j) );
		std::cout << "  Results of transfer host => device => host:" << std::endl;
		ecuda::matrix<Coordinate> deviceMatrix( 10, 20 );
		deviceMatrix.assign( hostCoordinates.begin(), hostCoordinates.end() );
		std::cout << deviceMatrix << std::endl;

		ecuda::vector<Coordinate> deviceVector( deviceMatrix.size() );
		deviceMatrix >> deviceVector;
		deviceMatrix.assign( deviceVector.begin(), deviceVector.end() );
		std::cout << "  Results of transfer device => device => device => host: " << std::endl;
		std::cout << deviceMatrix << std::endl;
	}

	return EXIT_SUCCESS;

}
