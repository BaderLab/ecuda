#include <iostream>
#include <vector>

#include "../include/ecuda/array.hpp"
#include "../include/ecuda/vector.hpp"
#include "../include/ecuda/matrix.hpp"
#include "../include/ecuda/cube.hpp"

template<typename Container>
void print_sequence( const Container& container ) {
	std::vector<typename Container::value_type> vec( container.size() );
	container >> vec;
	std::cout << "SEQUENCE";
	for( typename std::vector<typename Container::value_type>::const_iterator iter = vec.begin(); iter != vec.end(); ++iter ) {
		std::cout << " " << *iter;
	}
	std::cout << std::endl;
}

struct coord2d {
	double x, y;
	coord2d( const double x = 0, const double y = 0 ) : x(x), y(y) {}
	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const coord2d& item ) {
		out << "[" << item.x << "," << item.y << "]";
		return out;
	}
};

struct coord3d : coord2d {
	double z;
	coord3d( const double x = 0, const double y = 0, const double z = 0 ) : coord2d(x,y), z(z) {}
	template<typename U,typename V>
	friend std::basic_ostream<U,V>& operator<<( std::basic_ostream<U,V>& out, const coord3d& item ) {
		out << "[" << item.x << "," << item.y << "," << item.z << "]";
		return out;
	}
};

int main( int argc, char* argv[] ) {

	std::vector<coord2d> hostVec1( 3*5 );
	for( unsigned i = 0; i < 3; ++i )
		for( unsigned j = 0; j < 5; ++j )
			hostVec1[i*5+j] = coord2d(i,j);

	ecuda::matrix<coord2d> mat1( 3, 5 );
	mat1.assign( hostVec1.begin(), hostVec1.end() );
	print_sequence( mat1 );

	mat1.get_row(1).assign( hostVec1.begin(), hostVec1.begin()+5 );
	print_sequence( mat1 );

	// should fail to compile
	//mat1.get_column(1).assign( hostVec1.begin(), hostVec1.begin()+5 );

	mat1.get_row(1).assign( mat1.get_row(2).begin(), mat1.get_row(2).end() );
	print_sequence( mat1 );

	return EXIT_SUCCESS;

}

