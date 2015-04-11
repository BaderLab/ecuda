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

	ecuda::array<double,10> arr;
	print_sequence( arr );

	ecuda::vector<double> vec1;
	print_sequence( vec1 );

	ecuda::vector<double> vec2( 10 );
	print_sequence( vec2 );

	ecuda::vector<double> vec3( 10, 99 );
	print_sequence( vec3 );

	ecuda::vector<double> vec4( vec3.begin(), vec3.end() );
	print_sequence( vec4 );

	ecuda::vector<double> vec5( vec3 );
	print_sequence( vec5 );

	ecuda::matrix<coord2d> mat1;
	print_sequence( mat1 );

	ecuda::matrix<coord2d> mat2( 2, 5, coord2d(66,99) );
	print_sequence( mat2 );

	ecuda::cube<coord3d> cube1;
	print_sequence( cube1 );

	ecuda::cube<coord3d> cube2( 2, 3, 4, coord3d(66,99) );
	print_sequence( cube2 );

	return EXIT_SUCCESS;

}

