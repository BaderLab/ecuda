#include <iostream>
#include <vector>

#include "../include/ecuda/array.hpp"
#include "../include/ecuda/vector.hpp"
#include "../include/ecuda/matrix.hpp"
#include "../include/ecuda/cube.hpp"

template<typename Iterator>
void print_sequence( Iterator first, Iterator last ) {
	std::cout << "SEQUENCE";
	while( first != last ) {
		std::cout << " " << *first;
		++first;
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
	print_sequence( arr.begin(), arr.end() );

	ecuda::vector<double> vec1;
	print_sequence( vec1.begin(), vec1.end() );

	ecuda::vector<double> vec2( 10 );
	print_sequence( vec2.begin(), vec2.end() );

	ecuda::vector<double> vec3( 10, 99 );
	print_sequence( vec3.begin(), vec3.end() );

	ecuda::vector<double> vec4( vec3.begin(), vec3.end() );
	print_sequence( vec4.begin(), vec4.end() );

	ecuda::vector<double> vec5( vec3 );
	print_sequence( vec5.begin(), vec5.end() );

	ecuda::matrix<coord2d> mat1;
	print_sequence( mat1.begin(), mat1.end() );

	ecuda::matrix<coord2d> mat2( 2, 5, coord2d(66,99) );
	print_sequence( mat2.begin(), mat2.end() );

	ecuda::cube<coord3d> cube1;
	print_sequence( cube1.begin(), cube1.end() );

	ecuda::cube<coord3d> cube2( 2, 3, 4, coord3d(66,99) );
	print_sequence( cube2.begin(), cube2.end() );

	return EXIT_SUCCESS;

}

