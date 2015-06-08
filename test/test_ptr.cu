#include <iostream>

#include "../include/ecuda/device_ptr.hpp"
//#include "../include/ecuda/iterator.hpp"
#include "../include/ecuda/models.hpp"
#include "../include/ecuda/algorithm.hpp"

int main( int argc, char* argv[] ) {

	ecuda::shared_ptr<int> p1( new int );
	ecuda::shared_ptr<double> p2( new double );

	int* ptr = new int;
	ecuda::padded_ptr<int> pp( ptr, 10, 2, ptr );

	std::cout << "sizeof(int)=" << sizeof(int) << std::endl;

	std::cout << pp << " DIFF " << ( reinterpret_cast<char*>(pp.get())-reinterpret_cast<char*>(ptr) ) << std::endl;

	++pp;
	pp += 2;
	std::cout << pp << " DIFF " << ( reinterpret_cast<char*>(pp.get())-reinterpret_cast<char*>(ptr) ) << std::endl;


	if( p1.owner_before( p2 ) ) {
		return 1;
	}

	return 0;

}

