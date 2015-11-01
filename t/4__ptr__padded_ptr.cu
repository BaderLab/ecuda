#include <iostream>
#include <vector>

#include "../include/ecuda/memory.hpp"

int main( int argc, char* argv[] ) {

	std::vector<int> v( 100 );
	for( std::size_t i = 0; i < v.size(); ++i ) v[i] = i;

	ecuda::padded_ptr<int> p( &v.front(), sizeof(int)*10, 8, &v.front() );
	for( std::size_t i = 0; i < 10; ++i, ++p ) std::cout << "padded_ptr[" << p.get() << "]=" << (p.get()-&v.front()) << std::endl;

	return 0;

}

