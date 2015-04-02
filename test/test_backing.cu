
#include <iostream>
#include <vector>

#include "../include/ecuda/views.hpp"

int main( int argc, char* argv[] ) {

	const std::size_t n = 100;

	ecuda::device_allocator<double> allocator;

	ecuda::device_ptr<double> devicePtr1( allocator.allocate( n ) );
	ecuda::__device_sequence<double> sequence1( devicePtr1.get(), n );

	std::vector< double, ecuda::host_allocator<double> > hostVector( n, 99 );
	sequence1.assign( hostVector.begin(), hostVector.end() );

	hostVector.assign( n, 0 );
	std::cout << "hostVector ="; for( std::size_t i = 0; i < n; ++i ) std::cout << " " << hostVector[i]; std::cout << std::endl;
	sequence1 >> hostVector;

	std::cout << "hostVector ="; for( std::size_t i = 0; i < n; ++i ) std::cout << " " << hostVector[i]; std::cout << std::endl;

	ecuda::device_ptr<double> devicePtr2( allocator.allocate( n ) );
	ecuda::__device_sequence<double> sequence2( devicePtr2.get(), n );
	sequence1 >> sequence2;

	sequence2 >> hostVector;

	std::cout << "hostVector ="; for( std::size_t i = 0; i < n; ++i ) std::cout << " " << hostVector[i]; std::cout << std::endl;

	return EXIT_SUCCESS;

}

