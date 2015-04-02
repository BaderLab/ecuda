
#include <iostream>
#include <vector>

#include "../include/ecuda/views.hpp"

int main( int argc, char* argv[] ) {

	{
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

		hostVector.assign( n, 0 );
		sequence2 >> hostVector;

		std::cout << "hostVector ="; for( std::size_t i = 0; i < n; ++i ) std::cout << " " << hostVector[i]; std::cout << std::endl;
	}

	{
		const std::size_t w = 20;
		const std::size_t h = 10;
		ecuda::device_pitch_allocator<double> allocator;
		std::size_t pitch;
		ecuda::device_ptr<double> devicePtr1( allocator.allocate( w, h, pitch ) );
		ecuda::padded_ptr<double,double*,1> paddedPtr1( devicePtr1.get(), w, pitch-w*sizeof(double) );
		ecuda::__device_grid< double, ecuda::padded_ptr<double,double*,1> > grid1( paddedPtr1, w, h );

std::cerr << "paddedPtr1( " << devicePtr1.get() << ", " << w << ", pitch=" << pitch << ", " << (pitch-w*sizeof(double)) << ")" << std::endl;
		std::vector< double, ecuda::host_allocator<double> > hostVector( w*h, 99 );
		grid1.assign( hostVector.begin(), hostVector.end() );

		hostVector.assign( w*h, 0 );
		for( std::size_t i = 0; i < h; ++i ) {
			std::cout << "hostVector[" << i << "]";
			for( std::size_t j = 0; j < w; ++j ) std::cout << " " << hostVector[i*w+j];
			std::cout << std::endl;
		}
		std::cout << std::endl;

		grid1 >> hostVector;
		for( std::size_t i = 0; i < h; ++i ) {
			std::cout << "hostVector[" << i << "]";
			for( std::size_t j = 0; j < w; ++j ) std::cout << " " << hostVector[i*w+j];
			std::cout << std::endl;
		}
		std::cout << std::endl;

		ecuda::device_ptr<double> devicePtr2( allocator.allocate( w, h, pitch ) );
		ecuda::padded_ptr<double,double*,1> paddedPtr2( devicePtr2.get(), w, pitch-w*sizeof(double) );
		ecuda::__device_grid< double, ecuda::padded_ptr<double,double*,1> > grid2( paddedPtr2, w, h );
		grid1 >> grid2;

		hostVector.assign( w*h, 0 );
		grid2 >> hostVector;
		for( std::size_t i = 0; i < h; ++i ) {
			std::cout << "hostVector[" << i << "]";
			for( std::size_t j = 0; j < w; ++j ) std::cout << " " << hostVector[i*w+j];
			std::cout << std::endl;
		}
		std::cout << std::endl;

	}

	return EXIT_SUCCESS;

}

