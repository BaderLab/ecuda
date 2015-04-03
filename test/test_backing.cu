
#include <iostream>
#include <vector>

#include "../include/ecuda/views.hpp"

int main( int argc, char* argv[] ) {

	{
		const std::size_t n = 100;

		ecuda::device_allocator<double> allocator;

		//ecuda::device_ptr<double> devicePtr1( allocator.allocate( n ) );
		ecuda::__device_sequence< double, ecuda::device_ptr<double> > sequence1( ecuda::device_ptr<double>( allocator.allocate(n) ), n );

		std::vector< double, ecuda::host_allocator<double> > hostVector( n, 99 );
		sequence1.assign( hostVector.begin(), hostVector.end() );

		hostVector.assign( n, 0 );
		std::cout << "hostVector ="; for( std::size_t i = 0; i < n; ++i ) std::cout << " " << hostVector[i]; std::cout << std::endl;
		sequence1 >> hostVector;

		std::cout << "hostVector ="; for( std::size_t i = 0; i < n; ++i ) std::cout << " " << hostVector[i]; std::cout << std::endl;

		ecuda::__device_sequence< double, ecuda::device_ptr<double> > sequence2( ecuda::device_ptr<double>( allocator.allocate(n) ), n );
		sequence1 >> sequence2;

		hostVector.assign( n, 0 );
		sequence2 >> hostVector;

		std::cout << "hostVector ="; for( std::size_t i = 0; i < n; ++i ) std::cout << " " << hostVector[i]; std::cout << std::endl;
	}

	{
		const std::size_t w = 20;
		const std::size_t h = 10;

		ecuda::device_pitch_allocator<double> allocator;
		ecuda::padded_ptr<double,double*,1> paddedPtr( allocator.allocate( w, h ) );
		ecuda::device_ptr< double, ecuda::padded_ptr<double,double*,1> > devicePtr1( paddedPtr );
		//ecuda::__device_grid< double, ecuda::padded_ptr<double,double*,1> > grid1( devicePtr1.get(), h, w );
		ecuda::__device_grid< double, ecuda::device_ptr< double, ecuda::padded_ptr<double,double*,1> > > grid1( devicePtr1, h, w );

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

		ecuda::device_ptr< double, ecuda::padded_ptr<double,double*,1> > devicePtr2( allocator.allocate( w, h ) );
		//ecuda::__device_grid< double, ecuda::padded_ptr<double,double*,1> > grid2( devicePtr2.get(), h, w );
		ecuda::__device_grid< double, ecuda::device_ptr< double, ecuda::padded_ptr<double,double*,1> > > grid2( devicePtr2, h, w );
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

