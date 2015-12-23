#include <getopt.h>
#include <iostream>
#include "../include/ecuda/device.hpp"

int main( int argc, char* argv[] )
{

	int deviceCount = ecuda::device::getDeviceCount();
	for( int i = 0; i < deviceCount; ++i ) {
		std::cout << "DEVICE #" << i << std::endl;
		ecuda::device device( i );
		device.print_summary( std::cout );
		std::cout << std::endl;
	}

	return EXIT_SUCCESS;

}

