#include <iostream>
#include <vector>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/matrix.hpp"
#include "../include/ecuda/cube.hpp"
#include "../include/ecuda/vector.hpp"

template<class Container>
void print_device_container( const Container& c ) {
	std::vector<typename Container::value_type> v( c.size() );
	ecuda::copy( c.begin(), c.end(), v.begin() );
	for( typename std::vector<typename Container::value_type>::const_iterator iter = v.begin(); iter != v.end(); ++iter ) {
		std::cout << " " << *iter;
	}
}

template<class Container>
void print_host_container( const Container& c ) {
	for( typename Container::const_iterator iter = c.begin(); iter != c.end(); ++iter ) {
		std::cout << " " << *iter;
	}
}

int main( int argc, char* argv[] ) {

	const std::size_t N = 20;
	typedef int value_type;

	std::cout << "COPY " << N << " VALUES FROM HOST TO DEVICE" << std::endl;

	std::vector<value_type> hostVector1( N ); for( std::size_t i = 0; i < N; ++i ) hostVector1[i] = i;
	ecuda::array<value_type,N> deviceArray1;

	ecuda::copy( hostVector1.begin(), hostVector1.end(), deviceArray1.begin() );
	std::cout << "HOST VECTOR  :"; print_host_container( hostVector1 ); std::cout << std::endl;
	std::cout << "DEVICE ARRAY :"; print_device_container( deviceArray1 ); std::cout << std::endl;

	return 0;

}
