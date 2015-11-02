#include <iostream>
#include <list>
#include <vector>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/models.hpp"

#ifndef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
template<typename T,std::size_t N>
__global__ void testIterators( const ecuda::array<T,N> src, ecuda::array<T,N> dest ) {
	typename ecuda::array<T,N>::iterator result = dest.begin();
	for( typename ecuda::array<T,N>::const_iterator iter = src.begin(); iter != src.end(); ++iter, ++result ) *result = *iter;
}
#endif

int main( int argc, char* argv[] ) {

	std::vector<int> hostVector( 100 );	for( unsigned i = 0; i < 100; ++i ) hostVector[i] = i;

	//ecuda::array<int,100> deviceArray; deviceArray.operator<<( hostVector );
	//if( !ecuda::equal( hostVector.begin(), hostVector.end(), deviceArray.begin() ) ) throw std::runtime_error( "operator<< failed" );

	ecuda::array<int,100> deviceArray;
	ecuda::copy( hostVector.begin(), hostVector.end(), deviceArray.begin() );

	#ifndef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
	{
		ecuda::array<int,100> deviceArray2;
		testIterators<<<1,1>>>( deviceArray, deviceArray2 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::cout << "EQUAL " << ( deviceArray == deviceArray2 ? "true" : "false" ) << std::endl;
		std::cout << "LESSTHAN " << ( deviceArray < deviceArray2 ? "true" : "false" ) << std::endl;
	}
	#endif

	ecuda::reverse( deviceArray.begin(), deviceArray.end() );

	std::cout << "HOST   VECTOR ="; for( unsigned i = 0; i < hostVector.size(); ++i ) std::cout << " " << hostVector[i]; std::cout << std::endl;
	{
		std::vector<int> tmp( 100 );
		ecuda::copy( deviceArray.begin(), deviceArray.end(), tmp.begin() );
		std::cout << "DEVICE VECTOR ="; for( unsigned i = 0; i < tmp.size(); ++i ) std::cout << " " << tmp[i]; std::cout << std::endl;
	}

	//int* p = 0;
	//typename ecuda::pointer_traits<int*>::unmanaged_pointer q = ecuda::pointer_traits<int*>().make_unmanaged(p);
	//typename ecuda::pointer_traits<int*>::unmanaged_pointer r = ecuda::pointer_traits<int*>::cast_unmanaged(q);

	return EXIT_SUCCESS;

}

