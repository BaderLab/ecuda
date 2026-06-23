#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES // prevent CHECK and FAIL namespace collision
#include <doctest.h>

#include <iostream>
#include <list>
#include <vector>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/event.hpp"

#ifdef __CUDACC__
template<typename T,std::size_t N>
__global__ void kernel_test_iterators( const typename ecuda::array<T,N>::kernel_argument src, typename ecuda::array<T,N>::kernel_argument dest )
{
	ecuda::copy( src.begin(), src.end(), dest.begin() );
	//typename ecuda::array<T,N>::iterator result = dest.begin();
	//for( typename ecuda::array<T,N>::const_iterator iter = src.begin(); iter != src.end(); ++iter, ++result ) *result = *iter;
}
#endif

constexpr std::size_t N = 1000;

DOCTEST_TEST_CASE( "ctor1" )
{
	ecuda::array<double,N> deviceArray;
	std::vector<double> hostVector( N );
	DOCTEST_REQUIRE( ecuda::equal( deviceArray.begin(), deviceArray.end(), hostVector.begin() ) );
}

DOCTEST_TEST_CASE( "ctor2" )
{
	ecuda::array<double,N> deviceArray1;
	ecuda::fill( deviceArray1.begin(), deviceArray1.end(), 99.0 );
	ecuda::array<double,N> deviceArray2( deviceArray1 );
	std::vector<double> hostVector( N, 99.0 );
	DOCTEST_REQUIRE( ecuda::equal( deviceArray2.begin(), deviceArray2.end(), hostVector.begin() ) );
}

DOCTEST_TEST_CASE( "ctor3" )
{
	ecuda::array<double,N> deviceArray1;
	ecuda::fill( deviceArray1.begin(), deviceArray1.end(), 99.0 );
	ecuda::array<double,N> deviceArray2( std::move(deviceArray1) );
	std::vector<double> hostVector( N, 99.0 );
	DOCTEST_REQUIRE( ecuda::equal( deviceArray2.begin(), deviceArray2.end(), hostVector.begin() ) );
	DOCTEST_REQUIRE( !deviceArray1.data() ); // source now empty
}

DOCTEST_TEST_CASE( "ctor4" )
{
	ecuda::array<double,N> deviceArray1;
	ecuda::fill( deviceArray1.begin(), deviceArray1.end(), 99.0 );
	ecuda::array<double,N> deviceArray2 = deviceArray1;
	ecuda::fill( deviceArray1.begin(), deviceArray1.end(), 0.0 );
	std::vector<double> hostVector( N, 99.0 );
	DOCTEST_REQUIRE( ecuda::equal( deviceArray2.begin(), deviceArray2.end(), hostVector.begin() ) );
	DOCTEST_REQUIRE( !ecuda::equal( deviceArray1.begin(), deviceArray1.end(), hostVector.begin() ) );
}

DOCTEST_TEST_CASE( "ctor5" )
{
	ecuda::array<double,N> deviceArray1;
	ecuda::fill( deviceArray1.begin(), deviceArray1.end(), 99.0 );
	ecuda::array<double,N> deviceArray2 = std::move(deviceArray1);
	std::vector<double> hostVector( N, 99.0 );
	DOCTEST_REQUIRE( ecuda::equal( deviceArray2.begin(), deviceArray2.end(), hostVector.begin() ) );
	DOCTEST_REQUIRE( !deviceArray1.data() );
}

DOCTEST_TEST_CASE( "accessors" )
{
	ecuda::array<double,N> deviceArray;
	#ifdef __CUDACC__
	//ECUDA_STATIC_ASSERT(false,MUST_IMPLEMENT_ACCESSOR_AS_KERNEL);
	#else
	for( typename ecuda::array<double,N>::size_type i = 0; i < deviceArray.size(); ++i ) deviceArray[i] = static_cast<double>(i);
	DOCTEST_REQUIRE_THROWS_AS( deviceArray.at(N), std::out_of_range );
	DOCTEST_REQUIRE( deviceArray[10] == double(10.0) );
	DOCTEST_REQUIRE( deviceArray.front() == double(0.0) );
	DOCTEST_REQUIRE( deviceArray.back() == double(N-1) );
	DOCTEST_REQUIRE( deviceArray.data().get() != nullptr );
	DOCTEST_REQUIRE( !deviceArray.empty() );
	DOCTEST_REQUIRE( deviceArray.size() == N );
	#endif
}

DOCTEST_TEST_CASE( "iterators" )
{
	ecuda::array<double,N> deviceArray;
	DOCTEST_REQUIRE( ( deviceArray.end() - deviceArray.begin() ) == N );
	DOCTEST_REQUIRE( ( deviceArray.rbegin() - deviceArray.rend() ) == N );
}

DOCTEST_TEST_CASE( "transforms" )
{
	ecuda::array<double,N> deviceArray1;
	deviceArray1.fill( static_cast<double>(99) );
	std::vector<double> hostVector1( N, static_cast<double>(99) );
	DOCTEST_REQUIRE( ecuda::equal( deviceArray1.begin(), deviceArray1.end(), hostVector1.begin() ) );

	ecuda::array<double,N> deviceArray2;
	deviceArray2.fill( static_cast<double>(66) );
	deviceArray1.swap( deviceArray2 );
	DOCTEST_REQUIRE( ecuda::equal( deviceArray2.begin(), deviceArray2.end(), hostVector1.begin() ) );
}

DOCTEST_TEST_CASE( "kernels" )
{
	// std::vector<int> hostVector( N ); for( int i = 0; i < static_cast<int>(N); ++i ) hostVector[i] = i;
	// ecuda::array<int,N> deviceArray;
	// ecuda::copy( hostVector.begin(), hostVector.end(), deviceArray.begin() );

	// #ifdef __CUDACC__
	// {
	// 	std::cout << "TESTING KERNELS" << std::endl;
	// 	std::cout << "---------------" << std::endl;
	// 	{
	// 		ecuda::array<int,N> deviceArray2;
	// 		CUDA_CALL_KERNEL_AND_WAIT( kernel_test_iterators<int,N><<<1,1>>>( deviceArray, deviceArray2 ) );
	// 		//CUDA_CHECK_ERRORS();
	// 		//CUDA_CALL( cudaDeviceSynchronize() );
	// 		std::cout << "ecuda::array::iterator : " << std::boolalpha << ecuda::equal( deviceArray.begin(), deviceArray.end(), deviceArray2.begin() ) << std::endl;
	// 	}
	// 	std::cout << std::endl;
	// }
	// #endif

	// ecuda::reverse( deviceArray.begin(), deviceArray.end() );

	// std::cout << "HOST   VECTOR ="; for( unsigned i = 0; i < hostVector.size(); ++i ) std::cout << " " << hostVector[i]; std::cout << std::endl;
	// {
	// 	std::vector<int> tmp( N );
	// 	ecuda::copy( deviceArray.begin(), deviceArray.end(), tmp.begin() );
	// 	std::cout << "DEVICE VECTOR ="; for( unsigned i = 0; i < tmp.size(); ++i ) std::cout << " " << tmp[i]; std::cout << std::endl;
	// }

}
