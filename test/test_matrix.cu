#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES // prevent CHECK and FAIL namespace collision
#include <doctest.h>

#include <iostream>
#include <list>
#include <tuple>
#include <vector>

#include <estd/matrix.hpp>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/matrix.hpp"

#ifdef __CUDACC__

// template<typename T>
// __global__
// void GrabXY( const std::size_t z, const typename ecuda::cube<T>::const_kernel_argument src, typename ecuda::matrix<T>::kernel_argument dest )
// {
// 	const auto x = blockIdx.x * blockDim.x + threadIdx.x;
// 	const auto y = blockIdx.y * blockDim.y + threadIdx.y;
// 	if( x < src.number_rows() && y < src.number_columns() ) {
// 		dest(x,y) = src(x,y,z);
// 	}
// }

#endif

DOCTEST_TEST_CASE( "matrix" )
{
	constexpr std::size_t R = 3;
	constexpr std::size_t C = 5;
	typedef int value_type;
	struct coord_type
	{
		value_type x, y;
		__HOST__ __DEVICE__ bool operator==( const coord_type& other ) const { return x == other.x && y == other.y; }
	};
	// typedef std::tuple<value_type,value_type,value_type> coord_type;

	estd::matrix<coord_type> hostMatrix( R, C );
	for( std::size_t i = 0; i < R; ++i ) {
		for( std::size_t j = 0; j < C; ++j ) {
			hostMatrix(i,j) = coord_type{value_type(i),value_type(j)};
		}
	}

	ecuda::matrix<coord_type> deviceMatrix( R, C );
	ecuda::copy( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );

	DOCTEST_REQUIRE( deviceMatrix.number_rows() == R );
	DOCTEST_REQUIRE( deviceMatrix.number_columns() == C );
	DOCTEST_REQUIRE( ecuda::equal( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() ) );
	DOCTEST_REQUIRE( ecuda::equal( deviceMatrix.begin(), deviceMatrix.end(), hostMatrix.begin() ) );

	std::fill( hostMatrix.begin(), hostMatrix.end(), coord_type{0,0} );
	DOCTEST_REQUIRE( !ecuda::equal( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() ) );
	DOCTEST_REQUIRE( !ecuda::equal( deviceMatrix.begin(), deviceMatrix.end(), hostMatrix.begin() ) );

	ecuda::copy( deviceMatrix.begin(), deviceMatrix.end(), hostMatrix.begin() );
	DOCTEST_REQUIRE( ecuda::equal( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() ) );
	DOCTEST_REQUIRE( ecuda::equal( deviceMatrix.begin(), deviceMatrix.end(), hostMatrix.begin() ) );

	// TODO: add some kernel tests

}
