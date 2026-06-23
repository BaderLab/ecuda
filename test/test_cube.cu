#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES // prevent CHECK and FAIL namespace collision
#include <doctest.h>

#include <iostream>
#include <list>
#include <tuple>
#include <vector>

#include <estd/cube.hpp>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/cube.hpp"

#ifdef __CUDACC__
// template<typename T>
// __global__ void testIterators( const ecuda::cube<T> src, ecuda::cube<T> dest ) {
// 	typename ecuda::cube<T>::iterator result = dest.begin();
// 	//typename ecuda::matrix<T>::const_iterator result2 = result;
// 	for( typename ecuda::cube<T>::const_iterator iter = src.begin(); iter != src.end(); ++iter, ++result ) *result = *iter;
// }

// template<typename T>
// __global__ void testIterators2( const ecuda::cube<T> src, ecuda::cube<T> dest ) {
// 	for( typename ecuda::cube<T>::size_type i = 0; i < src.number_rows(); ++i ) {
// 		for( typename ecuda::cube<T>::size_type j = 0; j < src.number_columns(); ++j ) {
// 			typename ecuda::cube<T>::const_depth_type srcDepth = src.get_depth(i,j);
// 			typename ecuda::cube<T>::depth_type destDepth = dest.get_depth(i,j);
// 			ecuda::copy( srcDepth.begin(), srcDepth.end(), destDepth.begin() );
// 		}
// 	}
// }

template<typename T>
__global__
void GrabXY( const std::size_t z, const typename ecuda::cube<T>::const_kernel_argument src, typename ecuda::matrix<T>::kernel_argument dest )
{
	const auto x = blockIdx.x * blockDim.x + threadIdx.x;
	const auto y = blockIdx.y * blockDim.y + threadIdx.y;
	if( x < src.number_rows() && y < src.number_columns() ) {
		dest(x,y) = src(x,y,z);
	}
}

template<typename T>
__global__
void GrabYZ( const std::size_t x, const typename ecuda::cube<T>::const_kernel_argument src, typename ecuda::matrix<T>::kernel_argument dest )
{
	const auto y = blockIdx.x * blockDim.x + threadIdx.x;
	const auto z = blockIdx.y * blockDim.y + threadIdx.y;
	if( y < src.number_columns() && z < src.number_depths() ) {
		dest(y,z) = src(x,y,z);
	}
}

template<typename T>
__global__
void GrabZX( const std::size_t y, const typename ecuda::cube<T>::const_kernel_argument src, typename ecuda::matrix<T>::kernel_argument dest )
{
	const auto z = blockIdx.x * blockDim.x + threadIdx.x;
	const auto x = blockIdx.y * blockDim.y + threadIdx.y;
	if( z < src.number_depths() && x < src.number_rows() ) {
		dest(z,x) = src(x,y,z);
	}
}

#endif

DOCTEST_TEST_CASE( "cube" )
{
	constexpr std::size_t R = 3;
	constexpr std::size_t C = 5;
	constexpr std::size_t D = 6;
	typedef int value_type;
	struct coord_type
	{
		value_type x, y, z;
		__HOST__ __DEVICE__ bool operator==( const coord_type& other ) const { return x == other.x && y == other.y && z == other.z; }
	};
	// typedef std::tuple<value_type,value_type,value_type> coord_type;

	estd::cube<coord_type> hostCube( R, C, D );
	for( std::size_t i = 0; i < R; ++i ) {
		for( std::size_t j = 0; j < C; ++j ) {
			for( std::size_t k = 0; k < D; ++k ) {
				hostCube(i,j,k) = coord_type{value_type(i),value_type(j),value_type(k)};
			}
		}
	}

	ecuda::cube<coord_type> deviceCube( R, C, D );
	ecuda::copy( hostCube.begin(), hostCube.end(), deviceCube.begin() );

	DOCTEST_REQUIRE( deviceCube.number_rows() == R );
	DOCTEST_REQUIRE( deviceCube.number_columns() == C );
	DOCTEST_REQUIRE( deviceCube.number_depths() == D );
	DOCTEST_REQUIRE( ecuda::equal( hostCube.begin(), hostCube.end(), deviceCube.begin() ) );
	DOCTEST_REQUIRE( ecuda::equal( deviceCube.begin(), deviceCube.end(), hostCube.begin() ) );

	std::fill( hostCube.begin(), hostCube.end(), coord_type{0,0,0} );
	DOCTEST_REQUIRE( !ecuda::equal( hostCube.begin(), hostCube.end(), deviceCube.begin() ) );
	DOCTEST_REQUIRE( !ecuda::equal( deviceCube.begin(), deviceCube.end(), hostCube.begin() ) );

	ecuda::copy( deviceCube.begin(), deviceCube.end(), hostCube.begin() );
	DOCTEST_REQUIRE( ecuda::equal( hostCube.begin(), hostCube.end(), deviceCube.begin() ) );
	DOCTEST_REQUIRE( ecuda::equal( deviceCube.begin(), deviceCube.end(), hostCube.begin() ) );

	// TODO: add some kernel tests

}
