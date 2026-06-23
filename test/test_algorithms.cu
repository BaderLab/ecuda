#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES // prevent CHECK and FAIL namespace collision
#include <doctest.h>

#include <iomanip>
#include <iostream>
#include <list>
#include <tuple>
#include <vector>

#include <estd/algorithm.hpp>
#include <estd/matrix.hpp>
#include <estd/cube.hpp>

#include "../include/ecuda/ecuda.hpp"

constexpr std::size_t R = 5;
constexpr std::size_t C = 5;
constexpr std::size_t D = 5;

template<typename T,std::size_t N>
std::vector<T>
MakeHostVector()
{
	std::vector<T> v( N );
	estd::fill_strictly_increasing( v.begin(), v.end(), 0, 1 );
	return v;
}

template<typename T,std::size_t R,std::size_t C>
estd::matrix<std::pair<T,T>>
MakeHostMatrix()
{
	estd::matrix<std::pair<T,T>> mat( R, C );
	for( std::size_t i = 0; i < R; ++i ) {
		for( std::size_t j = 0; j < C; ++j ) {
			mat(i,j) = std::make_pair(i,j);
		}
	}
	return mat;
}

template<typename T,std::size_t R,std::size_t C,std::size_t D>
estd::cube<std::tuple<T,T,T>>
MakeHostCube()
{
	estd::cube<std::tuple<T,T,T>> cube( R, C, D );
	for( std::size_t i = 0; i < R; ++i ) {
		for( std::size_t j = 0; j < C; ++j ) {
			for( std::size_t k = 0; k < D; ++k ) {
				cube(i,j,k) = std::make_tuple(i,j,k);
			}
		}
	}
	return cube;
}

DOCTEST_TEST_CASE( "array" )
{
	typedef double value_type;
	value_type hostArray[R];
	ecuda::array<value_type,R> deviceArray;

}

DOCTEST_TEST_CASE( "vector" )
{
	typedef double value_type;

	auto hostVector = MakeHostVector<value_type,R>();
	ecuda::vector<value_type> deviceVector( R );
	ecuda::copy( hostVector.begin(), hostVector.end(), deviceVector.begin() );
	DOCTEST_REQUIRE( ecuda::equal( hostVector.begin(), hostVector.end(), deviceVector.begin() ) );
	DOCTEST_REQUIRE( ecuda::equal( deviceVector.begin(), deviceVector.end(), hostVector.begin() ) );

	DOCTEST_REQUIRE( ecuda::distance( hostVector.begin(), ecuda::find( hostVector.begin(), hostVector.end(), 2 ) ) == 2 );
	DOCTEST_REQUIRE( ecuda::distance( deviceVector.begin(), ecuda::find( deviceVector.begin(), deviceVector.end(), 2 ) ) == 2 );

	DOCTEST_REQUIRE( ecuda::distance( hostVector.begin(), ecuda::find_if( hostVector.begin(), hostVector.end(), []( value_type value ) { return value == 3; } ) ) == 3 );
	DOCTEST_REQUIRE( ecuda::distance( deviceVector.begin(), ecuda::find_if( deviceVector.begin(), deviceVector.end(), []( value_type value ) { return value == 3; } ) ) == 3 );

	DOCTEST_REQUIRE( ecuda::any_of( hostVector.begin(), hostVector.end(), []( value_type value ) { return value == 1; } ) );
	DOCTEST_REQUIRE( ecuda::any_of( deviceVector.begin(), deviceVector.end(), []( value_type value ) { return value == 1; } ) );
	DOCTEST_REQUIRE( !ecuda::any_of( hostVector.begin(), hostVector.end(), []( value_type value ) { return value == 20; } ) );
	DOCTEST_REQUIRE( !ecuda::any_of( deviceVector.begin(), deviceVector.end(), []( value_type value ) { return value == 20; } ) );

	DOCTEST_REQUIRE( ecuda::none_of( hostVector.begin(), hostVector.end(), []( value_type value ) { return value == 20; } ) );
	DOCTEST_REQUIRE( ecuda::none_of( deviceVector.begin(), deviceVector.end(), []( value_type value ) { return value == 20; } ) );
	DOCTEST_REQUIRE( !ecuda::none_of( hostVector.begin(), hostVector.end(), []( value_type value ) { return value == 1; } ) );
	DOCTEST_REQUIRE( !ecuda::none_of( deviceVector.begin(), deviceVector.end(), []( value_type value ) { return value == 1; } ) );

	ecuda::reverse( hostVector.begin(), hostVector.end() );
	DOCTEST_REQUIRE( ecuda::equal( hostVector.rbegin(), hostVector.rend(), deviceVector.begin() ) );
	// DOCTEST_REQUIRE( ecuda::equal( deviceVector.rbegin(), deviceVector.rend(), hostVector.begin() ) );

	DOCTEST_REQUIRE( !ecuda::lexicographical_compare( hostVector.rbegin(), hostVector.rend(), deviceVector.begin(), deviceVector.end() ) );
	DOCTEST_REQUIRE( !ecuda::lexicographical_compare( deviceVector.begin(), deviceVector.end(), hostVector.rbegin(), hostVector.rend() ) );
	// DOCTEST_REQUIRE( ecuda::lexicographical_compare( deviceVector.rbegin(), deviceVector.rend(), hostVector.begin(), hostVector.end() ) );

	DOCTEST_REQUIRE( !ecuda::all_of( hostVector.begin(), hostVector.end(), []( value_type value ) { return value == 20; } ) );
	DOCTEST_REQUIRE( !ecuda::all_of( deviceVector.begin(), deviceVector.end(), []( value_type value ) { return value == 20; } ) );
	ecuda::fill( hostVector.begin(), hostVector.end(), value_type(20.0) );
	DOCTEST_REQUIRE( !ecuda::equal( hostVector.begin(), hostVector.end(), deviceVector.begin() ) );
	DOCTEST_REQUIRE( !ecuda::equal( deviceVector.begin(), deviceVector.end(), hostVector.begin() ) );
	ecuda::fill( deviceVector.begin(), deviceVector.end(), value_type(20.0) );
	DOCTEST_REQUIRE( ecuda::equal( hostVector.begin(), hostVector.end(), deviceVector.begin() ) );
	DOCTEST_REQUIRE( ecuda::equal( deviceVector.begin(), deviceVector.end(), hostVector.begin() ) );
	DOCTEST_REQUIRE( ecuda::all_of( hostVector.begin(), hostVector.end(), []( value_type value ) { return value == 20; } ) );
	DOCTEST_REQUIRE( ecuda::all_of( deviceVector.begin(), deviceVector.end(), []( value_type value ) { return value == 20; } ) );

	ecuda::for_each( hostVector.begin(), hostVector.end(), []( value_type& value ) { value *= 2.0; } );
	struct DOUBLE_VALUE
	{
		__HOST__ __DEVICE__ void operator()( value_type& t ) { t *= 2; }
	};
	ecuda::for_each( deviceVector.begin(), deviceVector.end(), DOUBLE_VALUE() );
	DOCTEST_REQUIRE( ecuda::all_of( hostVector.begin(), hostVector.end(), []( value_type value ) { return value == 40; } ) );
	// DOCTEST_REQUIRE( ecuda::all_of( deviceVector.begin(), deviceVector.end(), []( value_type value ) { return value == 40; } ) );

	DOCTEST_REQUIRE( ecuda::count( hostVector.begin(), hostVector.end(), 40.0 ) == hostVector.size() );
	DOCTEST_REQUIRE( ecuda::count( deviceVector.begin(), deviceVector.end(), 40.0 ) == deviceVector.size() );
	DOCTEST_REQUIRE( ecuda::count( hostVector.begin(), hostVector.end(), 20.0 ) == 0 );
	DOCTEST_REQUIRE( ecuda::count( deviceVector.begin(), deviceVector.end(), 20.0 ) == 0 );

	DOCTEST_REQUIRE( ecuda::count_if( hostVector.begin(), hostVector.end(), []( const value_type& value ) { return value == 40.0; } ) == hostVector.size() );
	struct CHECK_VALUE
	{
		value_type value;
		__HOST__ __DEVICE__
		bool operator()( const value_type& value ) { return value == this->value; }
	};
	DOCTEST_REQUIRE( ecuda::count_if( deviceVector.begin(), deviceVector.end(), CHECK_VALUE{40.0} ) == deviceVector.size() );

	auto pr = ecuda::mismatch( hostVector.begin(), hostVector.end(), deviceVector.begin() );
	DOCTEST_REQUIRE( pr.first == hostVector.end() );
	DOCTEST_REQUIRE( pr.second == deviceVector.end() );

}


DOCTEST_TEST_CASE( "matrix" )
{
	typedef double value_type;
	auto hostMatrix = MakeHostMatrix<value_type,R,C>();
	ecuda::matrix<std::pair<value_type,value_type>> deviceMatrix( R, C );
	// TODO...
}

DOCTEST_TEST_CASE( "cube" )
{
	typedef double value_type;
	auto hostCube = MakeHostCube<value_type,R,C,D>();
	ecuda::cube<std::tuple<value_type,value_type,value_type>> deviceCube( R, C, D );
	// TODO...

}
