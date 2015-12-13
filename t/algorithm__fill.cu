
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include "../include/ecuda/ecuda.hpp"

typedef double data_type;
const std::size_t N = 100;

SCENARIO( "containers can be filled", "fill" ) {
	const data_type item_value = 66.0;
	GIVEN( "A host and device array" ) {
		data_type hostArray[N];
		ecuda::array<data_type,N> deviceArray;
		WHEN( "the containers are filled with the same value" ) {
			ecuda::fill( hostArray, hostArray+N, item_value );
			ecuda::fill( deviceArray.begin(), deviceArray.end(), item_value );
			THEN( "the host container is equal to the device container" ) { REQUIRE( ecuda::equal( hostArray, hostArray+N, deviceArray.begin() ) ); }
			AND_THEN( "the device container is equal to the host container" ) { REQUIRE( ecuda::equal( deviceArray.begin(), deviceArray.end(), hostArray ) ); }
		}
		AND_WHEN( "one of the values in the host container is changed" ) {
			hostArray[N/2] += 1.0;
			THEN( "the host container is not equal to the device container" ) { REQUIRE( !ecuda::equal( hostArray, hostArray+N, deviceArray.begin() ) ); }
			AND_THEN( "the device container is not equal to the host container" ) { REQUIRE( !ecuda::equal( deviceArray.begin(), deviceArray.end(), hostArray ) ); }
		}
	}
	GIVEN( "A host and device vector" ) {
		std::vector<data_type> hostVector( N );
		ecuda::vector<data_type> deviceVector( N );
		WHEN( "the containers are filled with the same value" ) {
			ecuda::fill( hostVector.begin(), hostVector.end(), item_value );
			ecuda::fill( deviceVector.begin(), deviceVector.end(), item_value );
			THEN( "the host container is equal to the device container" ) { REQUIRE( ecuda::equal( hostVector.begin(), hostVector.end(), deviceVector.begin() ) ); }
			AND_THEN( "the device container is equal to the host container" ) { REQUIRE( ecuda::equal( deviceVector.begin(), deviceVector.end(), hostVector.begin() ) ); }
		}
		AND_WHEN( "one of the values in the host container is changed" ) {
			hostVector[N/2] += 1.0;
			THEN( "the host container is not equal to the device container" ) { REQUIRE( !ecuda::equal( hostVector.begin(), hostVector.end(), deviceVector.begin() ) ); }
			AND_THEN( "the device container is not equal to the host container" ) { REQUIRE( !ecuda::equal( deviceVector.begin(), deviceVector.end(), hostVector.begin() ) ); }
		}
	}
	GIVEN( "A host and device matrix" ) {
		data_type hostMatrix[N][N];
		ecuda::matrix<data_type> deviceMatrix( N, N );
		WHEN( "the containers are filled with the same value" ) {
			ecuda::fill( &hostMatrix[0][0], &hostMatrix[N][0], item_value );
			ecuda::fill( deviceMatrix.begin(), deviceMatrix.end(), item_value );
			THEN( "the host container is equal to the device container" ) { REQUIRE( ecuda::equal( &hostMatrix[0][0], &hostMatrix[N][0], deviceMatrix.begin() ) ); }
			AND_THEN( "the device container is equal to the host container" ) { REQUIRE( ecuda::equal( deviceMatrix.begin(), deviceMatrix.end(), &hostMatrix[0][0] ) ); }
		}
		AND_WHEN( "one of the values in the host container is changed" ) {
			hostMatrix[N/2][N/2] += 1.0;
			THEN( "the host container is not equal to the device container" ) { REQUIRE( !ecuda::equal( &hostMatrix[0][0], &hostMatrix[N][0], deviceMatrix.begin() ) ); }
			AND_THEN( "the device container is not equal to the host container" ) { REQUIRE( !ecuda::equal( deviceMatrix.begin(), deviceMatrix.end(), &hostMatrix[0][0] ) ); }
		}
	}
	GIVEN( "A host and device cube" ) {
		data_type hostCube[N][N][N];
		ecuda::cube<data_type> deviceCube( N, N, N );
		WHEN( "the containers are filled with the same value" ) {
			ecuda::fill( &hostCube[0][0][0], &hostCube[N][0][0], item_value );
			ecuda::fill( deviceCube.begin(), deviceCube.end(), item_value );
			THEN( "the host container is equal to the device container" ) { REQUIRE( ecuda::equal( &hostCube[0][0][0], &hostCube[N][0][0], deviceCube.begin() ) ); }
			AND_THEN( "the device container is equal to the host container" ) { REQUIRE( ecuda::equal( deviceCube.begin(), deviceCube.end(), &hostCube[0][0][0] ) ); }
		}
		AND_WHEN( "one of the values in the host container is changed" ) {
			hostCube[N/2][N/2][N/2] += 1.0;
			THEN( "the host container is not equal to the device container" ) { REQUIRE( !ecuda::equal( &hostCube[0][0][0], &hostCube[N][0][0], deviceCube.begin() ) ); }
			AND_THEN( "the device container is not equal to the host container" ) { REQUIRE( !ecuda::equal( deviceCube.begin(), deviceCube.end(), &hostCube[0][0][0] ) ); }
		}
	}
}


