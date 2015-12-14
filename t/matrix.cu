
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <iostream>
#include "../include/ecuda/ecuda.hpp"

template<typename T>
struct element_t
{
	T x, y;
	__HOST__ __DEVICE__ element_t() : x(static_cast<T>(66)), y(static_cast<T>(99)) {}
	__HOST__ __DEVICE__ element_t( const T& x, const T& y ) : x(x), y(y) {}
	__HOST__ __DEVICE__ inline bool operator==( const element_t<T>& other ) const { return x == other.x and y == other.y; }
	__HOST__ __DEVICE__ inline bool operator!=( const element_t<T>& other ) const { return !operator==(other); }
	__HOST__ __DEVICE__ inline bool operator<( const element_t<T>& other ) const { return x == y ? y < other.y : x < other.x; }
	friend std::ostream& operator<<( std::ostream& out, const element_t<T>& el ) {
		out << "Element[" << el.x << "," << el.y << "]";
		return out;
	}
};

typedef element_t<double> data_type;

//const std::size_t N = 100; // 100k bytes
const std::size_t R = 100;
const std::size_t C = 33;

#ifdef __CUDACC__
template<typename T>
__global__
void check_matrix_accessors_on_device( const typename ecuda::matrix<T>::kernel_argument matrix, typename ecuda::array<int,3>::kernel_argument results )
{
	if( !threadIdx.x ) {
		ecuda::fill( results.begin(), results.end(), 1 );
		for( std::size_t i = 0; i < matrix.number_rows(); ++i ) {
			for( std::size_t j = 0; j < matrix.number_columns(); ++j ) {
				if( matrix.at(i,j) != T(i,j) ) results[0] = 0;
				if( matrix[i][j] != T(i,j) ) results[1] = 0;
				if( matrix(i,j) != T(i,j) ) results[2] = 0;
			}
		}
	}
}
#endif

SCENARIO( "matrix functions correctly", "matrix" ) {
	GIVEN( "one default-intialized ecuda::matrix with a value of (66,99) and another ecuda::array with unique values from (0,0)-(99,33) inclusive" ) {
		ecuda::matrix<data_type> deviceMatrixWithDefaultValues(R,C);
		ecuda::matrix<data_type> deviceMatrixWithUniqueValues(R,C);
		data_type hostMatrixWithUniqueValues[R][C];
		for( std::size_t i = 0; i < R; ++i )
			for( std::size_t j = 0; j < C; ++j )
				hostMatrixWithUniqueValues[i][j] = data_type(i,j);
		ecuda::copy( &hostMatrixWithUniqueValues[0][0], &hostMatrixWithUniqueValues[R][0], deviceMatrixWithUniqueValues.begin() );
		WHEN( "the values of the former are inspected" ) {
			std::vector<data_type> v( R*C );
			THEN( "they should all be default initialized" ) {
				REQUIRE( ecuda::equal( deviceMatrixWithDefaultValues.begin(), deviceMatrixWithDefaultValues.end(), v.begin() ) );
				REQUIRE( ecuda::equal( v.begin(), v.end(), deviceMatrixWithDefaultValues.begin() ) );
			}
		}
		AND_WHEN( "the values of the latter are inspected" ) {
			THEN( "they should all have expected values" ) {
				REQUIRE( ecuda::equal( deviceMatrixWithUniqueValues.begin(), deviceMatrixWithUniqueValues.end(), &hostMatrixWithUniqueValues[0][0] ) );
				REQUIRE( ecuda::equal( &hostMatrixWithUniqueValues[0][0], &hostMatrixWithUniqueValues[R][0], deviceMatrixWithUniqueValues.begin() ) );
			}
		}
		AND_WHEN( "another matrix is copy constructed" ) {
			ecuda::matrix<data_type> mat2( deviceMatrixWithDefaultValues );
			THEN( "they should be equal" ) {
				REQUIRE( ecuda::equal( deviceMatrixWithDefaultValues.begin(), deviceMatrixWithDefaultValues.end(), mat2.begin() ) );
				REQUIRE( ecuda::equal( mat2.begin(), mat2.end(), deviceMatrixWithDefaultValues.begin() ) );
			}
			AND_THEN( "they should reside in different memory locations" ) { REQUIRE( deviceMatrixWithDefaultValues.data() != mat2.data() ); }
		}
		AND_WHEN( "another matrix is copy assigned") {
			ecuda::matrix<data_type> mat2( R, C );
			mat2 = deviceMatrixWithDefaultValues;
			THEN( "they should be equal" ) {
				REQUIRE( ecuda::equal( deviceMatrixWithDefaultValues.begin(), deviceMatrixWithDefaultValues.end(), mat2.begin() ) );
				REQUIRE( ecuda::equal( mat2.begin(), mat2.end(), deviceMatrixWithDefaultValues.begin() ) );
			}
			AND_THEN( "they should reside in different memory locations" ) { REQUIRE( deviceMatrixWithDefaultValues.data() != mat2.data() ); }
		}
		#ifdef __CPP11_SUPPORTED__
		AND_WHEN( "another matrix is move constructed" ) {
			ecuda::matrix<data_type> mat2( std::move(deviceMatrixWithDefaultValues) );
			THEN( "the original matrix should be invalid" ) { REQUIRE( !deviceMatrixWithDefaultValues.data() ); }
			AND_THEN( "the new matrix should have the original one's contents" ) { REQUIRE( ecuda::count( mat2.begin(), mat2.end(), data_type() ) == (R*C) ); }
			AND_THEN( "a move assignment operation should restore the original state" ) {
				deviceMatrixWithDefaultValues = std::move(mat2);
				REQUIRE( deviceMatrixWithDefaultValues.data() );
				REQUIRE( !mat2.data() );
			}
		}
		#endif
		#ifdef __CUDACC__
		AND_WHEN( "the at, operator[], and operator() accessors are used in device code" ) {
			THEN( "the values should be as expected" ) {
				ecuda::array<int,3> deviceResultCodes;
				CUDA_CALL_KERNEL_AND_WAIT( check_matrix_accessors_on_device<data_type><<<1,1>>>( deviceMatrixWithUniqueValues, deviceResultCodes ) );
				std::vector<int> hostResultCodes( 3 );
				ecuda::copy( deviceResultCodes.begin(), deviceResultCodes.end(), hostResultCodes.begin() );
				REQUIRE( hostResultCodes[0] ); // at()
				REQUIRE( hostResultCodes[1] ); // operator[]
				REQUIRE( hostResultCodes[2] ); // operator()
			}
		}
		#else
		AND_WHEN( "the at, operator[], and operator() accessors are used in host code" ) {
			THEN( "the values should be as expected" ) {
				std::vector<int> resultCodes( 3, 1 );
				for( std::size_t i = 0; i < R; ++i ) {
					for( std::size_t j = 0; j < C; ++j ) {
						if( deviceMatrixWithUniqueValues.at(i,j) != data_type(i,j) ) resultCodes[0] = 0;
						if( deviceMatrixWithUniqueValues[i][j]   != data_type(i,j) ) resultCodes[1] = 0;
						if( deviceMatrixWithUniqueValues(i,j)    != data_type(i,j) ) resultCodes[2] = 0;
					}
				}
				REQUIRE( resultCodes[0] );
				REQUIRE( resultCodes[1] );
				REQUIRE( resultCodes[2] );
				bool exceptionThrown = false;
				try {
					deviceMatrixWithUniqueValues.at(R,C);
				} catch( std::out_of_range& ex ) {
					exceptionThrown = true;
				}
				REQUIRE(exceptionThrown);
			}
		}
		#endif // __CUDACC__
		#ifdef __CUDACC__
		#else
		AND_WHEN( "the front() accessor is used" ) { THEN( "the value should be 0,0" )   { REQUIRE( deviceMatrixWithUniqueValues.front() == data_type(0,0) ); } }
		AND_WHEN( "the back() accessor is used" )  { THEN( "the value should be R-1,C-1" ) { REQUIRE( deviceMatrixWithUniqueValues.back() == data_type(R-1,C-1) ); } }
		#endif
		AND_WHEN( "the begin() and end() iterators are used to traverse the matrix" ) {
			THEN( "the values should increase from 0,0 to R-1,C-1" ) {
				#ifdef __CUDACC__
				#else
				std::size_t counter = 0;
				bool correctValues = true;
				for( typename ecuda::matrix<data_type>::const_iterator iter = deviceMatrixWithUniqueValues.begin(); iter != deviceMatrixWithUniqueValues.end(); ++iter, ++counter ) {
					if( *iter != data_type( counter/C, counter % C ) ) correctValues = false;
				}
				REQUIRE(correctValues);
				#endif
			}
		}
		AND_WHEN( "the rbegin() and rend() iterators are used to traverse the matrix" ) {
			THEN( "the values should decrease from R-1,C-1 to 0,0" ) {
				#ifdef __CUDACC__
				#else
				std::size_t counter = R*C-1;
				bool correctValues = true;
				for( typename ecuda::matrix<data_type>::const_reverse_iterator iter = deviceMatrixWithUniqueValues.rbegin(); iter != deviceMatrixWithUniqueValues.rend(); ++iter, --counter ) {
					if( *iter != data_type( counter/C, counter % C) ) correctValues = false;
				}
				REQUIRE(correctValues);
				#endif
			}
		}
		AND_WHEN( "the matrices are lexicographically compared" ) {
			THEN( "the former matrix should compare less than the latter matrix (since the default value is (66,99))" ) {
				REQUIRE( deviceMatrixWithUniqueValues < deviceMatrixWithDefaultValues );
				REQUIRE( !( deviceMatrixWithUniqueValues > deviceMatrixWithDefaultValues ) );
				REQUIRE( !( deviceMatrixWithUniqueValues >= deviceMatrixWithDefaultValues ) );
			}
		}
	}
}

SCENARIO( "matrix rows function correctly", "matrix_rows" ) {
	GIVEN( "the 16th row from: one default-initialized ecuda::matrix with a value of (66,99) and another ecuda::array with unique values from (0,0)-(99,33) inclusive" ) {
		ecuda::matrix<data_type> deviceMatrixWithDefaultValues(R,C);
		ecuda::matrix<data_type> deviceMatrixWithUniqueValues(R,C);
		data_type hostMatrixWithUniqueValues[R][C];
		for( std::size_t i = 0; i < R; ++i )
			for( std::size_t j = 0; j < C; ++j )
				hostMatrixWithUniqueValues[i][j] = data_type(i,j);
		ecuda::copy( &hostMatrixWithUniqueValues[0][0], &hostMatrixWithUniqueValues[R][0], deviceMatrixWithUniqueValues.begin() );
		typename ecuda::matrix<data_type>::row_type deviceRowWithDefaultValues = deviceMatrixWithDefaultValues.get_row(16);
		typename ecuda::matrix<data_type>::row_type deviceRowWithUniqueValues  = deviceMatrixWithUniqueValues[16];
		WHEN( "the size of the row is inspected" ) {
			THEN( "it should be equal to the number of matrix columns" ) { REQUIRE( deviceRowWithDefaultValues.size() == C ); }
		}
		AND_WHEN( "the values of the former are inspected" ) {
			std::vector<data_type> v( C );
			THEN( "they should all be default initialized" ) {
				REQUIRE( ecuda::equal( deviceRowWithDefaultValues.begin(), deviceRowWithDefaultValues.end(), v.begin() ) );
				REQUIRE( ecuda::equal( v.begin(), v.end(), deviceRowWithDefaultValues.begin() ) );
			}
		}
		AND_WHEN( "the values of the latter are inspected" ) {
			THEN( "they should all have expected values" ) {
				REQUIRE( ecuda::equal( deviceRowWithUniqueValues.begin(), deviceRowWithUniqueValues.end(), &hostMatrixWithUniqueValues[16][0] ) );
				REQUIRE( ecuda::equal( &hostMatrixWithUniqueValues[16][0], &hostMatrixWithUniqueValues[16][C], deviceRowWithUniqueValues.begin() ) );
			}
		}
	}
}

SCENARIO( "matrix columns function correctly", "matrix_columns" ) {
	GIVEN( "the 8th column from: one default-initialized ecuda::matrix with a value of (66,99) and another ecuda::array with unique values from (0,0)-(99,33) inclusive" ) {
		ecuda::matrix<data_type> deviceMatrixWithDefaultValues(R,C);
		ecuda::matrix<data_type> deviceMatrixWithUniqueValues(R,C);
		data_type hostMatrixWithUniqueValues[R][C];
		for( std::size_t i = 0; i < R; ++i )
			for( std::size_t j = 0; j < C; ++j )
				hostMatrixWithUniqueValues[i][j] = data_type(i,j);
		ecuda::copy( &hostMatrixWithUniqueValues[0][0], &hostMatrixWithUniqueValues[R][0], deviceMatrixWithUniqueValues.begin() );
		typename ecuda::matrix<data_type>::column_type deviceColumnWithDefaultValues = deviceMatrixWithDefaultValues.get_column(8);
		typename ecuda::matrix<data_type>::column_type deviceColumnWithUniqueValues  = deviceMatrixWithUniqueValues.get_column(8);
		WHEN( "the size of the row is inspected" ) {
			THEN( "it should be equal to the number of matrix rows" ) { REQUIRE( deviceColumnWithDefaultValues.size() == R ); }
		}
//		AND_WHEN( "the values of the former are inspected" ) {
//			std::vector<data_type> v( C );
//			THEN( "they should all be default initialized" ) {
//				REQUIRE( ecuda::equal( deviceColumnWithDefaultValues.begin(), deviceColumnWithDefaultValues.end(), v.begin() ) );
//				REQUIRE( ecuda::equal( v.begin(), v.end(), deviceColumnWithDefaultValues.begin() ) );
//			}
//		}
//		AND_WHEN( "the values of the latter are inspected" ) {
//			THEN( "they should all have expected values" ) {
//				REQUIRE( ecuda::equal( deviceColumnWithUniqueValues.begin(), deviceColumnWithUniqueValues.end(), &hostMatrixWithUniqueValues[16][0] ) );
//				REQUIRE( ecuda::equal( &hostMatrixWithUniqueValues[16][0], &hostMatrixWithUniqueValues[16][C], deviceColumnWithUniqueValues.begin() ) );
//			}
//		}
	}
}

