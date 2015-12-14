
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <iostream>
#include "../include/ecuda/ecuda.hpp"

template<typename T>
struct element_t
{
	T val;
	element_t() : val(static_cast<T>(66)) {}
	element_t( const T& val ) : val(val) {}
	inline bool operator==( const element_t<T>& other ) const { return val == other.val; }
	inline bool operator!=( const element_t<T>& other ) const { return !operator==(other); }
	inline bool operator<( const element_t<T>& other ) const { return val < other.val; }
	friend std::ostream& operator<<( std::ostream& out, const element_t<T>& el ) {
		out << "Element[" << el.val << "]";
		return out;
	}
};

typedef element_t<double> data_type;

const std::size_t N = 100; // 100k bytes

SCENARIO( "array functions correctly", "array" ) {
	GIVEN( "one default-intialized ecuda::array with a value of 66 and another ecuda::array with unique values from 0-99 inclusive" ) {
		ecuda::array<data_type,N> deviceArrayWithDefaultValues;
		ecuda::array<data_type,N> deviceArrayWithUniqueValues;
		std::vector<data_type> hostArrayWithUniqueValues( N );
		for( std::size_t i = 0; i < N; ++i ) hostArrayWithUniqueValues[i] = data_type(i);
		ecuda::copy( hostArrayWithUniqueValues.begin(), hostArrayWithUniqueValues.end(), deviceArrayWithUniqueValues.begin() );
		WHEN( "the values of the former are inspected" ) {
			std::vector<data_type> v( N );
			THEN( "they should all be default initialized" ) {
				REQUIRE( ecuda::equal( deviceArrayWithDefaultValues.begin(), deviceArrayWithDefaultValues.end(), v.begin() ) );
				REQUIRE( ecuda::equal( v.begin(), v.end(), deviceArrayWithDefaultValues.begin() ) );
			}
		}
		AND_WHEN( "the values of the latter are inspected" ) {
			THEN( "they should all have expected values" ) {
				REQUIRE( ecuda::equal( deviceArrayWithUniqueValues.begin(), deviceArrayWithUniqueValues.end(), hostArrayWithUniqueValues.begin() ) );
				REQUIRE( ecuda::equal( hostArrayWithUniqueValues.begin(), hostArrayWithUniqueValues.end(), deviceArrayWithUniqueValues.begin() ) );
			}
		}
		AND_WHEN( "another array is copy constructed" ) {
			ecuda::array<data_type,N> arr2( deviceArrayWithDefaultValues );
			THEN( "they should be equal" ) {
				REQUIRE( ecuda::equal( deviceArrayWithDefaultValues.begin(), deviceArrayWithDefaultValues.end(), arr2.begin() ) );
				REQUIRE( ecuda::equal( arr2.begin(), arr2.end(), deviceArrayWithDefaultValues.begin() ) );
			}
			AND_THEN( "they should reside in different memory locations" ) { REQUIRE( deviceArrayWithDefaultValues.data() != arr2.data() ); }
		}
		AND_WHEN( "another array is copy assigned") {
			ecuda::array<data_type,N> arr2;
			arr2 = deviceArrayWithDefaultValues;
			THEN( "they should be equal" ) {
				REQUIRE( ecuda::equal( deviceArrayWithDefaultValues.begin(), deviceArrayWithDefaultValues.end(), arr2.begin() ) );
				REQUIRE( ecuda::equal( arr2.begin(), arr2.end(), deviceArrayWithDefaultValues.begin() ) );
			}
			AND_THEN( "they should reside in different memory locations" ) { REQUIRE( deviceArrayWithDefaultValues.data() != arr2.data() ); }
		}
		#ifdef __CPP11_SUPPORTED__
		AND_WHEN( "another array is move constructed" ) {
			ecuda::array<data_type,N> arr2( std::move(deviceArrayWithDefaultValues) );
			THEN( "the original array should be invalid" ) { REQUIRE( !deviceArrayWithDefaultValues.data() ); }
			AND_THEN( "the new array should have the original one's contents" ) { REQUIRE( ecuda::count( arr2.begin(), arr2.end(), data_type() ) == N ); }
			AND_THEN( "a move assignment operation should restore the original state" ) {
				deviceArrayWithDefaultValues = std::move(arr2);
				REQUIRE( deviceArrayWithDefaultValues.data() );
				REQUIRE( !arr2.data() );
			}
		}
		#endif
		AND_WHEN( "the at accessor method is used" ) {
			THEN( "any index 0 <= index < N should be valid" ) {
				bool exceptionThrown = false;
				bool correctValues = true;
				try {
					for( std::size_t i = 0; i < N; ++i ) {
						if( deviceArrayWithUniqueValues.at(i) != data_type(i) ) correctValues = false;
					}
				} catch( std::out_of_range& ex ) {
					exceptionThrown = true;
				}
				REQUIRE(correctValues);
				REQUIRE(!exceptionThrown);
			}
			AND_THEN( "an invalid index N should throw an exception" ) {
				bool exceptionThrown = false;
				try {
					deviceArrayWithUniqueValues.at(N);
				} catch( std::out_of_range& ex ) {
					exceptionThrown = true;
				}
				REQUIRE(exceptionThrown);
			}
		}
		AND_WHEN( "the operator[] accessor method is used" ) {
			THEN( "any index 0 <= index < N should be valid" ) {
				bool correctValues = true;
				for( std::size_t i = 0; i < N; ++i ) {
					if( deviceArrayWithUniqueValues[i] != data_type(i) ) correctValues = false;
				}
				REQUIRE(correctValues);
			}
		}
		AND_WHEN( "the operator() accessor method is used" ) {
			THEN( "any index 0 <= index < N should be valid" ) {
				bool correctValues = true;
				for( std::size_t i = 0; i < N; ++i ) {
					if( deviceArrayWithUniqueValues(i) != data_type(i) ) correctValues = false;
				}
				REQUIRE(correctValues);
			}
		}
		AND_WHEN( "the front() accessor is used" ) { THEN( "the value should be 0" )   { REQUIRE( deviceArrayWithUniqueValues.front() == data_type(0) ); } }
		AND_WHEN( "the back() accessor is used" )  { THEN( "the value should be N-1" ) { REQUIRE( deviceArrayWithUniqueValues.back() == data_type(N-1) ); } }
		AND_WHEN( "the begin() and end() iterators are used to traverse the array" ) {
			THEN( "the values should increase from 0 to N-1" ) {
				std::size_t counter = 0;
				bool correctValues = true;
				for( typename ecuda::array<data_type,N>::const_iterator iter = deviceArrayWithUniqueValues.begin(); iter != deviceArrayWithUniqueValues.end(); ++iter, ++counter ) {
					if( *iter != data_type(counter) ) correctValues = false;
				}
				REQUIRE(correctValues);
			}
		}
		AND_WHEN( "the rbegin() and rend() iterators are used to traverse the array" ) {
			THEN( "the values should decrease from N-1 to 0" ) {
				std::size_t counter = N-1;
				bool correctValues = true;
				for( typename ecuda::array<data_type,N>::const_reverse_iterator iter = deviceArrayWithUniqueValues.rbegin(); iter != deviceArrayWithUniqueValues.rend(); ++iter, --counter ) {
					if( *iter != data_type(counter) ) correctValues = false;
				}
				REQUIRE(correctValues);
			}
		}
		AND_WHEN( "the arrays are lexicographically compared" ) {
			THEN( "the former array should compare less than the latter array (since the default value is 66)" ) {
				REQUIRE( deviceArrayWithUniqueValues < deviceArrayWithDefaultValues );
				REQUIRE( !( deviceArrayWithUniqueValues > deviceArrayWithDefaultValues ) );
				REQUIRE( !( deviceArrayWithUniqueValues >= deviceArrayWithDefaultValues ) );
			}
		}
	}
}

