
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <iostream>
#include "../include/ecuda/ecuda.hpp"

template<typename T>
struct element_t
{
	T val;
	__HOST__ __DEVICE__ element_t() : val(static_cast<T>(66)) {}
	__HOST__ __DEVICE__ element_t( const T& val ) : val(val) {}
	__HOST__ __DEVICE__ inline bool operator==( const element_t<T>& other ) const { return val == other.val; }
	__HOST__ __DEVICE__ inline bool operator!=( const element_t<T>& other ) const { return !operator==(other); }
	__HOST__ __DEVICE__ inline bool operator<( const element_t<T>& other ) const { return val < other.val; }
	__HOST__ __DEVICE__ inline element_t& operator++() { ++val; return *this; }
	__HOST__ __DEVICE__ inline element_t& operator--() { --val; return *this; }
	friend std::ostream& operator<<( std::ostream& out, const element_t<T>& el ) {
		out << "Element[" << el.val << "]";
		return out;
	}
};

typedef element_t<double> data_type;

const std::size_t N = 100; // 100 elements

#ifdef __CUDACC__

template<typename T,std::size_t N>
__global__
void check_array_accessors_on_device( const typename ecuda::array<T,N>::kernel_argument array, typename ecuda::array<int,3>::kernel_argument results )
{
	if( !threadIdx.x ) {
		ecuda::fill( results.begin(), results.end(), 1 );
		for( std::size_t i = 0; i < array.size(); ++i ) {
			if( array.at(i) != T(i) ) results[0] = 0;
			if( array[i] != T(i) ) results[1] = 0;
			if( array(i) != T(i) ) results[2] = 0;
		}
	}
}

template<typename T,std::size_t N>
__global__
void check_array_bad_accessor_on_device( const typename ecuda::array<T,N>::kernel_argument array )
{
	if( !threadIdx.x ) const T& val = array.at(N); // should throw
}

template<typename T,std::size_t N>
__global__
void check_array_front_and_back_on_device( const typename ecuda::array<T,N>::kernel_argument array, typename ecuda::array<int,2>::kernel_argument results )
{
	if( !threadIdx.x ) {
		ecuda::fill( results.begin(), results.end(), 0 );
		if( array.front() != T(0) ) results[0] = 0;
		if( array.back() != T(N-1) ) results[1] = 0;
	}
}

template<typename T,std::size_t N>
__global__
void check_array_begin_and_end_on_device( const typename ecuda::array<T,N>::kernel_argument array, typename ecuda::array<int,1>::kernel_argument results )
{
	if( !threadIdx.x ) {
		ecuda::fill( results.begin(), results.end(), 1 );
		T counter = 0;
		for( typename ecuda::array<T,N>::const_iterator iter = array.begin(); iter != array.end(); ++iter, ++counter ) {
			if( *iter != counter ) results[0] = 0;
		}
	}
}

template<typename T,std::size_t N>
__global__
void check_array_rbegin_and_rend_on_device( const typename ecuda::array<T,N>::kernel_argument array, typename ecuda::array<int,1>::kernel_argument results )
{
	if( !threadIdx.x ) {
		ecuda::fill( results.begin(), results.end(), 1 );
		T counter = N-1;
		for( typename ecuda::array<T,N>::const_reverse_iterator iter = array.rbegin(); iter != array.rend(); ++iter, --counter ) {
			if( *iter != counter ) results[0] = 0;
		}
	}
}

template<typename T,std::size_t N>
__global__
void check_array_lexicographical_compare_on_device(
	const typename ecuda::array<T,N>::kernel_argument array1,
	const typename ecuda::array<T,N>::kernel_argument array2,
	typename ecuda::array<int,3>::kernel_argument results
)
{
	if( !threadIdx.x ) {
		ecuda::fill( results.begin(), results.end(), 0 );
		if( array1 < array2 )      results[0] = 1;
		if( !(array1 > array2) )   results[1] = 1;
		if( !(array1 >= array2 ) ) results[2] = 1;
	}
}

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
		AND_WHEN( "the at(), operator[], and operator() accessor methods are used on the device" ) {
			THEN( "any index 0 <= index < N should be valid" ) {
				ecuda::array<int,3> deviceResultCodes;
				ecuda::fill( deviceResultCodes.begin(), deviceResultCodes.end(), 0 );
				CUDA_CALL_KERNEL_AND_WAIT( check_array_accessors_on_device<data_type,N><<<1,1>>>( deviceArrayWithUniqueValues, deviceResultCodes ) );
				std::vector<int> hostResultCodes( 3 );
				ecuda::copy( deviceResultCodes.begin(), deviceResultCodes.end(), hostResultCodes.begin() );
				REQUIRE( hostResultCodes[0] );
				REQUIRE( hostResultCodes[1] );
				REQUIRE( hostResultCodes[2] );
			}
			AND_THEN( "an invalid index N should throw an exception" ) {
				bool exceptionThrown = false;
				try {
					CUDA_CALL_KERNEL_AND_WAIT( check_array_bad_accessor_on_device<data_type,N><<<1,1>>>( deviceArrayWithUniqueValues ) );
				} catch( ecuda::cuda_error& ex ) {
					exceptionThrown = true;
					std::cout << "exception=" << ex.get_error_code() << " : " << ex.what() << std::endl;
				}
				REQUIRE(exceptionThrown);
			}
		}
		AND_WHEN( "the front() and back() accessors are used on the device" ) {
			THEN( "the values should be as expected" ) {
				ecuda::array<int,2> deviceResultCodes;
				ecuda::fill( deviceResultCodes.begin(), deviceResultCodes.end(), 0 );
				CUDA_CALL_KERNEL_AND_WAIT( check_array_front_and_back_on_device<data_type,N><<<1,1>>>( deviceArrayWithUniqueValues, deviceResultCodes ) );
				std::vector<int> hostResultCodes( 2 );
				ecuda::copy( deviceResultCodes.begin(), deviceResultCodes.end(), hostResultCodes.begin() );
				REQUIRE( hostResultCodes[0] );
				REQUIRE( hostResultCodes[1] );
			}
		}
		AND_WHEN( "the begin() and end() iterators are used to traverse the array on the device" ) {
			THEN( "the values should increase from 0 to N-1" ) {
				ecuda::array<int,1> deviceResultCodes;
				ecuda::fill( deviceResultCodes.begin(), deviceResultCodes.end(), 0 );
				CUDA_CALL_KERNEL_AND_WAIT( check_array_begin_and_end_on_device<data_type,N><<<1,1>>>( deviceArrayWithUniqueValues, deviceResultCodes ) );
				std::vector<int> hostResultCodes( 1 );
				ecuda::copy( deviceResultCodes.begin(), deviceResultCodes.end(), hostResultCodes.begin() );
				REQUIRE( hostResultCodes[0] );
			}
		}
		AND_WHEN( "the rbegin() and rend() iterators are used to traverse the array on the device" ) {
			THEN( "the values should decrease from N-1 to 0" ) {
				ecuda::array<int,1> deviceResultCodes;
				ecuda::fill( deviceResultCodes.begin(), deviceResultCodes.end(), 0 );
				CUDA_CALL_KERNEL_AND_WAIT( check_array_rbegin_and_rend_on_device<data_type,N><<<1,1>>>( deviceArrayWithUniqueValues, deviceResultCodes ) );
				std::vector<int> hostResultCodes( 1 );
				ecuda::copy( deviceResultCodes.begin(), deviceResultCodes.end(), hostResultCodes.begin() );
				REQUIRE( hostResultCodes[0] );
			}
		}
		AND_WHEN( "the arrays are lexicographically compared on the device" ) {
			THEN( "the former array should compare less than the latter array (since the default value is 66)" ) {
				ecuda::array<int,3> deviceResultCodes;
				ecuda::fill( deviceResultCodes.begin(), deviceResultCodes.end(), 0 );
				CUDA_CALL_KERNEL_AND_WAIT( check_array_lexicographical_compare_on_device<data_type,N><<<1,1>>>( deviceArrayWithUniqueValues, deviceArrayWithDefaultValues, deviceResultCodes ) );
				std::vector<int> hostResultCodes( 3 );
				ecuda::copy( deviceResultCodes.begin(), deviceResultCodes.end(), hostResultCodes.begin() );
				REQUIRE( hostResultCodes[0] );
				REQUIRE( hostResultCodes[1] );
				REQUIRE( hostResultCodes[2] );
			}
		}
	}
}

#else

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

#endif // __CUDACC__
