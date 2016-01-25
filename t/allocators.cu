
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <iostream>
#include "../include/ecuda/ecuda.hpp"

template<typename T>
struct element_t
{
	T val;
	element_t() : val(static_cast<T>(66)) {}
	inline bool operator==( const element_t<T>& other ) const { return val == other.val; }
	friend std::ostream& operator<<( std::ostream& out, const element_t<T>& el ) {
		out << "Element[" << el.val << "]";
		return out;
	}
};

typedef element_t<double> data_type;

const std::size_t N = 100; // 100k bytes
//typename ecuda::add_pointer<data_type>::type ptr = NULL;

SCENARIO( "allocators function correctly", "allocators" ) {
	{
		ecuda::host_allocator<data_type> alloc;
		GIVEN( "a pointer to memory allocated with an ecuda::host_allocator and a std::vector of equal size" ) {
			typename ecuda::add_pointer<data_type>::type ptr = alloc.allocate( N );
			std::vector<data_type> v( N );
			WHEN( "the sequences are compared") {
				THEN( "they should be unequal" ) { REQUIRE( !ecuda::equal( v.begin(), v.end(), ptr ) ); }
			}
			#ifndef __CUDACC__
			AND_WHEN( "each element in the allocated memory is passed to the construct method" ) {
				data_type dummyValue;
				for( std::size_t i = 0; i < N; ++i ) alloc.construct( ptr+i, dummyValue );
				THEN( "the sequences should be equal" ) { REQUIRE( ecuda::equal( v.begin(), v.end(), ptr ) ); }
				for( std::size_t i = 0; i < N; ++i ) alloc.destroy( ptr+i );
			}
			#endif
			alloc.deallocate( ptr, N );
			ptr = NULL;
		}
	}
	{
		ecuda::device_allocator<data_type> alloc;
		GIVEN( "a pointer to memory allocated with an ecuda::device_allocator and a std::vector of equal size" ) {
			typename ecuda::add_pointer<data_type>::type ptr = alloc.allocate( N );
			std::vector<data_type> v( N );
			WHEN( "the sequences are compared") {
				THEN( "they should be unequal" ) { REQUIRE( !ecuda::equal( v.begin(), v.end(), ptr ) ); }
			}
			#ifndef __CUDACC__
			AND_WHEN( "each element in the allocated memory is passed to the construct method" ) {
				for( std::size_t i = 0; i < N; ++i ) alloc.construct( ptr+i, data_type() );
				THEN( "the sequences should be equal" ) { REQUIRE( ecuda::equal( v.begin(), v.end(), ptr ) ); }
				for( std::size_t i = 0; i < N; ++i ) alloc.destroy( ptr+i );
			}
			#endif
			alloc.deallocate( ptr, N );
			ptr = NULL;
		}
	}
	{
		ecuda::device_pitch_allocator<data_type> alloc;
		GIVEN( "a pointer to a block of memory allocated with an ecuda::device_pitch_allocator and a std::vector of equal size" ) {
			typename ecuda::device_pitch_allocator<data_type>::pointer ptr = alloc.allocate( N, N );
			std::vector<data_type> v( N*N );
			ecuda::device_contiguous_block_iterator<data_type,typename ecuda::add_pointer<data_type>::type> first( ptr, N );
			WHEN( "the sequences are compared") {
				THEN( "they should be unequal" ) { REQUIRE( !ecuda::equal( v.begin(), v.end(), first ) ); }
			}
			#ifndef __CUDACC__
			AND_WHEN( "each element in the allocated memory is passed to the construct method" ) {
				ecuda::device_contiguous_block_iterator<data_type,typename ecuda::add_pointer<data_type>::type> out( first );
				for( std::size_t i = 0; i < N*N; ++i, ++out ) alloc.construct( out.operator->(), data_type() );
				THEN( "the sequences should be equal" ) { REQUIRE( ecuda::equal( v.begin(), v.end(), first ) ); }
				out = first;
				for( std::size_t i = 0; i < N; ++i ) alloc.destroy( out.operator->() );
			}
			#endif
			AND_WHEN( "the start byte and the end byte are retrieved" ) {
				const char* rawStart = ecuda::naked_cast<const char*>( ptr );
				const char* rawEnd = ecuda::naked_cast<const char*>( ptr+(N*N) );
				THEN( "the region should not be contiguous" ) { REQUIRE( (rawEnd-rawStart) == (N*N*sizeof(data_type)) ); }
			}
			alloc.deallocate( ptr, N*N );
			ptr = NULL;
		}
	}
}



