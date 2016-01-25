#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

#include "../include/ecuda/model/device_sequence.hpp"

typedef double value_type;
const std::size_t N = 100000;

template<typename T,typename U> struct is_same      { enum { value = 0 }; };
template<typename T>            struct is_same<T,T> { enum { value = 1 }; };

SCENARIO( "model::device_sequence functions correctly", "model__device_sequence" ) {
	{
		std::vector<value_type> v( N );
		for( std::size_t i = 0; i < N; ++i ) v[i] = static_cast<value_type>(i);
		std::random_shuffle( v.begin(), v.end() );

		ecuda::device_allocator<value_type> alloc;
		typedef ecuda::device_allocator<value_type>::pointer pointer;
		pointer p = alloc.allocate( N );
		typedef ecuda::model::device_sequence<value_type,pointer> sequence_type;
		{ const bool b = is_same< typename sequence_type::value_type,      value_type        >::value; REQUIRE( b ); }
		{ const bool b = is_same< typename sequence_type::pointer,         value_type*       >::value; REQUIRE( b ); }
		{ const bool b = is_same< typename sequence_type::reference,       value_type&       >::value; REQUIRE( b ); }
		{ const bool b = is_same< typename sequence_type::const_reference, const value_type& >::value; REQUIRE( b ); }
		{ const bool b = is_same< typename sequence_type::size_type,       std::size_t       >::value; REQUIRE( b ); }
		{ const bool b = is_same< typename sequence_type::difference_type, std::ptrdiff_t    >::value; REQUIRE( b ); }

		// initialize the values
		CUDA_CALL( ecuda::cudaMemcpy<value_type>( p, &v.front(), N, cudaMemcpyHostToDevice ) );

		sequence_type seq( p, N );

		REQUIRE( seq.size() == N );

		alloc.deallocate( p, N );
	}
}

