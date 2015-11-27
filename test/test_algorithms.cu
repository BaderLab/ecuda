#include <iomanip>
#include <iostream>
#include <list>
#include <vector>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/matrix.hpp"
#include "../include/ecuda/vector.hpp"

#include <estd/matrix.hpp>

#ifndef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
template<typename T>
__global__ void testIterators( const typename ecuda::matrix<T>::kernel src, typename ecuda::matrix<T>::kernel dest ) {
	typename ecuda::matrix<T>::iterator result = dest.begin();
	//typename ecuda::matrix<T>::const_iterator result2 = result;
	for( typename ecuda::matrix<T>::const_iterator iter = src.begin(); iter != src.end(); ++iter, ++result ) *result = *iter;
}

template<typename T>
__global__ void testIterators2( const ecuda::matrix<T> src, ecuda::matrix<T> dest ) {
	for( typename ecuda::matrix<T>::size_type i = 0; i < src.number_columns(); ++i ) {
		typename ecuda::matrix<T>::const_column_type srcColumn = src.get_column(i);
		typename ecuda::matrix<T>::column_type destColumn = dest.get_column(i);
		ecuda::copy( srcColumn.begin(), srcColumn.end(), destColumn.begin() );
	}
}

template<typename T,class Alloc>
__global__ void testAccessors( const typename ecuda::matrix<T,Alloc>::kernel_argument src, typename ecuda::matrix<T,Alloc>::kernel_argument dest ) {
							   //ecuda::impl::matrix_device_argument<T,Alloc> dest ) {
//		const typename ecuda::matrix<T,Alloc>::argument src, typename ecuda::matrix<T,Alloc>::argument dest ) {
	//typedef ecuda::matrix<T,Alloc1> src_matrix_type;
	//typedef ecuda::matrix<U,Alloc2> dest_matrix_type;
	//for( typename src_matrix_type::size_type i = 0; i < src.number_rows(); ++i ) {
	//	for( typename src_matrix_type::size_type j = 0; j < src.number_columns(); ++j ) {
	//		dest[i][j] = src[i][j];
	//	}
	//}
}

#endif

template<typename T>
struct UnaryPredicate
{
	__DEVICE__ __HOST__ bool operator()( const T& val ) const { return true; }
};

int main( int argc, char* argv[] ) {

	const std::size_t R = 5;
	const std::size_t C = 5;
	estd::matrix<int> hostMatrix( R, C );
	ecuda::matrix<int> deviceMatrix( R, C );

	// copy
	{
		ecuda::copy( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );
		ecuda::copy( deviceMatrix.begin(), deviceMatrix.end(), hostMatrix.begin() );
		ecuda::copy( hostMatrix.begin(), hostMatrix.end(), hostMatrix.begin() );
		ecuda::copy( deviceMatrix.begin(), deviceMatrix.end(), deviceMatrix.begin() );
	}

	// fill
	{
		ecuda::fill( hostMatrix.begin(), hostMatrix.end(), 0 );
		ecuda::fill( deviceMatrix.begin(), deviceMatrix.end(), 0 );
	}

	// equal
	{
		ecuda::equal( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );
		ecuda::equal( deviceMatrix.begin(), deviceMatrix.end(), hostMatrix.begin() );
		ecuda::equal( hostMatrix.begin(), hostMatrix.end(), hostMatrix.begin() );
		ecuda::equal( deviceMatrix.begin(), deviceMatrix.end(), deviceMatrix.begin() );
	}

	// lexicographical_compare
	{
		ecuda::lexicographical_compare( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin(), deviceMatrix.end() );
		ecuda::lexicographical_compare( deviceMatrix.begin(), deviceMatrix.end(), hostMatrix.begin(), hostMatrix.end() );
		ecuda::lexicographical_compare( hostMatrix.begin(), hostMatrix.end(), hostMatrix.begin(), hostMatrix.end() );
		ecuda::lexicographical_compare( deviceMatrix.begin(), deviceMatrix.end(), deviceMatrix.begin(), deviceMatrix.end() );
	}

	// reverse
	{
		ecuda::reverse( hostMatrix.begin(), hostMatrix.end() );
		ecuda::reverse( deviceMatrix.begin(), deviceMatrix.end() );
	}

	// find
	{
		ecuda::find( hostMatrix.begin(), hostMatrix.end(), 0 );
		ecuda::find( deviceMatrix.begin(), deviceMatrix.end(), 0 );
	}

	// find_if
	{
		ecuda::find_if( hostMatrix.begin(), hostMatrix.end(), UnaryPredicate<int>() );
		ecuda::find_if( deviceMatrix.begin(), deviceMatrix.end(), UnaryPredicate<int>() );
	}

	// @TODO - some C++11 only algos go here

	// any_of
	{
		ecuda::any_of( hostMatrix.begin(), hostMatrix.end(), UnaryPredicate<int>() );
		ecuda::any_of( deviceMatrix.begin(), deviceMatrix.end(), UnaryPredicate<int>() );
	}

	// none_of
	{
		ecuda::none_of( hostMatrix.begin(), hostMatrix.end(), UnaryPredicate<int>() );
		ecuda::none_of( deviceMatrix.begin(), deviceMatrix.end(), UnaryPredicate<int>() );
	}

	// for_each
	{
		ecuda::for_each( hostMatrix.begin(), hostMatrix.end(), UnaryPredicate<int>() );
		ecuda::for_each( deviceMatrix.begin(), deviceMatrix.end(), UnaryPredicate<int>() );
	}

	// count
	{
		ecuda::count( hostMatrix.begin(), hostMatrix.end(), 0 );
		ecuda::count( deviceMatrix.begin(), deviceMatrix.end(), 0 );
	}

	// count_if
	{
		ecuda::count_if( hostMatrix.begin(), hostMatrix.end(), UnaryPredicate<int>() );
		ecuda::count_if( deviceMatrix.begin(), deviceMatrix.end(), UnaryPredicate<int>() );
	}

	// mismatch
	{
		ecuda::mismatch( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );
		ecuda::mismatch( deviceMatrix.begin(), deviceMatrix.end(), hostMatrix.begin() );
		ecuda::mismatch( hostMatrix.begin(), hostMatrix.end(), hostMatrix.begin() );
		ecuda::mismatch( deviceMatrix.begin(), deviceMatrix.end(), deviceMatrix.begin() );
	}

	return EXIT_SUCCESS;

}
