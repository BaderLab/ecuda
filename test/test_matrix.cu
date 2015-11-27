#include <iomanip>
#include <iostream>
#include <list>
#include <vector>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/matrix.hpp"
#include "../include/ecuda/vector.hpp"

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
__global__ void testAccessors( const typename ecuda::matrix<T,Alloc>::kernel_argument src, typename ecuda::matrix<T,Alloc>::kernel_argument dest )
{
	ecuda::copy( src.begin(), src.end(), dest.begin() );
	ecuda::equal( src.begin(), src.end(), dest.begin() );
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
struct coord_t
{
	T x, y;
	coord_t( T x, T y ) : x(x), y(y) {}
	coord_t() : x(0), y(0) {}
	bool operator==( const coord_t& other ) const { return x == other.x and y == other.y; }
};

template<typename T>
std::ostream& operator<<( std::ostream& out, const coord_t<T>& src )
{
	out << "(" << src.x << "," << src.y << ")";
	return out;
}

int main( int argc, char* argv[] ) {

	{
		std::cerr << "TESTING CONSTRUCTORS" << std::endl;
		std::cerr << "--------------------" << std::endl;
		{
			const std::size_t R = 5;
			const std::size_t C = 21;
			ecuda::matrix<double> deviceMatrix( R, C );
			std::vector<double> hostVector( R*C );
			std::cerr << "ecuda::matrix() : " << std::boolalpha << ecuda::equal( deviceMatrix.begin(), deviceMatrix.end(), hostVector.begin() ) << std::endl;
		}
		{
			const std::size_t R = 5;
			const std::size_t C = 21;
			ecuda::matrix<double> deviceMatrix1( R, C );
			ecuda::fill( deviceMatrix1.begin(), deviceMatrix1.end(), 99.0 );
			ecuda::matrix<double> deviceMatrix2( deviceMatrix1 );
			std::vector<double> hostVector( R*C, 99.0 );
			std::cerr << "ecuda::matrix( const ecuda::matrix& ) : " << std::boolalpha << ecuda::equal( deviceMatrix2.begin(), deviceMatrix2.end(), hostVector.begin() ) << std::endl;
		}
		#ifdef __CPP11_SUPPORTED__
		{
			std::cerr << "ecuda::matrix( ecuda::matrix&& ) : TEST NOT IMPLEMENTED" << std::endl;
		}
		#endif
		std::cerr << std::endl;
	}
	{
		std::cerr << "TESTING ACCESSORS" << std::endl;
		std::cerr << "-----------------" << std::endl;
		{
			const std::size_t R = 5;
			const std::size_t C = 21;
			ecuda::matrix<double> deviceMatrix( R, C );
			#ifdef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
			//for( typename ecuda::matrix<double>::size_type i = 0; i < deviceMatrix.size(); ++i ) deviceMatrix[i] = static_cast<double>(i);
			for( typename ecuda::matrix<double>::size_type i = 0; i < deviceMatrix.number_rows(); ++i )
				for( typename ecuda::matrix<double>::size_type j = 0; j < deviceMatrix.number_columns(); ++j )
					deviceMatrix[i][j] = static_cast<double>(i*deviceMatrix.number_columns()+j);
			//std::cerr << "ecuda::array::operator[] : " << std::boolalpha << ( deviceArray[10] == static_cast<double>(10) ) << std::endl;
			std::cerr << "ecuda::matrix::front()    : " << std::boolalpha << ( deviceMatrix.front() == static_cast<double>(0) ) << std::endl;
			std::cerr << "ecuda::matrix::back()     : " << std::boolalpha << ( deviceMatrix.back() == static_cast<double>(R*C-1) ) << std::endl;
			//std::vector<double> hostVector( N );
			//ecuda::copy( deviceArray.rbegin(), deviceArray.rend(), hostVector.begin() );
			//std::cerr << "ecuda::array::rbegin(),rend() : " << std::boolalpha << ( deviceArray.front() == static_cast<double>(N-1) ) << "," << std::boolalpha << ( deviceArray.back() == static_cast<double>(0) ) << std::endl;
			#else
			ecuda::matrix<double> deviceMatrix2( R, C );
			{
				ecuda::matrix<double> arg1( deviceMatrix );
				ecuda::matrix<double> arg2( deviceMatrix2 );
				//testAccessors< double, ecuda::device_pitch_allocator<double> ><<<1,1>>>( arg1, arg2 );
				//CUDA_CHECK_ERRORS();
				//CUDA_CALL( cudaDeviceSynchronize() );
				CUDA_CALL_KERNEL_AND_WAIT( testAccessors< double, ecuda::device_pitch_allocator<double> ><<<1,1>>>( arg1, arg2 ) );
			}
			//ECUDA_STATIC_ASSERT(false,MUST_IMPLEMENT_ACCESSOR_AS_KERNEL);
			#endif
			std::cerr << "ecuda::matrix::empty()    : " << std::boolalpha << ( !deviceMatrix.empty() ) << std::endl;
			std::cerr << "ecuda::matrix::size()     : " << std::boolalpha << ( deviceMatrix.size() == (R*C) ) << std::endl;
			//std::cerr << "ecuda::matrix::data()     : " << std::boolalpha << ( deviceMatrix.data() > 0 ) << std::endl;
		}
		std::cerr << std::endl;
	}
	{
		std::cerr << "TESTING ROWS" << std::endl;
		std::cerr << "------------" << std::endl;
		const std::size_t R = 5;
		const std::size_t C = 21;
		std::vector< coord_t<int> > hostVector;
		hostVector.reserve( R*C );
		for( std::size_t i = 0; i < R; ++i )
			for( std::size_t j = 0; j < C; ++j )
				hostVector.push_back( coord_t<int>(i,j) );
		ecuda::matrix< coord_t<int> > deviceMatrix( R, C );
		ecuda::copy( hostVector.begin(), hostVector.end(), deviceMatrix.begin() );
		for( std::size_t i = 0; i < R; ++i ) {
			typename ecuda::matrix< coord_t<int> >::row_type row = deviceMatrix[i];
			std::cerr << "row[" << i << "]=" << std::boolalpha << ecuda::equal( row.begin(), row.end(), hostVector.begin()+(i*C) ) << std::endl;
		}
	}
	{
		std::cerr << "TESTING COLUMNS" << std::endl;
		std::cerr << "---------------" << std::endl;
		const std::size_t R = 5;
		const std::size_t C = 21;
		std::vector< coord_t<int> > hostVector;
		hostVector.reserve( R*C );
		for( std::size_t i = 0; i < R; ++i )
			for( std::size_t j = 0; j < C; ++j )
				hostVector.push_back( coord_t<int>(i,j) );
		ecuda::matrix< coord_t<int> > deviceMatrix( R, C );
		ecuda::copy( hostVector.begin(), hostVector.end(), deviceMatrix.begin() );
		#ifdef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
		for( std::size_t i = 0; i < C; ++i ) {
			typename ecuda::matrix< coord_t<int> >::column_type col = deviceMatrix.get_column(i);
			std::cerr << "column[" << i << "]"; for( std::size_t j = 0; j < col.size(); ++j ) std::cerr << " " << col[j]; std::cerr << std::endl;
		}
		#endif
	}
	{
		std::cerr << "TESTING TRANSFORMS" << std::endl;
		std::cerr << "------------------" << std::endl;
		{
			const std::size_t R = 5;
			const std::size_t C = 21;
			ecuda::matrix<double> deviceMatrix1( R, C );
			deviceMatrix1.fill( static_cast<double>(99) );
			std::vector<double> hostVector1( R*C, static_cast<double>(99) );
			std::cerr << "ecuda::matrix::fill() : " << std::boolalpha << ecuda::equal( deviceMatrix1.begin(), deviceMatrix1.end(), hostVector1.begin() ) << std::endl;
			ecuda::matrix<double> deviceMatrix2;
			deviceMatrix2.fill( static_cast<double>(66) );
			deviceMatrix1.swap( deviceMatrix2 );
			std::cerr << "ecuda::matrix::swap() : " << std::boolalpha << ecuda::equal( deviceMatrix2.begin(), deviceMatrix2.end(), hostVector1.begin() ) << std::endl;
		}
		std::cerr << std::endl;
	}

	/*

	const std::size_t nRows = 5;
	const std::size_t nCols = 21;

	std::vector<int> hostVector( nRows*nCols );
	for( unsigned i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;

	ecuda::matrix<int> deviceMatrix( nRows, nCols );
	// below needs to be made to work
	ecuda::copy( hostVector.begin(), hostVector.end(), deviceMatrix.begin() ); // TODO: confirm the pseudo-contiguous nature of deviceMatrix.begin() is being accounted for
	#ifndef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
	{
		ecuda::matrix<int> deviceMatrix2( nRows, nCols );
		testIterators2<<<1,1>>>( deviceMatrix, deviceMatrix2 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );

		// need alternative to below
		//ecuda::copy( deviceMatrix.get_column(0).begin(), deviceMatrix.get_column(0).end(), deviceMatrix2.get_column(1).begin() );

		std::cout << "EQUAL " << ( deviceMatrix == deviceMatrix2 ? "true" : "false" ) << std::endl;
		std::cout << "LESS THAN " << ( deviceMatrix < deviceMatrix2 ? "true" : "false" ) << std::endl;
	}
	#endif

	{
		std::vector<int> tmp( nRows*nCols );
		ecuda::copy( deviceMatrix.begin(), deviceMatrix.end(), tmp.begin() ); // TODO: confirm the pseudo-contiguous nature of deviceMatrix.begin() is being accounted for
		unsigned mrkr = 0;
		for( unsigned i = 0; i < nRows; ++i ) {
			std::cout << "ROW[" << i << "] ="; for( unsigned j = 0; j < nCols; ++j, ++mrkr ) std::cout << " " << std::setw(3) << tmp[mrkr]; std::cout << std::endl;
		}
	}

	ecuda::matrix_transpose( deviceMatrix );

	{
		//ecuda::matrix<int> deviceMatrix2( 2, 2 );
		//deviceMatrix2.assign( { 1, 2, 3, 4 } );
	}
	*/

	return EXIT_SUCCESS;

}
