#include <iomanip>
#include <iostream>
#include <list>
#include <vector>

#include "../include/ecuda/ecuda.hpp"
#include <estd/matrix.hpp>
#include <estd/cube.hpp>

// https://github.com/philsquared/Catch
#include <catch.hpp>

#ifdef __CUDACC__
template<typename T>
__global__ void testIterators( const typename ecuda::matrix<T>::kernel src, typename ecuda::matrix<T>::kernel dest )
{
	typename ecuda::matrix<T>::iterator result = dest.begin();
	//typename ecuda::matrix<T>::const_iterator result2 = result;
	for( typename ecuda::matrix<T>::const_iterator iter = src.begin(); iter != src.end(); ++iter, ++result ) *result = *iter;
}

template<typename T>
__global__ void testIterators2( const ecuda::matrix<T> src, ecuda::matrix<T> dest )
{
	for( typename ecuda::matrix<T>::size_type i = 0; i < src.number_columns(); ++i ) {
		typename ecuda::matrix<T>::const_column_type srcColumn = src.get_column(i);
		typename ecuda::matrix<T>::column_type destColumn = dest.get_column(i);
		ecuda::copy( srcColumn.begin(), srcColumn.end(), destColumn.begin() );
	}
}

template<typename T,class Alloc>
__global__ void testAccessors( const typename ecuda::matrix<T,Alloc>::kernel_argument src, typename ecuda::matrix<T,Alloc>::kernel_argument dest )
{
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

template<typename T>
struct matrix_index_t
{
	T x, y;
	matrix_index_t( const T x = T(), const T y = T() ) : x(x), y(y) {}
	bool operator==( const matrix_index_t& other ) const { return x == other.x and y == other.y; }
	bool operator<( const matrix_index_t& other ) const { return x == other.x ? y < other.y : x < other.x; }
	template<typename U>
	friend std::ostream& operator<<( std::ostream& out, const matrix_index_t<U>& mat ) {
		out << "(" << mat.x << "," << mat.y << ")";
		return out;
	}
};
template<typename T>
struct cube_index_t : matrix_index_t<T>
{
	T z;
	cube_index_t( const T x = T(), const T y = T(), const T z = T() ) : matrix_index_t<T>(x,y), z(z) {}
	bool operator==( const cube_index_t& other ) const { return matrix_index_t<T>::operator==(other) and z == other.z; }
	bool operator<( const cube_index_t& other ) const { return matrix_index_t<T>::operator==(other) ? z < other.z : matrix_index_t<T>::operator<(other); }
	template<typename U>
	friend std::ostream& operator<<( std::ostream& out, const cube_index_t<U>& cbe ) {
		out << "(" << cbe.x << "," << cbe.y << "," << cbe.z << ")";
		return out;
	}
};

typedef double data_type;
typedef matrix_index_t<data_type> matrix_index;
typedef cube_index_t<data_type> cube_index;

template<typename T,std::size_t N>
std::ostream& operator<<( std::ostream& out, const ecuda::array<T,N>& arr )
{
	for( std::size_t i = 0; i < N; ++i ) {
		if( i ) out << " ";
		out << arr(i);
	}
	out << std::endl;
	return out;
}

template<typename T>
std::ostream& operator<<( std::ostream& out, const ecuda::vector<T>& vec )
{
	for( std::size_t i = 0; i < vec.size(); ++i ) {
		if( i ) out << " ";
		out << vec(i);
	}
	out << std::endl;
	return out;
}

template<typename T>
std::ostream& operator<<( std::ostream& out, const ecuda::matrix<T>& mat )
{
	for( std::size_t i = 0; i < mat.number_rows(); ++i ) {
		for( std::size_t j = 0; j < mat.number_columns(); ++j ) {
			if( j ) out << " ";
			out << mat(i,j);
		}
		out << std::endl;
	}
	return out;
}

template<typename T>
std::ostream& operator<<( std::ostream& out, const estd::matrix<T>& mat )
{
	for( std::size_t i = 0; i < mat.number_rows(); ++i ) {
		for( std::size_t j = 0; j < mat.number_columns(); ++j ) {
			if( j ) out << " ";
			out << mat(i,j);
		}
		out << std::endl;
	}
	return out;
}

template<typename T>
std::ostream& operator<<( std::ostream& out, const ecuda::cube<T>& cbe )
{
	for( std::size_t i = 0; i < cbe.number_rows(); ++i ) {
		out << "ROW[" << i << "]:" << std::endl;
		for( std::size_t j = 0; j < cbe.number_columns(); ++j ) {
			for( std::size_t k = 0; k < cbe.number_depths(); ++k ) {
				if( k ) out << " ";
				out << cbe(i,j,k);
			}
			out << std::endl;
		}
	}
	return out;
}

template<typename T>
std::ostream& operator<<( std::ostream& out, const estd::cube<T>& cbe )
{
	for( std::size_t i = 0; i < cbe.number_rows(); ++i ) {
		out << "ROW[" << i << "]:" << std::endl;
		for( std::size_t j = 0; j < cbe.number_columns(); ++j ) {
			for( std::size_t k = 0; k < cbe.number_depths(); ++k ) {
				if( k ) out << " ";
				out << cbe(i,j,k);
			}
			out << std::endl;
		}
	}
	return out;
}

//const std::size_t N = 1000;
const std::size_t R = 5;
const std::size_t C = 5;
const std::size_t D = 5;


void test_reverse()
{
	data_type hostArray[R];
	std::vector<data_type> hostVector( R );
	estd::matrix<matrix_index> hostMatrix( R, C );
	estd::cube<cube_index> hostCube( R, C, D );
	for( std::size_t i = 0; i < R; ++i ) {
		hostArray[i] = data_type(i);
		hostVector[i] = data_type(i);
		for( std::size_t j = 0; j < C; ++j ) {
			hostMatrix(i,j) = matrix_index(i,j);
			for( std::size_t k = 0; k < D; ++k ) {
				hostCube(i,j,k) = cube_index(i,j,k);
			}
		}
	}

	ecuda::array<data_type,R> deviceArray;
	ecuda::vector<data_type> deviceVector( R );
	ecuda::matrix<matrix_index> deviceMatrix( R, C );
	ecuda::cube<cube_index> deviceCube( R, C, D );
	ecuda::copy( hostArray, hostArray+R, deviceArray.begin() );
	ecuda::copy( hostVector.begin(), hostVector.end(), deviceVector.begin() );
	ecuda::copy( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );
	ecuda::copy( hostCube.begin(), hostCube.end(), deviceCube.begin() );

	// reverse
	{
		ecuda::reverse( hostArray, hostArray+R );
		ecuda::reverse( hostVector.begin(), hostVector.end() );
		ecuda::reverse( hostMatrix.begin(), hostMatrix.end() );
		ecuda::reverse( hostCube.begin(), hostCube.end() );
		ecuda::reverse( deviceArray.begin(), deviceArray.end() );
		ecuda::reverse( deviceVector.begin(), deviceVector.end() );
		ecuda::reverse( deviceMatrix.begin(), deviceMatrix.end() );
		ecuda::reverse( deviceCube.begin(), deviceCube.end() );
		std::cout << deviceArray << std::endl;
		std::cout << deviceVector << std::endl;
		std::cout << deviceMatrix << std::endl;
		std::cout << deviceCube << std::endl;
	}
}

void test_find()
{
	data_type hostArray[R];
	std::vector<data_type> hostVector( R );
	estd::matrix<matrix_index> hostMatrix( R, C );
	estd::cube<cube_index> hostCube( R, C, D );
	for( std::size_t i = 0; i < R; ++i ) {
		hostArray[i] = data_type(i);
		hostVector[i] = data_type(i);
		for( std::size_t j = 0; j < C; ++j ) {
			hostMatrix(i,j) = matrix_index(i,j);
			for( std::size_t k = 0; k < D; ++k ) {
				hostCube(i,j,k) = cube_index(i,j,k);
			}
		}
	}

	ecuda::array<data_type,R> deviceArray;
	ecuda::vector<data_type> deviceVector( R );
	ecuda::matrix<matrix_index> deviceMatrix( R, C );
	ecuda::cube<cube_index> deviceCube( R, C, D );
	ecuda::copy( hostArray, hostArray+R, deviceArray.begin() );
	ecuda::copy( hostVector.begin(), hostVector.end(), deviceVector.begin() );
	ecuda::copy( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );
	ecuda::copy( hostCube.begin(), hostCube.end(), deviceCube.begin() );

	std::cout << "H(1)     = " << ( ecuda::find( hostArray, hostArray+R, 1 ) - hostArray ) << std::endl;
	std::cout << "D(1)     = " << ( ecuda::find( deviceArray.begin(), deviceArray.end(), 1 ) - deviceArray.begin() ) << std::endl;
	std::cout << "H(1,1)   = " << ( ecuda::find( hostMatrix.begin(), hostMatrix.end(), matrix_index(1,1) ) - hostMatrix.begin() ) << std::endl;
	std::cout << "D(1,1)   = " << ( ecuda::find( deviceMatrix.begin(), deviceMatrix.end(), matrix_index(1,1) ) - deviceMatrix.begin() ) << std::endl;
	std::cout << "H(1,1,1) = " << ( ecuda::find( hostCube.begin(), hostCube.end(), cube_index(1,1,1) ) - hostCube.begin() ) << std::endl;
	std::cout << "D(1,1,1) = " << ( ecuda::find( deviceCube.begin(), deviceCube.end(), cube_index(1,1,1) ) - deviceCube.begin() ) << std::endl;
}


int main( int, char** )
{

	data_type hostArray[R];
	std::vector<data_type> hostVector( R );
	estd::matrix<matrix_index> hostMatrix( R, C );
	estd::cube<cube_index> hostCube( R, C, D );
	for( std::size_t i = 0; i < R; ++i ) {
		hostArray[i] = data_type(i);
		hostVector[i] = data_type(i);
		for( std::size_t j = 0; j < C; ++j ) {
			hostMatrix(i,j) = matrix_index(i,j);
			for( std::size_t k = 0; k < D; ++k ) {
				hostCube(i,j,k) = cube_index(i,j,k);
			}
		}
	}

	ecuda::array<data_type,R> deviceArray;
	ecuda::vector<data_type> deviceVector( R );
	ecuda::matrix<matrix_index> deviceMatrix( R, C );
	ecuda::cube<cube_index> deviceCube( R, C, D );

	// copy, equal, and fill
	{
		std::cout << "ecuda::copy, ecuda::equal and ecuda::fill:" << std::endl;

		ecuda::copy( hostArray, hostArray+R, deviceArray.begin() );
		std::cout << "  array  =>";
		std::cout << " " << std::boolalpha << ecuda::equal( hostArray, hostArray+R, deviceArray.begin() );
		ecuda::fill( hostArray, hostArray+R, data_type() );
		std::cout << " " << std::boolalpha << !ecuda::equal( hostArray, hostArray+R, deviceArray.begin() );
		ecuda::copy( deviceArray.begin(), deviceArray.end(), hostArray );
		std::cout << " " << std::boolalpha << ecuda::equal( hostArray, hostArray+R, deviceArray.begin() );
		ecuda::fill( hostArray, hostArray+R, data_type() );
		ecuda::fill( deviceArray.begin(), deviceArray.end(), data_type() );
		std::cout << " " << std::boolalpha << ecuda::equal( hostArray, hostArray+R, deviceArray.begin() );
		std::cout << std::endl;

		ecuda::copy( hostVector.begin(), hostVector.end(), deviceVector.begin() );
		std::cout << "  vector =>";
		std::cout << " " << std::boolalpha << ecuda::equal( hostVector.begin(), hostVector.end(), deviceVector.begin() );
		ecuda::fill( hostVector.begin(), hostVector.end(), data_type() );
		std::cout << " " << std::boolalpha << !ecuda::equal( hostVector.begin(), hostVector.end(), deviceVector.begin() );
		ecuda::copy( deviceVector.begin(), deviceVector.end(), hostVector.begin() );
		std::cout << " " << std::boolalpha << ecuda::equal( hostVector.begin(), hostVector.end(), deviceVector.begin() );
		ecuda::fill( hostVector.begin(), hostVector.end(), data_type() );
		ecuda::fill( deviceVector.begin(), deviceVector.end(), data_type() );
		std::cout << " " << std::boolalpha << ecuda::equal( hostVector.begin(), hostVector.end(), deviceVector.begin() );
		std::cout << std::endl;

		ecuda::copy( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );
		std::cout << "  matrix =>";
		std::cout << " " << std::boolalpha << ecuda::equal( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );
		ecuda::fill( hostMatrix.begin(), hostMatrix.end(), matrix_index() );
		std::cout << " " << std::boolalpha << !ecuda::equal( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );
		ecuda::copy( deviceMatrix.begin(), deviceMatrix.end(), hostMatrix.begin() );
		std::cout << " " << std::boolalpha << ecuda::equal( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );
		ecuda::fill( hostMatrix.begin(), hostMatrix.end(), matrix_index() );
		ecuda::fill( deviceMatrix.begin(), deviceMatrix.end(), matrix_index() );
		std::cout << " " << std::boolalpha << ecuda::equal( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );
		std::cout << std::endl;

		ecuda::copy( hostCube.begin(), hostCube.end(), deviceCube.begin() );
		std::cout << "  cube   =>";
		std::cout << " " << std::boolalpha << ecuda::equal( hostCube.begin(), hostCube.end(), deviceCube.begin() );
		ecuda::fill( hostCube.begin(), hostCube.end(), cube_index() );
		std::cout << " " << std::boolalpha << !ecuda::equal( hostCube.begin(), hostCube.end(), deviceCube.begin() );
		ecuda::copy( deviceCube.begin(), deviceCube.end(), hostCube.begin() );
		std::cout << " " << std::boolalpha << ecuda::equal( hostCube.begin(), hostCube.end(), deviceCube.begin() );
		ecuda::fill( hostCube.begin(), hostCube.end(), cube_index() );
		ecuda::fill( deviceCube.begin(), deviceCube.end(), cube_index() );
		std::cout << " " << std::boolalpha << ecuda::equal( hostCube.begin(), hostCube.end(), deviceCube.begin() );
		std::cout << std::endl;

		std::cout << std::endl;
	}

	// copy
	{
		ecuda::copy( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );
		ecuda::copy( deviceMatrix.begin(), deviceMatrix.end(), hostMatrix.begin() );
		ecuda::copy( hostMatrix.begin(), hostMatrix.end(), hostMatrix.begin() );
		ecuda::copy( deviceMatrix.begin(), deviceMatrix.end(), deviceMatrix.begin() );
	}

	// equal
	{
		ecuda::equal( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );
		ecuda::equal( deviceMatrix.begin(), deviceMatrix.end(), hostMatrix.begin() );
		ecuda::equal( hostMatrix.begin(), hostMatrix.end(), hostMatrix.begin() );
		ecuda::equal( deviceMatrix.begin(), deviceMatrix.end(), deviceMatrix.begin() );
	}

	// fill
	{
		ecuda::fill( hostMatrix.begin(), hostMatrix.end(), matrix_index() );
		ecuda::fill( deviceMatrix.begin(), deviceMatrix.end(), matrix_index() );
	}

	// lexicographical_compare
	{
		ecuda::lexicographical_compare( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin(), deviceMatrix.end() );
		ecuda::lexicographical_compare( deviceMatrix.begin(), deviceMatrix.end(), hostMatrix.begin(), hostMatrix.end() );
		ecuda::lexicographical_compare( hostMatrix.begin(), hostMatrix.end(), hostMatrix.begin(), hostMatrix.end() );
		ecuda::lexicographical_compare( deviceMatrix.begin(), deviceMatrix.end(), deviceMatrix.begin(), deviceMatrix.end() );
	}

	test_reverse();
	test_find();

	// find_if
	{
		ecuda::find_if( hostMatrix.begin(), hostMatrix.end(), UnaryPredicate<matrix_index>() );
		ecuda::find_if( deviceMatrix.begin(), deviceMatrix.end(), UnaryPredicate<matrix_index>() );
	}

	// @TODO - some C++11 only algos go here

	// any_of
	{
		ecuda::any_of( hostMatrix.begin(), hostMatrix.end(), UnaryPredicate<matrix_index>() );
		ecuda::any_of( deviceMatrix.begin(), deviceMatrix.end(), UnaryPredicate<matrix_index>() );
	}

	// none_of
	{
		ecuda::none_of( hostMatrix.begin(), hostMatrix.end(), UnaryPredicate<matrix_index>() );
		ecuda::none_of( deviceMatrix.begin(), deviceMatrix.end(), UnaryPredicate<matrix_index>() );
	}

	// for_each
	{
		ecuda::for_each( hostMatrix.begin(), hostMatrix.end(), UnaryPredicate<matrix_index>() );
		ecuda::for_each( deviceMatrix.begin(), deviceMatrix.end(), UnaryPredicate<matrix_index>() );
	}

	// count
	{
		ecuda::count( hostMatrix.begin(), hostMatrix.end(), 0 );
		ecuda::count( deviceMatrix.begin(), deviceMatrix.end(), 0 );
	}

	// count_if
	{
		ecuda::count_if( hostMatrix.begin(), hostMatrix.end(), UnaryPredicate<matrix_index>() );
		ecuda::count_if( deviceMatrix.begin(), deviceMatrix.end(), UnaryPredicate<matrix_index>() );
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
