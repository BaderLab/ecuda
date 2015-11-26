#include <iostream>
#include <list>
//#include <initializer_list>
#include <vector>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/cube.hpp"

#ifndef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
template<typename T>
__global__ void testIterators( const ecuda::cube<T> src, ecuda::cube<T> dest ) {
	typename ecuda::cube<T>::iterator result = dest.begin();
	//typename ecuda::matrix<T>::const_iterator result2 = result;
	for( typename ecuda::cube<T>::const_iterator iter = src.begin(); iter != src.end(); ++iter, ++result ) *result = *iter;
}

template<typename T>
__global__ void testIterators2( const ecuda::cube<T> src, ecuda::cube<T> dest ) {
	for( typename ecuda::cube<T>::size_type i = 0; i < src.number_rows(); ++i ) {
		for( typename ecuda::cube<T>::size_type j = 0; j < src.number_columns(); ++j ) {
			typename ecuda::cube<T>::const_depth_type srcDepth = src.get_depth(i,j);
			typename ecuda::cube<T>::depth_type destDepth = dest.get_depth(i,j);
			ecuda::copy( srcDepth.begin(), srcDepth.end(), destDepth.begin() );
		}
	}
}
#endif

template<typename T>
void print_matrix( std::ostream& out, T* ptr, const std::size_t w, const std::size_t h )
{
	T* p = ptr;
	for( std::size_t i = 0; i < w; ++i ) {
		for( std::size_t j = 0; j < h; ++j ) {
			if( j ) out << " ";
			out << *p;
			++p;
		}
		out << std::endl;
	}
}

template<typename T>
struct coord_t {
	T x, y, z;
	coord_t() : x(0), y(0), z(0) {}
	coord_t( T x, T y, T z ) : x(x), y(y), z(z) {}
	coord_t& operator=( const coord_t& src ) {
		x = src.x;
		y = src.y;
		z = src.z;
		return *this;
	}
};

template<typename T>
std::ostream& operator<<( std::ostream& out, const coord_t<T>& src )
{
	out << "(" << src.x << "," << src.y << "," << src.z << ")";
	return out;
}

template<class Slice>
void print_slice( const std::string& label, Slice slice )
{
	std::cout << label << ":" << std::endl;
	std::cout << "dimensions: " << slice.number_rows() << " x " << slice.number_columns() << std::endl;
	for( std::size_t i = 0; i < slice.number_rows(); ++i ) {
		typename Slice::row_type row = slice[i];
		for( std::size_t j = 0; j < slice.number_columns(); ++j ) {
			if( j ) std::cout << " ";
			std::cout << row[j];
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	for( std::size_t i = 0; i < slice.number_columns(); ++i ) {
		typename Slice::column_type column = slice.get_column(i);
		for( std::size_t j = 0; j < slice.number_rows(); ++j ) {
			if( j ) std::cout << " ";
			std::cout << column[j];
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main( int argc, char* argv[] ) {

	const std::size_t nr = 3;
	const std::size_t nc = 5;
	const std::size_t nd = 6;
	std::vector< coord_t<int> > hostVector( nr*nc*nd );
	{
		unsigned index = 0;
		for( unsigned i = 0; i < nr; ++i )
			for( unsigned j = 0; j < nc; ++j )
				for( unsigned k = 0; k < nd; ++k )
					hostVector[index++] = coord_t<int>(i,j,k);
	}

	ecuda::cube< coord_t<int> > deviceCube( nr, nc, nd );
	ecuda::copy( hostVector.begin(), hostVector.end(), deviceCube.begin() );

	std::cout << "deviceCube.number_rows()=" << deviceCube.number_rows() << std::endl;
	std::cout << "deviceCube.number_columns()=" << deviceCube.number_columns() << std::endl;
	std::cout << "deviceCube.number_depths()=" << deviceCube.number_depths() << std::endl;

	std::fill( hostVector.begin(), hostVector.end(), coord_t<int>() );
	//hostVector.assign( hostVector.size(), 0 );
	ecuda::copy( deviceCube.begin(), deviceCube.end(), hostVector.begin() );
	//deviceCube >> hostVector;

	for( std::size_t i = 0; i < hostVector.size(); ++i ) std::cout << "[" << i << "]=" << hostVector[i] << std::endl;

	print_matrix( std::cout, &hostVector.front(), nr*nc, nd );

	print_slice( "SLICEXY", deviceCube.get_xy(0) );
	print_slice( "SLICEXZ", deviceCube.get_xz(0) );
	print_slice( "SLICEYZ", deviceCube.get_yz(0) );

	{
		const ecuda::cube< coord_t<int> > deviceCube2( deviceCube );
		print_slice( "CONSTSLICEXY", deviceCube2.get_xy(0) );
		print_slice( "CONSTSLICEXZ", deviceCube2.get_xz(0) );
		print_slice( "CONSTSLICEYZ", deviceCube2.get_yz(0) );
	}

	typename ecuda::cube< coord_t<int> >::row_type cubeRow = deviceCube.get_row( 0, 0 );
	std::cout << "ROW ="; for( std::size_t i = 0; i < cubeRow.size(); ++i ) std::cout << " " << cubeRow[i]; std::cout << std::endl;
	typename ecuda::cube< coord_t<int> >::column_type cubeColumn = deviceCube.get_column( 0, 0 );
	std::cout << "COLUMN ="; for( std::size_t i = 0; i < cubeColumn.size(); ++i ) std::cout << " " << cubeColumn[i]; std::cout << std::endl;
	typename ecuda::cube< coord_t<int> >::depth_type cubeDepth = deviceCube.get_depth( 0, 0 );
	std::cout << "DEPTH ="; for( std::size_t i = 0; i < cubeDepth.size(); ++i ) std::cout << " " << cubeDepth[i]; std::cout << std::endl;

//	{
//		ecuda::cube<int> deviceCube2( 10, 10, 10 );
//		testIterators2<<<1,1>>>( deviceCube, deviceCube2 );
//		CUDA_CHECK_ERRORS();
//		CUDA_CALL( cudaDeviceSynchronize() );
//	}

	return EXIT_SUCCESS;

}

