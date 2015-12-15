#include <iostream>
<<<<<<< HEAD
#include <cstdio>
//#include <estd/cube.hpp>
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/cube.hpp"

template<typename T>
struct coord_t {
	T x, y, z;
	coord_t( const T& x = T(), const T& y = T(), const T& z = T() ) : x(x), y(y), z(z) {}
	bool operator==( const coord_t& other ) const { return x == other.x and y == other.y and z == other.z; }
	bool operator!=( const coord_t& other ) const { return !operator==(other); }
	friend std::ostream& operator<<( std::ostream& out, const coord_t& coord ) {
		out << "[" << coord.x << "," << coord.y << "," << coord.z << "]";
		return out;
	}
};

typedef coord_t<double> Coordinate;

typedef unsigned char uint8_t;
/*
template<typename T,std::size_t U> __global__
void fetchRow( const ecuda::cube<T> cube, ecuda::array<T,U> array ) {
	typename ecuda::cube<T>::const_row_type row = cube.get_row( 2, 3 );
	for( typename ecuda::cube<T>::const_row_type::size_type i = 0; i < row.size(); ++i ) array[i] = row[i];
}

template<typename T,std::size_t U> __global__
void fetchColumn( const ecuda::cube<T> cube, ecuda::array<T,U> array ) {
	typename ecuda::cube<T>::const_column_type column = cube.get_column( 1, 4 );
	for( typename ecuda::cube<T>::const_column_type::size_type i = 0; i < column.size(); ++i ) array[i] = column[i];
}

template<typename T,std::size_t U> __global__
void fetchDepth( const ecuda::cube<T> cube, ecuda::array<T,U> array ) {
	typename ecuda::cube<T>::const_depth_type depth = cube.get_depth( 2, 3 );
	for( typename ecuda::cube<T>::const_depth_type::size_type i = 0; i < depth.size(); ++i ) array[i] = depth[i];
}
*/

template<typename T> __global__
void fetchSliceYZ( const ecuda::cube<T> cube, ecuda::matrix<T> matrix ) {
	typename ecuda::cube<T>::const_slice_yz_type sliceYZ = cube.get_yz( 1 );
printf( "get_width()=%i\n", sliceYZ.number_columns() );
printf( "get_height()=%i\n", sliceYZ.number_rows() );
	for( unsigned i = 0; i < sliceYZ.number_rows(); ++i ) {
		for( unsigned j = 0; j < sliceYZ.number_columns(); ++j ) {
			matrix[i][j] = sliceYZ[i][j];
		}
	}
}


template<typename T> __global__
void fetchSliceXY( const ecuda::cube<T> cube, ecuda::matrix<T> matrix ) {
	typename ecuda::cube<T>::const_slice_xy_type sliceXY = cube.get_xy( 3 );
printf( "get_width()=%i\n", sliceXY.number_columns() );
printf( "get_height()=%i\n", sliceXY.number_rows() );
	for( unsigned i = 0; i < sliceXY.number_rows(); ++i ) {
		for( unsigned j = 0; j < sliceXY.number_columns(); ++j ) {
			matrix[i][j] = sliceXY[i][j];
		}
	}
}

template<typename T> __global__
void fetchSliceXZ( const ecuda::cube<T> cube, ecuda::matrix<T> matrix ) {
	typename ecuda::cube<T>::const_slice_xz_type sliceXZ = cube.get_xz( 2 );
printf( "get_width()=%i\n", sliceXZ.number_columns() );
printf( "get_height()=%i\n", sliceXZ.number_rows() );
	for( unsigned i = 0; i < sliceXZ.number_rows(); ++i ) {
		for( unsigned j = 0; j < sliceXZ.number_columns(); ++j ) {
			matrix[i][j] = sliceXZ[i][j];
		}
	}
}
/*
template<typename T,std::size_t U> __global__
void fetchAll( const ecuda::cube<T> cube, ecuda::array<T,U> array ) {
	typename ecuda::array<T,U>::size_type index = 0;
	for( typename ecuda::cube<T>::const_iterator iter = cube.begin(); iter != cube.end(); ++iter, ++index ) array[index] = *iter;
}


template<typename T,std::size_t U> __global__
void iterateAll( const ecuda::cube<T> cube, ecuda::array<T,U> array ) {
	typename ecuda::array<T,U>::size_type index = 0;
	typename ecuda::cube<T>::const_row_type row = cube.get_row(0,0);
	for( typename ecuda::cube<T>::const_row_type::const_iterator iter = row.begin(); iter != row.end(); ++iter, ++index ) array[index] = *iter;
}
*/

int main( int argc, char* argv[] ) {

/*
	estd::cube<Coordinate> hostCube( 3, 4, 5 );
	for( estd::cube<Coordinate>::size_type i = 0; i < hostCube.row_size(); ++i ) {
		for( estd::cube<Coordinate>::size_type j = 0; j < hostCube.column_size(); ++j ) {
			for( estd::cube<Coordinate>::size_type k = 0; k < hostCube.depth_size(); ++k ) {
				hostCube[i][j][k] = Coordinate(i,j,k);
			}
		}
	}
*/
	std::vector<Coordinate> hostCube( 3*4*5 );
	for( unsigned i = 0; i < 3; ++i )
		for( unsigned j = 0; j < 4; ++j )
			for( unsigned k = 0; k < 5; ++k )
				hostCube[i*(4*5)+j*5+k] = Coordinate(i,j,k);

	ecuda::cube<Coordinate> deviceCube( 3, 4, 5 );
	deviceCube << hostCube;

//	std::cout << "(1,2,3)=" << hostCube[1][2][3] << std::endl;
	deviceCube >> hostCube;
//	std::cout << "(1,2,3)=" << hostCube[1][2][3] << std::endl;

	std::cout << "sizeof(Coordinate)=" << sizeof(Coordinate) << std::endl;
/*
	{
		ecuda::cube<Coordinate>::slice_yz_type slice = deviceCube.get_yz(0);
		std::cerr << "sliceYZ.get_width()=" << slice.get_width() << std::endl;
		std::cerr << "sliceYZ.get_height()=" << slice.get_height() << std::endl;
		std::vector<Coordinate> v( slice[0].size() );
		slice[0].assign( v.begin(), v.end() );
	}
*/
/*
	{
		ecuda::array<Coordinate,3> deviceRow;
		fetchRow<<<1,1>>>( deviceCube, deviceRow );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::vector<Coordinate> hostRow( 3 );
		deviceRow >> hostRow;
		std::cout << "ROW";
		for( std::vector<Coordinate>::size_type i = 0; i < hostRow.size(); ++i ) std::cout << hostRow[i];
		std::cout << std::endl;
	}

	{
		ecuda::array<Coordinate,4> deviceColumn;
		fetchColumn<<<1,1>>>( deviceCube, deviceColumn );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::vector<Coordinate> hostColumn( 4 );
		deviceColumn >> hostColumn;
		std::cout << "COLUMN";
		for( std::vector<Coordinate>::size_type i = 0; i < hostColumn.size(); ++i ) std::cout << hostColumn[i];
		std::cout << std::endl;
	}

	{
		ecuda::array<Coordinate,5> deviceDepth;
		fetchDepth<<<1,1>>>( deviceCube, deviceDepth );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::vector<Coordinate> hostDepth( 5 );
		deviceDepth >> hostDepth;
		std::cout << "DEPTH";
		for( std::vector<Coordinate>::size_type i = 0; i < hostDepth.size(); ++i ) std::cout << hostDepth[i];
		std::cout << std::endl;
	}
*/
	{
		ecuda::matrix<Coordinate> deviceMatrix( 4, 5 );
		fetchSliceYZ<<<1,1>>>( deviceCube, deviceMatrix );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
//		estd::matrix<Coordinate> hostMatrix( 4, 5 );
		std::vector<Coordinate> hostMatrix( 4*5 );
		deviceMatrix >> hostMatrix;
		for( unsigned i = 0; i < deviceMatrix.number_rows(); ++i ) {
			std::cout << "SLICE_YZ_ROW";
			for( unsigned j = 0; j < deviceMatrix.number_columns(); ++j ) {
				std::cout << " " << hostMatrix[i*deviceMatrix.number_columns()+j];
			}
			std::cout << std::endl;
		}
	}

	{
		ecuda::matrix<Coordinate> deviceMatrix( 3, 4 );
		fetchSliceXY<<<1,1>>>( deviceCube, deviceMatrix );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		//estd::matrix<Coordinate> hostMatrix( 3, 4 );
		std::vector<Coordinate> hostMatrix( 3*4 );
		deviceMatrix >> hostMatrix;
		for( unsigned i = 0; i < deviceMatrix.number_rows(); ++i ) {
			std::cout << "SLICE_XY_ROW";
			for( unsigned j = 0; j < deviceMatrix.number_columns(); ++j ) {
				std::cout << " " << hostMatrix[i*deviceMatrix.number_columns()+j];
			}
			std::cout << std::endl;
		}
	}

	{
		ecuda::matrix<Coordinate> deviceMatrix( 3, 5 );
		fetchSliceXZ<<<1,1>>>( deviceCube, deviceMatrix );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		//estd::matrix<Coordinate> hostMatrix( 3, 5 );
		std::vector<Coordinate> hostMatrix( 3*5 );
		deviceMatrix >> hostMatrix;
		for( unsigned i = 0; i < deviceMatrix.number_rows(); ++i ) {
			std::cout << "SLICE_XZ_ROW";
			for( unsigned j = 0; j < deviceMatrix.number_columns(); ++j ) {
				std::cout << " " << hostMatrix[i*deviceMatrix.number_columns()+j];
			}
			std::cout << std::endl;
		}
	}
/*
	{
		ecuda::array<Coordinate,3> deviceArray;
		iterateAll<<<1,1>>>( deviceCube, deviceArray );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::vector<Coordinate> hostArray( 3 );
		deviceArray >> hostArray;
		for( unsigned i = 0; i < hostArray.size(); ++i ) {
			std::cout << "LINEAR";
			std::cout << " " << hostArray[i];
			std::cout << std::endl;
		}
	}

	{
		ecuda::array<Coordinate,60> deviceArray;
		fetchAll<<<1,1>>>( deviceCube, deviceArray );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::vector<Coordinate> hostArray( 60 );
		deviceArray >> hostArray;
		for( unsigned i = 0; i < hostArray.size(); ++i ) {
			std::cout << "LINEAR";
			std::cout << " " << hostArray[i];
			std::cout << std::endl;
		}
	}
*/
=======
#include <list>
//#include <initializer_list>
#include <vector>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/cube.hpp"

#ifdef __CUDACC__
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
	#ifndef __CUDACC__
	std::cout << "ROW ="; for( std::size_t i = 0; i < cubeRow.size(); ++i ) std::cout << " " << cubeRow[i]; std::cout << std::endl;
	#endif
	typename ecuda::cube< coord_t<int> >::column_type cubeColumn = deviceCube.get_column( 0, 0 );
	#ifndef __CUDACC__
	std::cout << "COLUMN ="; for( std::size_t i = 0; i < cubeColumn.size(); ++i ) std::cout << " " << cubeColumn[i]; std::cout << std::endl;
	#endif
	typename ecuda::cube< coord_t<int> >::depth_type cubeDepth = deviceCube.get_depth( 0, 0 );
	#ifndef __CUDACC__
	std::cout << "DEPTH ="; for( std::size_t i = 0; i < cubeDepth.size(); ++i ) std::cout << " " << cubeDepth[i]; std::cout << std::endl;
	#endif

//	{
//		ecuda::cube<int> deviceCube2( 10, 10, 10 );
//		testIterators2<<<1,1>>>( deviceCube, deviceCube2 );
//		CUDA_CHECK_ERRORS();
//		CUDA_CALL( cudaDeviceSynchronize() );
//	}

>>>>>>> ecuda2/master
	return EXIT_SUCCESS;

}

