#include <iostream>
#include <cstdio>
#include <estd/cube.hpp>
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/cube.hpp"

template<typename T>
struct coord_t {
	T x, y, z;
	HOST DEVICE coord_t( const T& x = T(), const T& y = T(), const T& z = T() ) : x(x), y(y), z(z) {}
	bool operator==( const coord_t& other ) const { return x == other.x and y == other.y and z == other.z; }
	bool operator!=( const coord_t& other ) const { return !operator==(other); }
	friend std::ostream& operator<<( std::ostream& out, const coord_t& coord ) {
		out << "[" << coord.x << "," << coord.y << "," << coord.z << "]";
		return out;
	}
};

typedef coord_t<double> Coordinate;

typedef unsigned char uint8_t;

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

template<typename T> __global__
void fetchSliceYZ( /*const*/ ecuda::cube<T> cube, ecuda::matrix<T> matrix ) {
	typename ecuda::cube<T>::/*const_*/slice_yz_type sliceYZ = cube.get_yz( 1 );
printf( "number_rows()=%i\n", sliceYZ.number_rows() );
printf( "number_columns()=%i\n", sliceYZ.number_columns() );
	for( unsigned i = 0; i < sliceYZ.number_rows(); ++i ) {
		for( unsigned j = 0; j < sliceYZ.number_columns(); ++j ) {
			matrix[i][j] = sliceYZ[i][j];
		}
	}
}

template<typename T> __global__
void fetchSliceXY( /*const*/ ecuda::cube<T> cube, ecuda::matrix<T> matrix ) {
	typename ecuda::cube<T>::/*const_*/slice_xy_type sliceXY = cube.get_xy( 3 );
printf( "number_rows()=%i\n", sliceXY.number_rows() );
printf( "number_columns()=%i\n", sliceXY.number_columns() );
	for( unsigned i = 0; i < sliceXY.number_rows(); ++i ) {
		for( unsigned j = 0; j < sliceXY.number_columns(); ++j ) {
			matrix[i][j] = sliceXY[i][j];
		}
	}
}

template<typename T> __global__
void fetchSliceXZ( /*const*/ ecuda::cube<T> cube, ecuda::matrix<T> matrix ) {
	typename ecuda::cube<T>::/*const_*/slice_xz_type sliceXZ = cube.get_xz( 2 );
printf( "number_rows()=%i\n", sliceXZ.number_rows() );
printf( "number_columns()=%i\n", sliceXZ.number_columns() );
	for( unsigned i = 0; i < sliceXZ.number_rows(); ++i ) {
		for( unsigned j = 0; j < sliceXZ.number_columns(); ++j ) {
			matrix[i][j] = sliceXZ[i][j];
		}
	}
}

template<typename T,std::size_t U> __global__
void fetchAll( const ecuda::cube<T> cube, ecuda::array<T,U> array ) {
	typename ecuda::array<T,U>::size_type index = 0;
	for( typename ecuda::cube<T>::const_iterator iter = cube.begin(); iter != cube.end(); ++iter, ++index ) {
		array[index] = *iter;
	}
}

template<typename T,std::size_t U> __global__
void fetchAll( typename ecuda::cube<T>::const_iterator first, typename ecuda::cube<T>::const_iterator last, ecuda::array<T,U> array ) {
//	typename ecuda::array<T,U>::size_type index = 0;
	for( unsigned i = 0; i < 60; ++i ) {
		array[i] = *first;
		printf( "i=%i %i %i %i\n", i, first->x, first->y, first->z );
		++first;
	}
//	while( first != last ) {
//		array[index] = *first;
//		++first;
//		index++;
//	}
}

template<typename T,std::size_t U> __global__
void iterateAll( const ecuda::cube<T> cube, ecuda::array<T,U> array ) {
	typename ecuda::array<T,U>::size_type index = 0;
	typename ecuda::cube<T>::const_row_type row = cube.get_row(0,0);
	for( typename ecuda::cube<T>::const_row_type::const_iterator iter = row.begin(); iter != row.end(); ++iter, ++index ) array[index] = *iter;
}

int main( int argc, char* argv[] ) {

	estd::cube<Coordinate> hostCube( 3, 4, 5 );
	for( estd::cube<Coordinate>::size_type i = 0; i < hostCube.row_size(); ++i ) {
		for( estd::cube<Coordinate>::size_type j = 0; j < hostCube.column_size(); ++j ) {
			for( estd::cube<Coordinate>::size_type k = 0; k < hostCube.depth_size(); ++k ) {
				hostCube[i][j][k] = Coordinate(i,j,k);
			}
		}
	}

	ecuda::cube<Coordinate> deviceCube( 3, 4, 5 );
	ecuda::cudaMemset<char>( reinterpret_cast<char*>( (Coordinate*)deviceCube.data() ), 9, 1440 );
	deviceCube.assign( hostCube.begin(), hostCube.end() );
	//deviceCube << hostCube;

	std::cout << "(1,2,3)=" << hostCube[1][2][3] << std::endl;
	deviceCube >> hostCube;
	std::cout << "(1,2,3)=" << hostCube[1][2][3] << std::endl;

	std::cout << "sizeof(Coordinate)=" << sizeof(Coordinate) << std::endl;

	{
		std::cout << "deviceCube.data()=" << deviceCube.data() << std::endl;
		std::cout << "deviceCube.xy_slice[3].data()=" << deviceCube.get_xy(3).data() << std::endl;
		std::cout << "deviceCube.yz_slice[1].data()=" << deviceCube.get_yz(1).data() << std::endl;
		std::cout << "deviceCube.xz_slice[2].data()=" << deviceCube.get_xz(2).data() << std::endl;
		std::cout << "deviceCube.begin()=" << deviceCube.begin().operator->() << std::endl;
		std::cout << "deviceCube.end()=" << deviceCube.end().operator->() << std::endl;
	}

	{
		std::vector<Coordinate> v( deviceCube.size() );
		deviceCube >> v;
		for( unsigned i = 0; i < v.size(); ++i ) {
			std::cout << i << " " << v[i] << std::endl;
		}
	}

	{
		std::vector<Coordinate> v( 5 );
		ecuda::cube<Coordinate>::slice_yz_type slice = deviceCube.get_yz(0);
		slice[0].assign( v.begin(), v.end() );
	}
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

	{
		ecuda::matrix<Coordinate> deviceMatrix( 4, 5 );
		fetchSliceYZ<<<1,1>>>( deviceCube, deviceMatrix );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		estd::matrix<Coordinate> hostMatrix( 4, 5 );
		deviceMatrix >> hostMatrix;
		for( unsigned i = 0; i < hostMatrix.row_size(); ++i ) {
			std::cout << "SLICE_YZ_ROW";
			for( unsigned j = 0; j < hostMatrix.column_size(); ++j ) {
				std::cout << " " << hostMatrix[i][j];
			}
			std::cout << std::endl;
		}
	}

	{
		ecuda::matrix<Coordinate> deviceMatrix( 3, 4 );
ecuda::cube<Coordinate>::slice_xy_type xy_slice = deviceCube.get_xy(3);
std::cout << "xy_slice.data()=" << xy_slice.data() << std::endl;
		fetchSliceXY<<<1,1>>>( deviceCube, deviceMatrix );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		estd::matrix<Coordinate> hostMatrix( 3, 4 );
		deviceMatrix >> hostMatrix;
		for( unsigned i = 0; i < hostMatrix.row_size(); ++i ) {
			std::cout << "SLICE_XY_ROW";
			for( unsigned j = 0; j < hostMatrix.column_size(); ++j ) {
				std::cout << " " << hostMatrix[i][j];
			}
			std::cout << std::endl;
		}
	}

	{
		ecuda::matrix<Coordinate> deviceMatrix( 3, 5 );
		fetchSliceXZ<<<1,1>>>( deviceCube, deviceMatrix );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		estd::matrix<Coordinate> hostMatrix( 3, 5 );
		deviceMatrix >> hostMatrix;
		for( unsigned i = 0; i < hostMatrix.row_size(); ++i ) {
			std::cout << "SLICE_XZ_ROW";
			for( unsigned j = 0; j < hostMatrix.column_size(); ++j ) {
				std::cout << " " << hostMatrix[i][j];
			}
			std::cout << std::endl;
		}
	}
*/
	{
		ecuda::array<Coordinate,3> deviceArray;
		iterateAll<<<1,1>>>( deviceCube, deviceArray );
		CUDA_CALL( cudaDeviceSynchronize() );
		CUDA_CHECK_ERRORS();
		std::vector<Coordinate> hostArray( 3 );
		deviceArray >> hostArray;
		for( unsigned i = 0; i < hostArray.size(); ++i ) {
			std::cout << "ITERATE_ALL";
			std::cout << " " << hostArray[i];
			std::cout << std::endl;
		}
	}

	{
		ecuda::array<Coordinate,60> deviceArray;
		fetchAll<<<1,1>>>( deviceCube, deviceArray );
//		fetchAll<<<1,1>>>( deviceCube.begin(), deviceCube.end(), deviceArray );
		CUDA_CALL( cudaDeviceSynchronize() );
		CUDA_CHECK_ERRORS();
		std::vector<Coordinate> hostArray( 60 );
		deviceArray >> hostArray;
		for( unsigned i = 0; i < hostArray.size(); ++i ) {
			std::cout << "FETCH_ALL";
			std::cout << " " << hostArray[i];
			std::cout << std::endl;
		}
	}

	return EXIT_SUCCESS;

}

