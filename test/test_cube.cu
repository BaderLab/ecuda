#include <iostream>
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
	return EXIT_SUCCESS;

}

