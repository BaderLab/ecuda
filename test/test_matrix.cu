//#define NDEBUG
//#include <cassert>

#include <iostream>
#include <cstdio>
#include <vector>
#include <estd/matrix.hpp>
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/matrix.hpp"

template<typename T>
struct coord_t {
	T x, y;
	coord_t( const T& x = T(), const T& y = T() ) : x(x), y(y) {}
	bool operator==( const coord_t& other ) const { return x == other.x and y == other.y; }
	bool operator!=( const coord_t& other ) const { return !operator==(other); }
	friend std::ostream& operator<<( std::ostream& out, const coord_t& coord ) {
		out << "[" << coord.x << "," << coord.y << "]";
		return out;
	}
};

typedef coord_t<double> Coordinate;

typedef unsigned char uint8_t;


template<typename T,std::size_t U>
__global__
void fetchRow( const ecuda::matrix<T> matrix, ecuda::array<T,U> array ) {
	typename ecuda::matrix<T>::const_row_type row = matrix[0];
	for( typename ecuda::matrix<T>::const_row_type::size_type i = 0; i < row.size(); ++i ) array[i] = row[i];
}

int main( int argc, char* argv[] ) {

	estd::matrix<Coordinate> hostMatrix( 5, 10 );
	for( estd::matrix<Coordinate>::row_index_type i = 0; i < hostMatrix.row_size(); ++i ) {
		for( estd::matrix<Coordinate>::column_index_type j = 0; j < hostMatrix.column_size(); ++j ) {
			hostMatrix[i][j] = Coordinate( i, j );
		}
	}

	for( estd::matrix<Coordinate>::row_index_type i = 0; i < hostMatrix.row_size(); ++i ) {
		std::cout << "HOST ";
		for( estd::matrix<Coordinate>::column_index_type j = 0; j < hostMatrix.column_size(); ++j ) {
			std::cout << " " << hostMatrix[i][j];
		}
		std::cout << std::endl;
	}

	ecuda::matrix<Coordinate> deviceMatrix( 5, 10 );
	deviceMatrix << hostMatrix;

	deviceMatrix >> hostMatrix;
	for( estd::matrix<Coordinate>::row_index_type i = 0; i < hostMatrix.row_size(); ++i ) {
		std::cout << "DEVICE";
		for( estd::matrix<Coordinate>::column_index_type j = 0; j < hostMatrix.column_size(); ++j ) {
			std::cout << " " << hostMatrix[i][j];
		}
		std::cout << std::endl;
	}

	ecuda::array<Coordinate,10> deviceRow;

	fetchRow<<<1,1>>>( deviceMatrix, deviceRow );
	CUDA_CHECK_ERRORS();
	CUDA_CALL( cudaDeviceSynchronize() );

	std::vector<Coordinate> hostRow;
	deviceRow >> hostRow;

	std::cout << "ROW";
	for( std::vector<Coordinate>::size_type i = 0; i < hostRow.size(); ++i ) std::cout << hostRow[i];
	std::cout << std::endl;

	return EXIT_SUCCESS;

}
