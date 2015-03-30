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
	typename ecuda::matrix<T>::const_row_type row = matrix[1];
	for( typename ecuda::matrix<T>::const_row_type::size_type i = 0; i < row.size(); ++i ) array[i] = row[i];
}

template<typename T,std::size_t U>
__global__
void fetchColumn( const ecuda::matrix<T> matrix, ecuda::array<T,U> array ) {
	typename ecuda::matrix<T>::const_column_type column = matrix.get_column(1);
	for( typename ecuda::matrix<T>::const_column_type::size_type i = 0; i < column.size(); ++i ) array[i] = column[i];
}

template<typename T,std::size_t U>
__global__
void fetchAll( const ecuda::matrix<T> matrix, ecuda::array<T,U> array ) {
	unsigned index = 0;
	//for( T x : matrix ) { array[index] = x; ++index; }
	for( typename ecuda::matrix<T>::const_reverse_iterator iter = matrix.rbegin(); iter != matrix.rend(); ++iter, ++index ) array[index] = *iter;
}


template<typename T> __global__
void kernel_checkMatrixProperties(
	const ecuda::matrix<T> constMatrix,
	ecuda::matrix<T> matrix,
	ecuda::vector<int> empties,
	ecuda::vector<typename ecuda::matrix<T>::size_type> sizes,
	ecuda::vector<typename ecuda::matrix<T>::pointer> pointers,
	ecuda::vector<typename ecuda::matrix<T>::const_pointer> constPointers
)
{
	const int row = blockIdx.x;
	const int column = threadIdx.x;
	if( row < matrix.number_rows() and column < matrix.number_columns() ) {
		const int index = row*matrix.number_columns()+column;
		empties[index] = constMatrix.empty() ? 1 : 0;
		sizes[index] = constMatrix.size();
		pointers[index] = matrix.data();
		constPointers[index] = constMatrix.data();
	}
}


int main( int argc, char* argv[] ) {

	std::cout << "Testing ecuda::matrix..." << std::endl;

	std::vector<int> testResults;

	// Test 1: default constructor, copy to host and general info
	std::cerr << "Test 1" << std::endl;
	{
		bool passed = true;
		{
			ecuda::matrix<int> deviceMatrix;
			if( deviceMatrix.size() ) passed = false;
			if( !deviceMatrix.empty() ) passed = false;
			if( deviceMatrix.number_rows() ) passed = false;
			if( deviceMatrix.number_columns() ) passed = false;
		}
		{
			const ecuda::matrix<int> deviceMatrix( 10, 20 );
			if( deviceMatrix.size() != 200 ) passed = false;
			if( deviceMatrix.empty() ) passed = false;
			if( !deviceMatrix.data() ) passed = false;
			std::vector<int> hostVector;
			deviceMatrix >> hostVector;
			if( hostVector.size() != 200 ) passed = false;
			for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] ) passed = false;
		}
		ecuda::matrix<int> deviceMatrix( 10, 20, 3 );
		if( deviceMatrix.size() != 200 ) passed = false;
		if( deviceMatrix.empty() ) passed = false;
		if( !deviceMatrix.data() ) passed = false;
		std::vector<int> hostVector;
		deviceMatrix >> hostVector;
		if( hostVector.size() != 200 ) passed = false;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != 3 ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	// Test 2: information is correct on device
	std::cerr << "Test 2" << std::endl;
	{
		std::vector<Coordinate> hostVector( 200 );
		std::vector<Coordinate>::size_type index = 0;
		for( unsigned i = 0; i < 10; ++i )
			for( unsigned j = 0; j < 20; ++j, ++index )
				hostVector[i] = Coordinate(i,j);
		ecuda::matrix<Coordinate> deviceMatrix( 10, 20 );
		deviceMatrix.assign( hostVector.begin(), hostVector.end() );
		ecuda::vector<int> deviceEmpties( 200, -1 );
		ecuda::vector<ecuda::matrix<Coordinate>::size_type> deviceSizes( 200 );
		ecuda::vector<ecuda::matrix<Coordinate>::pointer> devicePointers( 200 );
		ecuda::vector<ecuda::matrix<Coordinate>::const_pointer> deviceConstPointers( 200 );
		kernel_checkMatrixProperties<<<10,20>>>( deviceMatrix, deviceMatrix, deviceEmpties, deviceSizes, devicePointers, deviceConstPointers );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::vector<int> hostEmpties( 200, -1 );
		std::vector<ecuda::matrix<Coordinate>::size_type> hostSizes( 200 );
		std::vector<ecuda::matrix<Coordinate>::pointer> hostPointers( 200 );
		std::vector<ecuda::matrix<Coordinate>::const_pointer> hostConstPointers( 200 );
		deviceEmpties >> hostEmpties;
		deviceSizes >> hostSizes;
		devicePointers >> hostPointers;
		deviceConstPointers >> hostConstPointers;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostEmpties.size(); ++i ) if( hostEmpties[i] != 0 ) passed = false;
		for( std::vector<ecuda::vector<int>::size_type>::size_type i = 0; i < hostSizes.size(); ++i ) if( hostSizes[i] != 100 ) passed = false;
		for( std::vector<ecuda::vector<int>::pointer>::size_type i = 0; i < hostPointers.size(); ++i ) if( hostPointers[i] != deviceMatrix.data() ) passed = false;
		for( std::vector<ecuda::vector<int>::const_pointer>::size_type i = 0; i < hostConstPointers.size(); ++i ) if( hostConstPointers[i] != deviceMatrix.data() ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

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

	{
		std::vector<Coordinate> coordinates( deviceMatrix.number_columns() );
		for( ecuda::matrix<Coordinate>::size_type i = 0; i < deviceMatrix.number_rows(); ++i )
			deviceMatrix[i].assign( coordinates.begin(), coordinates.end() );
			//deviceMatrix.assign_row( i, coordinates.begin(), coordinates.end() );
	}

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

	ecuda::array<Coordinate,5> deviceColumn;

	fetchColumn<<<1,1>>>( deviceMatrix, deviceColumn );
	CUDA_CHECK_ERRORS();
	CUDA_CALL( cudaDeviceSynchronize() );

	std::vector<Coordinate> hostColumn;
	deviceColumn >> hostColumn;

	std::cout << "COLUMN";
	for( std::vector<Coordinate>::size_type i = 0; i < hostColumn.size(); ++i ) std::cout << hostColumn[i];
	std::cout << std::endl;

	ecuda::array<Coordinate,50> deviceLinearMatrix;
	fetchAll<<<1,1>>>( deviceMatrix, deviceLinearMatrix );
	CUDA_CHECK_ERRORS();
	CUDA_CALL( cudaDeviceSynchronize() );

	std::vector<Coordinate> hostLinearMatrix;
	deviceLinearMatrix >> hostLinearMatrix;

	std::cout << "LINEARIZED" << std::endl;
	for( std::vector<Coordinate>::size_type i = 0; i < hostLinearMatrix.size(); ++i ) std::cout << hostLinearMatrix[i] << std::endl;


	return EXIT_SUCCESS;

}
