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

template<typename T> __global__
void kernel_checkMatrixAccessors(
	const ecuda::matrix<T> srcMatrix,
	ecuda::matrix<T> srcMatrixNonConst,
	ecuda::matrix<T> destMatrix,
	ecuda::vector<T> srcFronts,
	ecuda::vector<T> srcBacks,
	ecuda::vector<T> srcFrontsNonConst,
	ecuda::vector<T> srcBacksNonConst
)
{
	const int row = blockIdx.x;
	const int column = threadIdx.x;
	if( row < srcMatrix.number_rows() and column < srcMatrix.number_columns() ) {
		const int index = row*srcMatrix.number_columns()+column;
		destMatrix[row][column] = srcMatrix[row][column];
		srcFronts[index] = srcMatrix.front();
		srcBacks[index] = srcMatrix.back();
		srcFrontsNonConst[index] = srcMatrixNonConst.front();
		srcBacksNonConst[index] = srcMatrixNonConst.back();
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
		for( std::vector<ecuda::vector<int>::size_type>::size_type i = 0; i < hostSizes.size(); ++i ) if( hostSizes[i] != 200 ) passed = false;
		for( std::vector<ecuda::vector<int>::pointer>::size_type i = 0; i < hostPointers.size(); ++i ) if( hostPointers[i] != deviceMatrix.data() ) passed = false;
		for( std::vector<ecuda::vector<int>::const_pointer>::size_type i = 0; i < hostConstPointers.size(); ++i ) if( hostConstPointers[i] != deviceMatrix.data() ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	// Test 3: C++11 assignment
	#ifdef __CPP11_SUPPORTED__
	{
		std::cerr << "Test 3" << std::endl;
		ecuda::matrix<Coordinate> deviceMatrix( 2, 2 );
		deviceMatrix.assign( { Coordinate(0,0), Coordinate(0,1), Coordinate(1,0), Coordinate(1,1) } );
		std::vector<Coordinate> hostVector;
		deviceMatrix >> hostVector;
		bool passed = true;
		std::vector<Coordinate>::size_type index = 0;
		for( std::vector<Coordinate>::size_type i = 0; i < 2; ++i ) {
			for( std::vector<Coordinate>::size_type j = 0; j < 2; ++j, ++index ) {
				if( hostVector[index] != Coordinate(i,j) ) passed = false;
			}
		}
		testResults.push_back( passed ? 1 : 0 );
	}
	#else
	std::cerr << "Test 3 (skipped)" << std::endl;
	testResults.push_back( -1 );
	#endif

	// Test 4: index accessors, front(), and back()
	std::cerr << "Test 4" << std::endl;
	{
		std::vector<Coordinate> hostVector( 10*20 );
		unsigned index = 0;
		for( unsigned i = 0; i < 10; ++i ) {
			for( unsigned j = 0; j < 20; ++j, ++index ) {
				hostVector[index] = Coordinate(i,j);
			}
		}
		ecuda::matrix<Coordinate> deviceMatrix( 10, 20 );
		deviceMatrix.assign( hostVector.begin(), hostVector.end() );
		ecuda::matrix<Coordinate> destDeviceMatrix( 10, 20 );
		ecuda::vector<Coordinate> deviceFronts( 10*20, -1 );
		ecuda::vector<Coordinate> deviceBacks( 10*20, -1 );
		ecuda::vector<Coordinate> deviceFrontsNonConst( 10*20, -1 );
		ecuda::vector<Coordinate> deviceBacksNonConst( 10*20, -1 );
		kernel_checkMatrixAccessors<<<10,20>>>( deviceMatrix, deviceMatrix, destDeviceMatrix, deviceFronts, deviceBacks, deviceFrontsNonConst, deviceBacksNonConst );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );

		bool passed = true;
		std::vector<Coordinate> hostResults;

		destDeviceMatrix >> hostResults;
for( std::vector<Coordinate>::size_type i = 0; i < hostResults.size(); ++i ) {
	std::cout << "[" << i << "] COORDINATE ( " << hostResults[i].x << "," << hostResults[i].y << " )" << std::endl;
}
		for( std::vector<Coordinate>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != Coordinate(i/20,i%20) ) passed = false;
std::cerr << "passed = " << ( passed ? "true" : "false" ) << std::endl;

		deviceFronts >> hostResults;
		for( std::vector<Coordinate>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != Coordinate(0,0) ) passed = false;
std::cerr << "passed = " << ( passed ? "true" : "false" ) << std::endl;

		deviceBacks >> hostResults;
		for( std::vector<Coordinate>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != Coordinate(9,19) ) passed = false;
std::cerr << "passed = " << ( passed ? "true" : "false" ) << std::endl;

		deviceFrontsNonConst >> hostResults;
		for( std::vector<Coordinate>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != Coordinate(0,0) ) passed = false;
std::cerr << "passed = " << ( passed ? "true" : "false" ) << std::endl;

		deviceBacksNonConst >> hostResults;
		for( std::vector<Coordinate>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != Coordinate(9,19) ) passed = false;
std::cerr << "passed = " << ( passed ? "true" : "false" ) << std::endl;

		testResults.push_back( passed ? 1 : 0 );

	}

	for( std::vector<bool>::size_type i = 0; i < testResults.size(); ++i ) std::cout << ( testResults[i] == 1 ? "P" : ( testResults[i] == -1 ? "?" : "F" ) ) << "|";

	return EXIT_SUCCESS;

}
