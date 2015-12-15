<<<<<<< HEAD
//#define NDEBUG
//#include <cassert>

#include <iostream>
#include <cstdio>
#include <vector>
//#include <estd/matrix.hpp>
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/matrix.hpp"

template<typename T>
struct coord_t {
	T x, y;
	HOST DEVICE coord_t( const T& x = T(), const T& y = T() ) : x(x), y(y) {}
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


template<typename T> __global__
void kernel_checkDeviceIterators(
	const ecuda::matrix<T> srcMatrix,
	ecuda::matrix<T> destMatrix
)
{
	typename ecuda::matrix<T>::const_iterator srcIterator = srcMatrix.begin();
	typename ecuda::matrix<T>::iterator destIterator = destMatrix.begin();
	for( ; srcIterator != srcMatrix.end() and destIterator != destMatrix.end(); ++srcIterator, ++destIterator ) *destIterator = *srcIterator;
}

template<typename T> __global__
void kernel_checkHostIterators(
	typename ecuda::matrix<T>::const_iterator srcBegin,
	typename ecuda::matrix<T>::const_iterator srcEnd,
	typename ecuda::matrix<T>::iterator destBegin,
	typename ecuda::matrix<T>::iterator destEnd
)
{
	for( ; srcBegin != srcEnd and destBegin != destEnd; ++srcBegin, ++destBegin )
		*destBegin = *srcBegin;
}

template<typename T> __global__
void kernel_checkDeviceReverseIterators(
	const ecuda::matrix<T> srcMatrix,
	ecuda::matrix<T> destMatrix
)
{
	typename ecuda::matrix<T>::const_reverse_iterator srcIterator = srcMatrix.rbegin();
	typename ecuda::matrix<T>::reverse_iterator destIterator = destMatrix.rbegin();
	for( ; srcIterator != srcMatrix.rend() and destIterator != destMatrix.rend(); ++srcIterator, ++destIterator ) *destIterator = *srcIterator;
}

template<typename T> __global__
void kernel_checkHostReverseIterators(
	typename ecuda::matrix<T>::const_reverse_iterator srcBegin,
	typename ecuda::matrix<T>::const_reverse_iterator srcEnd,
	typename ecuda::matrix<T>::reverse_iterator destBegin,
	typename ecuda::matrix<T>::reverse_iterator destEnd
)
{
	for( ; srcBegin != srcEnd and destBegin != destEnd; ++srcBegin, ++destBegin )
		*destBegin = *srcBegin;
}

template<typename T> __global__
void kernel_testSwapAndFill(
	ecuda::matrix<T> matrix1,
	ecuda::matrix<T> matrix2,
	ecuda::matrix<T> dummyMatrix,
	ecuda::array<int,1> swapResult
)
{
	dummyMatrix.fill( 99 );
	T oldStart1 = matrix1.front();
	T oldStart2 = matrix2.front();
	matrix1.swap( matrix2 );
	if( matrix2.front() == oldStart1 and matrix1.front() == oldStart2 ) swapResult.front() = 1;
}

template<typename T> __global__
void kernel_testDeviceRowsAndColumns(
	const ecuda::matrix<T> matrixIn,
	ecuda::matrix<T> matrixOut1,
	ecuda::matrix<T> matrixOut2
)
{
	for( typename ecuda::matrix<T>::size_type i = 0; i < matrixIn.number_rows(); ++i ) {
		typename ecuda::matrix<T>::const_row_type row = matrixIn[i];
		matrixOut1[i].assign( row.begin(), row.end() );
	}
	for( typename ecuda::matrix<T>::size_type i = 0; i < matrixIn.number_columns(); ++i ) {
		typename ecuda::matrix<T>::const_column_type column = matrixIn.get_column(i);
		matrixOut2[i].assign( column.begin(), column.end() );
	}
}

template<typename T> __global__
void kernel_testRowView(
	typename ecuda::matrix<T>::row_type row
)
{
	row.fill(Coordinate(99,99));
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
			std::vector<int> hostVector( 200 );
			deviceMatrix >> hostVector;
			if( hostVector.size() != 200 ) passed = false;
			for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] ) passed = false;
		}
		ecuda::matrix<int> deviceMatrix( 10, 20, 3 );
		if( deviceMatrix.size() != 200 ) passed = false;
		if( deviceMatrix.empty() ) passed = false;
		if( !deviceMatrix.data() ) passed = false;
		std::vector<int> hostVector( 200 );
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
		std::vector<Coordinate> hostResults( 200 );

		destDeviceMatrix >> hostResults;

		for( std::vector<Coordinate>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != Coordinate(i/20,i%20) ) passed = false;

		deviceFronts >> hostResults;
		for( std::vector<Coordinate>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != Coordinate(0,0) ) passed = false;

		deviceBacks >> hostResults;
		for( std::vector<Coordinate>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != Coordinate(9,19) ) passed = false;

		deviceFrontsNonConst >> hostResults;
		for( std::vector<Coordinate>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != Coordinate(0,0) ) passed = false;

		deviceBacksNonConst >> hostResults;
		for( std::vector<Coordinate>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != Coordinate(9,19) ) passed = false;

		testResults.push_back( passed ? 1 : 0 );

	}

	// Test 5: check device iterators
	std::cerr << "Test 5" << std::endl;
	{
		std::vector<Coordinate> hostVector( 10*20 );
		unsigned index = 0;
		for( unsigned i = 0; i < 10; ++i ) {
			for( unsigned j = 0; j < 20; ++j, ++index ) {
				hostVector[index] = Coordinate(i,j);
			}
		}
		ecuda::matrix<Coordinate> srcDeviceMatrix( 10, 20 );
		srcDeviceMatrix.assign( hostVector.begin(), hostVector.end() );
		ecuda::matrix<Coordinate> destDeviceMatrix( 10, 20 );
		kernel_checkDeviceIterators<<<1,1>>>( srcDeviceMatrix, destDeviceMatrix );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::fill( hostVector.begin(), hostVector.end(), Coordinate(9000,9000) );
		destDeviceMatrix >> hostVector;
		bool passed = true;
		for( std::vector<Coordinate>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != Coordinate(i/20,i%20) ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	// Test 6: check host iterators
	std::cerr << "Test 6" << std::endl;
	{
		std::vector<Coordinate> hostVector( 10*20 );
		unsigned index = 0;
		for( unsigned i = 0; i < 10; ++i ) {
			for( unsigned j = 0; j < 20; ++j, ++index ) {
				hostVector[index] = Coordinate(i,j);
			}
		}
		ecuda::matrix<Coordinate> srcDeviceMatrix( 10, 20 );
		srcDeviceMatrix.assign( hostVector.begin(), hostVector.end() );
		ecuda::matrix<Coordinate> destDeviceMatrix( 10, 20 );
		kernel_checkHostIterators<Coordinate><<<1,1>>>( srcDeviceMatrix.begin(), srcDeviceMatrix.end(), destDeviceMatrix.begin(), destDeviceMatrix.end() );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::fill( hostVector.begin(), hostVector.end(), Coordinate(9000,9000) );
		destDeviceMatrix >> hostVector;
		bool passed = true;
		for( std::vector<Coordinate>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != Coordinate(i/20,i%20) ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	// Test 7: check device reverse iterators
	std::cerr << "Test 7" << std::endl;
	{
		std::vector<Coordinate> hostVector( 10*20 );
		unsigned index = 0;
		for( unsigned i = 0; i < 10; ++i ) {
			for( unsigned j = 0; j < 20; ++j, ++index ) {
				hostVector[index] = Coordinate(i,j);
			}
		}
		ecuda::matrix<Coordinate> srcDeviceMatrix( 10, 20 );
		srcDeviceMatrix.assign( hostVector.begin(), hostVector.end() );
		ecuda::matrix<Coordinate> destDeviceMatrix( 10, 20 );
		kernel_checkDeviceReverseIterators<<<1,1>>>( srcDeviceMatrix, destDeviceMatrix );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::fill( hostVector.begin(), hostVector.end(), Coordinate(9000,9000) );
		destDeviceMatrix >> hostVector;
		bool passed = true;
		for( std::vector<Coordinate>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != Coordinate(i/20,i%20) ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	// Test 8: check host reverse iterators
	std::cerr << "Test 8" << std::endl;
	{
		std::vector<Coordinate> hostVector( 10*20 );
		unsigned index = 0;
		for( unsigned i = 0; i < 10; ++i ) {
			for( unsigned j = 0; j < 20; ++j, ++index ) {
				hostVector[index] = Coordinate(i,j);
			}
		}
		ecuda::matrix<Coordinate> srcDeviceMatrix( 10, 20 );
		srcDeviceMatrix.assign( hostVector.begin(), hostVector.end() );
		ecuda::matrix<Coordinate> destDeviceMatrix( 10, 20 );
		kernel_checkHostReverseIterators<Coordinate><<<1,1>>>( srcDeviceMatrix.rbegin(), srcDeviceMatrix.rend(), destDeviceMatrix.rbegin(), destDeviceMatrix.rend() );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::fill( hostVector.begin(), hostVector.end(), Coordinate(9000,9000) );
		destDeviceMatrix >> hostVector;
		bool passed = true;
		for( std::vector<Coordinate>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != Coordinate(i/20,i%20) ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 9: host swap and fill
	//
	std::cerr << "Test 9" << std::endl;
	{
		ecuda::matrix<int> deviceMatrix1( 10, 20, 3 );
		ecuda::matrix<int> deviceMatrix2( 10, 20, 5 );
		deviceMatrix1.swap( deviceMatrix2 );
		std::vector<int> hostVector1( 10*20 );
		std::vector<int> hostVector2( 10*20 );
		deviceMatrix1 >> hostVector1;
		deviceMatrix2 >> hostVector2;
		bool passed = true;
		if( hostVector1.size() != 200 ) passed = false;
		if( hostVector2.size() != 200 ) passed = false;
		for( std::vector<int>::size_type i = 0; i < hostVector1.size(); ++i ) if( hostVector1[i] != 5 ) passed = false;
		for( std::vector<int>::size_type i = 0; i < hostVector2.size(); ++i ) if( hostVector2[i] != 3 ) passed = false;
		deviceMatrix1.fill(0);
		deviceMatrix2.fill(99);
		deviceMatrix1 >> hostVector1;
		deviceMatrix2 >> hostVector2;
		for( std::vector<int>::size_type i = 0; i < hostVector1.size(); ++i ) if( hostVector1[i] != 0 ) passed = false;
		for( std::vector<int>::size_type i = 0; i < hostVector2.size(); ++i ) if( hostVector2[i] != 99 ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test A: device swap and fill
	//
	std::cerr << "Test A" << std::endl;
	{
		ecuda::matrix<int> deviceMatrix1( 10, 20, 3 );
		ecuda::matrix<int> deviceMatrix2( 10, 20, 5 );
		ecuda::matrix<int> deviceDummyMatrix( 10, 20 );
		ecuda::array<int,1> deviceArray;
		kernel_testSwapAndFill<<<1,1>>>( deviceMatrix1, deviceMatrix2, deviceDummyMatrix, deviceArray );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		//std::vector<int> hostVector1; deviceMatrix1 >> hostVector1;
		//std::vector<int> hostVector2; deviceMatrix2 >> hostVector2;
		std::vector<int> hostArray( 1 ); deviceArray >> hostArray;
		std::vector<int> hostDummyVector( 200 ); deviceDummyMatrix >> hostDummyVector;
		bool passed = hostArray.size() == 1 and hostArray.front() == 1;
		//for( std::vector<int>::size_type i = 0; i < hostVector1.size(); ++i ) if( hostVector1[i] != 5 ) passed = false;
		//for( std::vector<int>::size_type i = 0; i < hostVector2.size(); ++i ) if( hostVector2[i] != 3 ) passed = false;
		for( std::vector<int>::size_type i = 0; i < hostDummyVector.size(); ++i ) if( hostDummyVector[i] != 99 ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	// Test B: device rows and columns
	std::cerr << "Test B" << std::endl;
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
		ecuda::matrix<Coordinate> deviceOutputMatrix1( 10, 20 );
		ecuda::matrix<Coordinate> deviceOutputMatrix2( 20, 10 );
		kernel_testDeviceRowsAndColumns<<<1,1>>>( deviceMatrix, deviceOutputMatrix1, deviceOutputMatrix2 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );

		bool passed = true;

		deviceOutputMatrix1 >> hostVector;
		for( std::vector<Coordinate>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != Coordinate(i/20,i%20) ) passed = false;

		deviceOutputMatrix2 >> hostVector;
		for( std::vector<Coordinate>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != Coordinate(i%10,i/10) ) passed = false;

		testResults.push_back( passed ? 1 : 0 );

	}

	// Test C: host rows and columns
	std::cerr << "Test C" << std::endl;
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
		ecuda::matrix<Coordinate> deviceOutputMatrix1( 10, 20 );
		//ecuda::matrix<Coordinate> deviceOutputMatrix2( 20, 10 );

		for( typename ecuda::matrix<Coordinate>::size_type i = 0; i < deviceMatrix.number_rows(); ++i ) {
			typename ecuda::matrix<Coordinate>::const_row_type row = deviceMatrix[i];
			deviceOutputMatrix1[i].assign( row.begin(), row.end() );
		}
		// is not allowed because columns aren't contiguous memory
		//for( typename ecuda::matrix<Coordinate>::size_type i = 0; i < deviceMatrix.number_columns(); ++i ) {
		//	typename ecuda::matrix<Coordinate>::const_column_type column = deviceMatrix.get_column(i);
		//	deviceOutputMatrix2[i].assign( column.begin(), column.end() );
		//}

		bool passed = true;

		deviceOutputMatrix1 >> hostVector;
		for( std::vector<Coordinate>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != Coordinate(i/20,i%20) ) passed = false;
		//deviceOutputMatrix2 >> hostVector;
		//for( std::vector<Coordinate>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != Coordinate(i%10,i/10) ) passed = false;
		testResults.push_back( passed ? 1 : 0 );

	}

	// Test D: views
	std::cerr << "Test D" << std::endl;
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
		ecuda::matrix<Coordinate>::row_type row = deviceMatrix[1];
		kernel_testRowView<Coordinate><<<1,1>>>( row );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );

		bool passed = true;
		deviceMatrix >> hostVector;
		for( std::vector<Coordinate>::size_type i = 0; i < hostVector.size(); ++i ) {
			if( i/20 == 1 ) {
				if( hostVector[i] != Coordinate(99,99) ) passed = false;
				continue;
			}
			if( hostVector[i] != Coordinate(i/20,i%20) ) passed = false;
		}
		testResults.push_back( passed ? 1 : 0 );

	}

	for( std::vector<bool>::size_type i = 0; i < testResults.size(); ++i ) std::cout << ( testResults[i] == 1 ? "P" : ( testResults[i] == -1 ? "?" : "F" ) ) << "|";
	std::cout << std::endl;
=======
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

#ifdef __CUDACC__
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

template<typename T,class Alloc>
void print_matrix( const ecuda::matrix<T,Alloc>& mat )
{
	for( std::size_t i = 0; i < mat.number_rows(); ++i ) {
		std::cout << "ROW[" << i << "]";
		for( std::size_t j = 0; j < mat.number_columns(); ++j ) {
			std::cout << " " << mat[i][j];
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

template<typename T,class Alloc>
void dummyFunction( typename ecuda::matrix<T,Alloc>::kernel_argument arg )
{
	std::cout << "matrix::kernel_argument=[" << arg.number_rows() << "x" << arg.number_columns() << "]" << std::endl;
	std::cout << "KERNEL:" << std::endl;
	for( std::size_t i = 0; i < arg.number_rows(); ++i ) {
		std::cout << "ROW[" << i << "]";
		for( std::size_t j = 0; j < arg.number_columns(); ++j ) std::cout << " " << arg[i][j];
		std::cout << std::endl;
	}
	arg[arg.number_rows()-1][arg.number_columns()-1] = 33.0;
	arg(0,0) = 66.0;
}

int main( int argc, char* argv[] ) {

	{
		estd::matrix<int> hostMatrix(5,5);
		ecuda::matrix<int> deviceMatrix(5,5);
		ecuda::copy( hostMatrix.begin(), hostMatrix.end(), deviceMatrix.begin() );
		ecuda::copy( deviceMatrix.begin(), deviceMatrix.end(), hostMatrix.begin() );
		ecuda::copy( hostMatrix.begin(), hostMatrix.end(), hostMatrix.begin() );
		ecuda::copy( deviceMatrix.begin(), deviceMatrix.end(), deviceMatrix.begin() );
	}
	{
		const std::size_t R = 5;
		const std::size_t C = 21;
		ecuda::matrix<double> deviceMatrix( R, C, 99.0 );
		dummyFunction<double,typename ecuda::matrix<double>::allocator_type>( deviceMatrix );
		std::cout << "ecuda::distance=" << ecuda::distance( deviceMatrix.begin(), deviceMatrix.end() ) << std::endl;
		for( std::size_t i = 0; i < R; ++i ) *(deviceMatrix[i].begin()+3) = 33.0;

		for( std::size_t i = 0; i < deviceMatrix.size(); ++i ) *(deviceMatrix.begin()+i) = 11.0;
		for( std::size_t i = 0; i < deviceMatrix.size(); ++i ) {
			typename ecuda::matrix<double>::iterator::difference_type delta = (deviceMatrix.begin()+i) - deviceMatrix.begin();
			if( static_cast<int>(i) != delta ) throw std::runtime_error("");
			//std::cout << "SANITY\t" << i << "\t" << delta << std::endl;
		}

		std::cout << "HOST:" << std::endl;
		for( std::size_t i = 0; i < deviceMatrix.number_rows(); ++i ) {
			std::cout << "ROW[" << i << "]";
			for( std::size_t j = 0; j < deviceMatrix.number_columns(); ++j ) std::cout << " " << deviceMatrix[i][j];
			std::cout << std::endl;
		}
	}
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
//print_matrix( deviceMatrix1 );
			ecuda::matrix<double> deviceMatrix2( deviceMatrix1 );
//print_matrix( deviceMatrix2 );
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
			#ifdef __CUDACC__
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
			#else
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
		#ifndef __CUDACC__
		for( std::size_t i = 0; i < R; ++i ) {
			typename ecuda::matrix< coord_t<int> >::row_type row = deviceMatrix[i];
			std::cerr << "row[" << i << "]=" << std::boolalpha << std::equal( row.begin(), row.end(), hostVector.begin()+(i*C) ) << std::endl;
		}
		#endif
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
		#ifndef __CUDACC__
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
>>>>>>> ecuda2/master

	return EXIT_SUCCESS;

}
