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
	ecuda::vector<typename ecuda::matrix<T>::pointer> pointers
	//ecuda::vector<typename ecuda::matrix<T>::const_pointer> constPointers
)
{
	const int row = blockIdx.x;
	const int column = threadIdx.x;
	if( row < matrix.number_rows() and column < matrix.number_columns() ) {
		const int index = row*matrix.number_columns()+column;
		empties[index] = constMatrix.empty() ? 1 : 0;
		sizes[index] = constMatrix.size();
		pointers[index] = matrix.data();
		//constPointers[index] = constMatrix.data();
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
// removed temporarily from spec
//	for( typename ecuda::matrix<T>::size_type i = 0; i < matrixIn.number_rows(); ++i ) {
//		typename ecuda::matrix<T>::const_row_type row = matrixIn[i];
//		matrixOut1[i].assign( row.begin(), row.end() );
//	}
//	for( typename ecuda::matrix<T>::size_type i = 0; i < matrixIn.number_columns(); ++i ) {
//		typename ecuda::matrix<T>::const_column_type column = matrixIn.get_column(i);
//		matrixOut2[i].assign( column.begin(), column.end() );
//	}
}

template<typename T> __global__
void kernel_testRowView(
	typename ecuda::matrix<T>::row_type row
)
{
	row.fill(Coordinate(99,99));
}


template<typename T> __global__
void kernel_linearize(
	const ecuda::matrix<T> matrix, ecuda::vector<T> vector
)
{
	std::size_t index = 0;
	for( typename ecuda::matrix<T>::const_iterator iter = matrix.begin(); iter != matrix.end(); ++iter, ++index ) vector[index] = *iter;
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
std::cerr << "cpA" << std::endl;
			if( deviceMatrix.size() ) passed = false;
std::cerr << "cpB" << std::endl;
			if( !deviceMatrix.empty() ) passed = false;
std::cerr << "cpC" << std::endl;
			if( deviceMatrix.number_rows() ) passed = false;
std::cerr << "cpD" << std::endl;
			if( deviceMatrix.number_columns() ) passed = false;
std::cerr << "cpE" << std::endl;
		}
		{
			const ecuda::matrix<int> deviceMatrix( 10, 20 );
			if( deviceMatrix.size() != 200 ) passed = false;
			if( deviceMatrix.empty() ) passed = false;
			if( !deviceMatrix.data() ) passed = false;
std::cerr << "cp1" << std::endl;
			std::vector<int> hostVector( 200 );
std::cerr << "cp2" << std::endl;
			deviceMatrix >> hostVector;
std::cerr << "cp3" << std::endl;
			if( hostVector.size() != 200 ) passed = false;
std::cerr << "cp4" << std::endl;
			for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] ) passed = false;
std::cerr << "cp5" << std::endl;
		}
		ecuda::matrix<int> deviceMatrix( 10, 20, 3 );
std::cerr << "cp1" << std::endl;
		if( deviceMatrix.size() != 200 ) passed = false;
		if( deviceMatrix.empty() ) passed = false;
		if( !deviceMatrix.data() ) passed = false;
		std::vector<int> hostVector( 200 );
		deviceMatrix >> hostVector;
std::cerr << "cp1" << std::endl;
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
		//ecuda::vector<ecuda::matrix<Coordinate>::const_pointer> deviceConstPointers( 200 );
		kernel_checkMatrixProperties<<<10,20>>>( deviceMatrix, deviceMatrix, deviceEmpties, deviceSizes, devicePointers ); //, deviceConstPointers );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::vector<int> hostEmpties( 200, -1 );
		std::vector<ecuda::matrix<Coordinate>::size_type> hostSizes( 200 );
		std::vector<ecuda::matrix<Coordinate>::pointer> hostPointers( 200 );
		//std::vector<ecuda::matrix<Coordinate>::const_pointer> hostConstPointers( 200 );
		deviceEmpties >> hostEmpties;
		deviceSizes >> hostSizes;
		devicePointers >> hostPointers;
		//deviceConstPointers >> hostConstPointers;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostEmpties.size(); ++i ) if( hostEmpties[i] != 0 ) passed = false;
		for( std::vector<ecuda::vector<int>::size_type>::size_type i = 0; i < hostSizes.size(); ++i ) if( hostSizes[i] != 200 ) passed = false;
		for( std::vector<ecuda::vector<int>::pointer>::size_type i = 0; i < hostPointers.size(); ++i ) if( hostPointers[i] != deviceMatrix.data() ) passed = false;
		//for( std::vector<ecuda::vector<int>::const_pointer>::size_type i = 0; i < hostConstPointers.size(); ++i ) if( hostConstPointers[i] != deviceMatrix.data() ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	// Test 3: C++11 assignment
	#ifdef __CPP11_SUPPORTED__
	{
		std::cerr << "Test 3" << std::endl;
		ecuda::matrix<Coordinate> deviceMatrix( 2, 2 );
		deviceMatrix.assign( { Coordinate(0,0), Coordinate(0,1), Coordinate(1,0), Coordinate(1,1) } );
		std::vector<Coordinate> hostVector( 4 );
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
		ecuda::array<int,1> deviceArray( 0 );
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
/** removing this test for now
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
***/
		testResults.push_back( -1 );

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
		ecuda::vector<Coordinate> deviceVector( 10*20 );
		kernel_linearize<<<1,1>>>( deviceMatrix, deviceVector );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );

		hostVector.clear();
		hostVector.resize( 200 );
		//hostVector.assign( 200, Coordinate() );
		deviceVector >> hostVector;
		for( std::size_t i = 0; i < hostVector.size(); ++i ) {
			std::cout << "LINEAR " << hostVector[i] << std::endl;
		}

	}

	for( std::vector<bool>::size_type i = 0; i < testResults.size(); ++i ) std::cout << ( testResults[i] == 1 ? "P" : ( testResults[i] == -1 ? "?" : "F" ) ) << "|";
	std::cout << std::endl;

	return EXIT_SUCCESS;

}
