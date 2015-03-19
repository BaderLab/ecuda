#include <algorithm>
#include <iostream>
#include <cstdio>
#include "../include/ecuda/array.hpp"

#ifdef __CPP11_SUPPORTED__
#include <array>
#endif

template<typename T,std::size_t U>
__global__ void multiplyVectors( ecuda::array<T,U> array1, const ecuda::array<T,U> array2 ) {
	const int index = threadIdx.x;
	array1[index] *= array2[index];
}

template<typename T,std::size_t U>
__global__ void testIterators( const ecuda::array<T,U> src, ecuda::array<T,U> dest ) {
	const int index = threadIdx.x;
	typename ecuda::array<T,U>::const_iterator srcIterator = src.begin();
	srcIterator += index;
	typename ecuda::array<T,U>::iterator destIterator = dest.begin();
	destIterator += index;
	for( ; srcIterator != src.end(); ++srcIterator ) *destIterator += *srcIterator;
}

template<typename T,std::size_t U>
__global__ void testReverseIterators( const ecuda::array<T,U> src, ecuda::array<T,U> dest ) {
	const int index = threadIdx.x;
	typename ecuda::array<T,U>::const_reverse_iterator srcIterator = src.rbegin();
	srcIterator += index;
	typename ecuda::array<T,U>::reverse_iterator destIterator = dest.rbegin();
	destIterator += index;
	for( ; srcIterator != src.rend(); ++srcIterator ) *destIterator += *srcIterator;
}

template<typename T,std::size_t U> __global__ void testFill( ecuda::array<T,U> array ) { array.fill( 5 ); }
template<typename T,std::size_t U> __global__ void testSwap( ecuda::array<T,U> array1, ecuda::array<T,U> array2 ) { array1.swap(array2); }
template<typename T,std::size_t U> __global__ void testEquality( const ecuda::array<T,U> array1, const ecuda::array<T,U> array2, ecuda::array<int,1> result ) { result.front() = array1 == array2 ? 1 : 0; }
template<typename T,std::size_t U> __global__ void testGreaterThan( const ecuda::array<T,U> array1, const ecuda::array<T,U> array2, ecuda::array<int,1> result ) { result.front() = array1 > array2 ? 1 : 0; }
template<typename T,std::size_t U> __global__ void testLessThan( const ecuda::array<T,U> array1, const ecuda::array<T,U> array2, ecuda::array<int,1> result ) { result.front() = array1 < array2 ? 1 : 0; }

template<class Container1,class Container2>
bool compare_containers( const Container1& c1, const Container2& c2 ) {
	if( c1.size() != c2.size() ) return false;
	typename Container1::const_iterator iter1 = c1.begin();
	typename Container2::const_iterator iter2 = c2.begin();
	for( ; iter1 != c1.end(); ++iter1, ++iter2 ) if( *iter1 != *iter2 ) return false;
	return true;
}

int main( int argc, char* argv[] ) {

	std::cout << "Testing ecuda::array..." << std::endl;

	// fixed sized array with single value, simple copy device->host
	{
		ecuda::array<int,100> deviceArray( 3 ); // array filled with number 3
		std::vector<int> hostVector;
		deviceArray >> hostVector;
		std::cout << "Single value device array copied to host vector has correct size    " << "\t" << ( hostVector.size() == 100 ? "PASSED" : "FAILED" ) << std::endl;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != 3 ) passed = false;
		std::cout << "Single value device array copied to host vector has correct contents" << "\t" << ( passed ? "PASSED" : "FAILED" ) << std::endl;
	}

	// fixed sized array with many values, host->device, then device->host
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::array<int,100> deviceArray( hostVector.begin(), hostVector.end() );
		hostVector.clear();
		deviceArray >> hostVector;
		std::cout << "Multiple value device array copied to host vector has correct size    " << "\t" << ( hostVector.size() == 100 ? "PASSED" : "FAILED" ) << std::endl;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != i ) passed = false;
		std::cout << "Multiple value device array copied to host vector has correct contents" << "\t" << ( passed ? "PASSED" : "FAILED" ) << std::endl;
	}

	// fixed sized array constructed with initializer list
	{
		#ifdef __CPP11_SUPPORTED__
		ecuda::array<int,5> deviceArray = { { 0, 1, 2, 3, 4 } };
		std::array<int,5> hostArray;
		deviceArray >> hostArray;
		std::cout << "Multiple value device array constructed with initializer list         " << "\t" << ( deviceArray.size() == 5 ? "PASSED" : "FAILED" ) << std::endl;
		bool passed = true;
		for( std::array<int,5>::size_type i = 0; i < 5; ++i ) if( hostArray[i] != i ) passed = false;
		std::cout << "Multiple value device array constructed with initializer list has correct contents" << "\t" << ( passed ? "PASSED" : "FAILED" ) << std::endl;
		#else
		std::cout << "Multiple value device array constructed with initializer list has correct size    " << "\t" << "NO C++11 SUPPORT" << std::endl;
		std::cout << "Multiple value device array constructed with initializer list has correct contents" << "\t" << "NO C++11 SUPPORT" << std::endl;
		#endif
	}

	// two fixed sized arrays of different size, copied from smaller to larger
	{
		ecuda::array<int,5> deviceArray1( 3 ); // array filled with number 3
		ecuda::array<int,10> deviceArray2( deviceArray1 );
		std::vector<int> hostVector2;
		deviceArray2 >> hostVector2;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < deviceArray1.size(); ++i ) if( hostVector2[i] != 3 ) passed = false;
		//for( std::vector<int>::size_type i = deviceArray1.size(); i < deviceArray2.size(); ++i ) if( hostVector2[i] != 0 ) passed = false;
		std::cout << "Single value device arrays of different size copied smaller to larger has correct contents" << "\t" << ( passed ? "PASSED" : "FAILED" ) << std::endl;
	}

	// two fixed sized arrays of different size, copied from larger to smaller
	{
		ecuda::array<int,10> deviceArray1( 3 ); // array filled with number 3
		ecuda::array<int,5> deviceArray2( deviceArray1 );
		std::vector<int> hostVector2;
		deviceArray2 >> hostVector2;
		bool passed = true;
		for( ecuda::array<int,5>::size_type i = 0; i < deviceArray2.size(); ++i ) if( hostVector2[i] != 3 ) passed = false;
		std::cout << "Single value device arrays of different size copied larger to smaller has correct contents" << "\t" << ( passed ? "PASSED" : "FAILED" ) << std::endl;
	}

	//
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::array<int,100> deviceArray1( hostVector.begin(), hostVector.end() );
		ecuda::array<int,100> deviceArray2( deviceArray1 );
		hostVector.clear();
		deviceArray2 >> hostVector;
		std::cout << "Multiple value device array mirrored on device and copied to host vector has correct size    " << "\t" << ( hostVector.size() == 100 ? "PASSED" : "FAILED" ) << std::endl;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != i ) passed = false;
		std::cout << "Multiple value device array mirrored on device and copied to host vector has correct contents" << "\t" << ( passed ? "PASSED" : "FAILED" ) << std::endl;
	}

	//
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::array<int,100> deviceArray1( hostVector.begin(), hostVector.end() );
		ecuda::array<int,100> deviceArray2( hostVector.rbegin(), hostVector.rend() );

		// multiply arrays on GPU
		dim3 dimBlock( 100, 1 ), dimGrid( 1, 1 );
		multiplyVectors<int,100><<<dimGrid,dimBlock>>>( deviceArray1, deviceArray2 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );

		hostVector.clear();
		deviceArray1 >> hostVector;
		std::cout << "Two multiple value device arrays multiplied on device and copied to host vector has correct size    " << "\t" << ( hostVector.size() == 100 ? "PASSED" : "FAILED" ) << std::endl;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != (i*(hostVector.size()-i-1)) ) passed = false;
		std::cout << "Two multiple value device arrays multiplied on device and copied to host vector has correct contents" << "\t" << ( passed ? "PASSED" : "FAILED" ) << std::endl;
	}

	//
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::array<int,100> deviceArray1( hostVector.begin(), hostVector.end() );
		ecuda::array<int,100> deviceArray2( 0 );

		// take moving sum of vectors using only iterators
		dim3 dimBlock( 100, 1 ), dimGrid( 1, 1 );
		testIterators<int,100><<<dimGrid,dimBlock>>>( deviceArray1, deviceArray2 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );

		// generate answer key
		std::vector<int> answerVector( 100, 0 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) {
			for( std::vector<int>::size_type j = i; j < hostVector.size(); ++j ) {
				answerVector[i] += hostVector[j];
			}
		}

		hostVector.clear();
		deviceArray2 >> hostVector;
		std::cout << "Multiple value device array used to generate running sum using iterators and copied to host vector has correct size    " << "\t" << ( hostVector.size() == 100 ? "PASSED" : "FAILED" ) << std::endl;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != answerVector[i] ) passed = false;
		std::cout << "Multiple value device array used to generate running sum using iterators and copied to host vector has correct contents" << "\t" << ( passed ? "PASSED" : "FAILED" ) << std::endl;
	}

	//
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::array<int,100> deviceArray1( hostVector.begin(), hostVector.end() );
		ecuda::array<int,100> deviceArray2( 0 );

		// take moving sum of vectors using only iterators
		dim3 dimBlock( 100, 1 ), dimGrid( 1, 1 );
		testReverseIterators<int,100><<<dimGrid,dimBlock>>>( deviceArray1, deviceArray2 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );

		// generate answer key
		std::vector<int> answerVector( 100, 0 );
		for( std::vector<int>::size_type i = hostVector.size()-1; i >= 0; ++i ) {
			for( std::vector<int>::size_type j = i; j >= 0; ++j ) {
				answerVector[i] += hostVector[j];
			}
		}

		hostVector.clear();
		deviceArray2 >> hostVector;
		std::cout << "Multiple value device array used to generate running sum using reverse iterators and copied to host vector has correct size    " << "\t" << ( hostVector.size() == 100 ? "PASSED" : "FAILED" ) << std::endl;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != answerVector[i] ) passed = false;
		std::cout << "Multiple value device array used to generate running sum using reverse iterators and copied to host vector has correct contents" << "\t" << ( passed ? "PASSED" : "FAILED" ) << std::endl;
	}

	{
		ecuda::array<int,100> deviceArray;
		deviceArray.fill( 3 );
		std::vector<int> hostVector( 100 );
		deviceArray >> hostVector;
		{
			bool passed = true;
			for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != 3 ) passed = false;
			std::cout << "Device array fill operation performed on host  " << "\t" << ( passed ? "PASSED" : "FAILED" ) << std::endl;
		}
		dim3 dimBlock( 1, 1 ), dimGrid( 1, 1 );
		testFill<int,100><<<dimGrid,dimBlock>>>( deviceArray );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		deviceArray >> hostVector;
		{
			bool passed = true;
			for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != 5 ) passed = false;
			std::cout << "Device array fill operation performed on device" << "\t" << ( passed ? "PASSED" : "FAILED" ) << std::endl;
		}
	}

	{
		std::vector<int> hostVector1( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector1.size(); ++i ) hostVector1[i] = i;
		std::vector<int> hostVector2( hostVector1.rbegin(), hostVector1.rend() );
		{
			ecuda::array<int,100> deviceArray1( hostVector1.begin(), hostVector1.end() );
			ecuda::array<int,100> deviceArray2( hostVector2.begin(), hostVector2.end() );
			deviceArray1.swap( deviceArray2 );
			std::vector<int> resultHostVector1, resultHostVector2;
			deviceArray1 >> resultHostVector1;
			deviceArray2 >> resultHostVector2;
			std::cout << "Device array swap operation performed on host  " << "\t" << ( compare_containers(hostVector1,resultHostVector2) and compare_containers(hostVector2,resultHostVector1) ? "PASSED" : "FAILED" ) << std::endl;
		}
		{
			ecuda::array<int,100> deviceArray1( hostVector1.begin(), hostVector1.end() );
			ecuda::array<int,100> deviceArray2( hostVector2.begin(), hostVector2.end() );
			testSwap<int,100><<<1,1>>>( deviceArray1, deviceArray2 );
			CUDA_CHECK_ERRORS();
			CUDA_CALL( cudaDeviceSynchronize() );
			std::vector<int> resultHostVector1, resultHostVector2;
			deviceArray1 >> resultHostVector1;
			deviceArray2 >> resultHostVector2;
			std::cout << "Device array swap operation performed on device" << "\t" << ( compare_containers(hostVector1,resultHostVector2) and compare_containers(hostVector2,resultHostVector1) ? "PASSED" : "FAILED" ) << std::endl;
		}
	}

	{
		std::vector<int> hostVector1( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector1.size(); ++i ) hostVector1[i] = i;
		std::vector<int> hostVector2( hostVector1.begin(), hostVector1.end() );
		ecuda::array<int,100> deviceArray1( hostVector1.begin(), hostVector1.end() );
		ecuda::array<int,100> deviceArray2( hostVector2.begin(), hostVector2.end() );
		ecuda::array<int,1> deviceResultArray( 0 );
		testEquality<int,100><<<1,1>>>( deviceArray1, deviceArray2, deviceResultArray );
		std::vector<int> hostResultArray;
		deviceResultArray >> hostResultArray;
		std::cout << "operator== on device\t" << ( hostResultArray.front() ? "PASSED" : "FAILED" ) << std::endl;
		std::cout << "operator== on host  \t" << ( deviceArray1 == deviceArray2 ? "PASSED" : "FAILED" ) << std::endl;
	}

	{
		std::vector<int> hostVector1( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector1.size(); ++i ) hostVector1[i] = i;
		std::vector<int> hostVector2( hostVector1.rbegin(), hostVector1.rend() );
		ecuda::array<int,100> deviceArray1( hostVector1.begin(), hostVector1.end() );
		ecuda::array<int,100> deviceArray2( hostVector2.begin(), hostVector2.end() );
		ecuda::array<int,1> deviceResultArray( 0 );
		testLessThan<int,100><<<1,1>>>( deviceArray1, deviceArray2, deviceResultArray );
		std::vector<int> hostResultArray;
		deviceResultArray >> hostResultArray;
		std::cout << "operator< on device\t" << ( hostResultArray.front() ? "PASSED" : "FAILED" ) << std::endl;
		std::cout << "operator< on host  \t" << ( deviceArray1 < deviceArray2 ? "PASSED" : "FAILED" ) << std::endl;
	}

	{
		std::vector<int> hostVector1( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector1.size(); ++i ) hostVector1[i] = i;
		std::vector<int> hostVector2( hostVector1.rbegin(), hostVector1.rend() );
		ecuda::array<int,100> deviceArray1( hostVector1.begin(), hostVector1.end() );
		ecuda::array<int,100> deviceArray2( hostVector2.begin(), hostVector2.end() );
		ecuda::array<int,1> deviceResultArray( 0 );
		testGreaterThan<int,100><<<1,1>>>( deviceArray2, deviceArray1, deviceResultArray );
		std::vector<int> hostResultArray;
		deviceResultArray >> hostResultArray;
		std::cout << "operator> on device\t" << ( hostResultArray.front() ? "PASSED" : "FAILED" ) << std::endl;
		std::cout << "operator> on host  \t" << ( deviceArray2 > deviceArray1 ? "PASSED" : "FAILED" ) << std::endl;
	}

	return EXIT_SUCCESS;

}
