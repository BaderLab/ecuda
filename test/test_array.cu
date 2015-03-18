#include <iostream>
#include <cstdio>
#include "../include/ecuda/array.hpp"

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
		std::cout << "Multiple value device array used to generate running sum and copied to host vector has correct size    " << "\t" << ( hostVector.size() == 100 ? "PASSED" : "FAILED" ) << std::endl;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != answerVector[i] ) passed = false;
		std::cout << "Multiple value device array used to generate running sum and copied to host vector has correct contents" << "\t" << ( passed ? "PASSED" : "FAILED" ) << std::endl;
	}

	return EXIT_SUCCESS;

}
