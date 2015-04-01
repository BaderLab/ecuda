#include <algorithm>
#include <iostream>
#include <string>
#include <cstdio>
#include "../include/ecuda/array.hpp"
#include "../include/ecuda/vector.hpp"

template<typename T> __global__
void kernel_checkVectorProperties(
	const ecuda::vector<T> constVector,
	ecuda::vector<T> vector,
	ecuda::vector<T> empties,
	ecuda::vector<typename ecuda::vector<T>::size_type> sizes,
	ecuda::vector<typename ecuda::vector<T>::pointer> pointers,
	ecuda::vector<typename ecuda::vector<T>::const_pointer> constPointers
)
{
	const int threadNumber = threadIdx.x;
	empties[threadNumber] = constVector.empty() ? 1 : 0;
	sizes[threadNumber] = constVector.size();
	pointers[threadNumber] = vector.data();
	constPointers[threadNumber] = constVector.data();
}

template<typename T> __global__
void kernel_checkVectorAccessors(
	const ecuda::vector<T> srcVector,
	ecuda::vector<T> srcVectorNonConst,
	ecuda::vector<T> destVector,
	ecuda::vector<T> srcFronts,
	ecuda::vector<T> srcBacks,
	ecuda::vector<T> srcFrontsNonConst,
	ecuda::vector<T> srcBacksNonConst
)
{
	const int threadNumber = threadIdx.x;
	destVector[threadNumber] = srcVector[threadNumber];
	srcFronts[threadNumber] = srcVector.front();
	srcBacks[threadNumber] = srcVector.back();
	srcFrontsNonConst[threadNumber] = srcVectorNonConst.front();
	srcBacksNonConst[threadNumber] = srcVectorNonConst.back();
}

template<typename T> __global__
void kernel_checkDeviceIterators(
	const ecuda::vector<T> srcVector,
	ecuda::vector<T> destVector
)
{
	typename ecuda::vector<T>::const_iterator srcIterator = srcVector.begin();
	typename ecuda::vector<T>::iterator destIterator = destVector.begin();
	for( ; srcIterator != srcVector.end() and destIterator != destVector.end(); ++srcIterator, ++destIterator )
		*destIterator = *srcIterator;
}

template<typename T> __global__
void kernel_checkHostIterators(
	typename ecuda::vector<T>::const_iterator srcBegin,
	typename ecuda::vector<T>::const_iterator srcEnd,
	typename ecuda::vector<T>::iterator destBegin,
	typename ecuda::vector<T>::iterator destEnd
)
{
	for( ; srcBegin != srcEnd and destBegin != destEnd; ++srcBegin, ++destBegin )
		*destBegin = *srcBegin;
}

template<typename T> __global__
void kernel_checkDeviceReverseIterators(
	const ecuda::vector<T> srcVector,
	ecuda::vector<T> destVector
)
{
	typename ecuda::vector<T>::const_reverse_iterator srcIterator = srcVector.rbegin();
	typename ecuda::vector<T>::reverse_iterator destIterator = destVector.rbegin();
	for( ; srcIterator != srcVector.rend() and destIterator != destVector.rend(); ++srcIterator, ++destIterator )
		*destIterator = *srcIterator;
}

template<typename T> __global__
void kernel_checkHostReverseIterators(
	typename ecuda::vector<T>::const_reverse_iterator srcBegin,
	typename ecuda::vector<T>::const_reverse_iterator srcEnd,
	typename ecuda::vector<T>::reverse_iterator destBegin,
	typename ecuda::vector<T>::reverse_iterator destEnd
)
{
	for( ; srcBegin != srcEnd and destBegin != destEnd; ++srcBegin, ++destBegin ) *destBegin = *srcBegin;
}

template<typename T> __global__
void kernel_testSwapAndClear(
	ecuda::vector<T> vector1,
	ecuda::vector<T> vector2,
	ecuda::vector<T> dummyVector,
	ecuda::array<int,2> array
)
{
	dummyVector.clear();
	if( dummyVector.empty() ) array[0] = 1;
	T oldStart1 = vector1.front();
	T oldStart2 = vector2.front();
	vector1.swap( vector2 );
	if( vector2.front() == oldStart1 and vector1.front() == oldStart2 ) array[1] = 1;
}

template<typename T> __global__
void kernel_testComparisonOperators(
	const ecuda::vector<T> vector1,
	const ecuda::vector<T> vector2,
	ecuda::vector<int> results
)
{
	results[0] = vector1 == vector2 ? 0 : 1;
	results[1] = vector1 != vector2 ? 1 : 0;
	results[2] = vector1 < vector2 ? 1 : 0;
	results[3] = vector1 > vector2 ? 0 : 1;
	results[4] = vector1 <= vector2 ? 1 : 0;
	results[5] = vector1 >= vector2 ? 0 : 1;
	results[6] = vector2 < vector1 ? 0 : 1;
	results[7] = vector2 > vector1 ? 1 : 0;
	results[8] = vector2 <= vector1 ? 0 : 1;
	results[9] = vector2 >= vector1 ? 1 : 0;
}

int main( int argc, char* argv[] ) {

	std::cout << "Testing ecuda::vector..." << std::endl;

	std::vector<int> testResults;

	//
	// Test 1: default constructor, copy to host, and general info
	//
	std::cerr << "Test 1" << std::endl;
	{
		bool passed = true;
		{
			ecuda::vector<int> deviceVector;
			if( deviceVector.size() ) passed = false;
			if( !deviceVector.empty() ) passed = false;
		}
		{
			const ecuda::vector<int> deviceVector( 100 );
			if( deviceVector.size() != 100 ) passed = false;
			if( deviceVector.empty() ) passed = false;
			if( !deviceVector.data() ) passed = false;
			std::vector<int> hostVector;
			deviceVector >> hostVector;
			if( hostVector.size() != 100 ) passed = false;
			for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] ) passed = false;
		}
		ecuda::vector<int> deviceVector( 100, 3 );
		if( deviceVector.size() != 100 ) passed = false;
		if( deviceVector.empty() ) passed = false;
		if( !deviceVector.data() ) passed = false;
		std::vector<int> hostVector;
		deviceVector >> hostVector;
		if( hostVector.size() != 100 ) passed = false;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != 3 ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 2: information is correct on device
	//
	std::cerr << "Test 2" << std::endl;
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::vector<int> deviceVector( hostVector.begin(), hostVector.end() );
		ecuda::vector<int> deviceEmpties( 100, -1 );
		ecuda::vector<ecuda::vector<int>::size_type> deviceSizes( 100 );
		ecuda::vector<ecuda::vector<int>::pointer> devicePointers( 100 );
		ecuda::vector<ecuda::vector<int>::const_pointer> deviceConstPointers( 100 );
		kernel_checkVectorProperties<<<1,100>>>( deviceVector, deviceVector, deviceEmpties, deviceSizes, devicePointers, deviceConstPointers );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::vector<int> hostEmpties( 100, -1 );
		std::vector<ecuda::vector<int>::size_type> hostSizes( 100 );
		std::vector<ecuda::vector<int>::pointer> hostPointers( 100 );
		std::vector<ecuda::vector<int>::const_pointer> hostConstPointers( 100 );
		deviceEmpties >> hostEmpties;
		deviceSizes >> hostSizes;
		devicePointers >> hostPointers;
		deviceConstPointers >> hostConstPointers;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostEmpties.size(); ++i ) if( hostEmpties[i] != 0 ) passed = false;
		for( std::vector<ecuda::vector<int>::size_type>::size_type i = 0; i < hostSizes.size(); ++i ) if( hostSizes[i] != 100 ) passed = false;
		for( std::vector<ecuda::vector<int>::pointer>::size_type i = 0; i < hostPointers.size(); ++i ) if( hostPointers[i] != deviceVector.data() ) passed = false;
		for( std::vector<ecuda::vector<int>::const_pointer>::size_type i = 0; i < hostConstPointers.size(); ++i ) if( hostConstPointers[i] != deviceVector.data() ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 3: construction using a C++11 intializer list
	//
	#ifdef __CPP11_SUPPORTED__
	std::cerr << "Test 3" << std::endl;
	{
		ecuda::vector<int> deviceVector( { 0,1,2,3,4,5,6,7,8,9 } );
		std::vector<int> hostVector;
		deviceVector >> hostVector;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != i ) passed = false;
		std::reverse( hostVector.begin(), hostVector.end() );
		deviceVector << hostVector;
		std::fill( hostVector.begin(), hostVector.end(), 0 );
		deviceVector >> hostVector;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[hostVector.size()-1-i] != i ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}
	#else
	std::cerr << "Test 3 (skipped)" << std::endl;
	testResults.push_back( -1 );
	#endif

	//
	// Test 4: index accessors, front(), and back()
	//
	std::cerr << "Test 4" << std::endl;
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::vector<int> srcDeviceVector( hostVector.begin(), hostVector.end() );
		ecuda::vector<int> destDeviceVector( 100 );
		ecuda::vector<int> deviceFronts( 100, -1 ), deviceBacks( 100, -1 ), deviceFrontsNonConst( 100, -1 ), deviceBacksNonConst( 100, -1 );
		kernel_checkVectorAccessors<<<1,100>>>( srcDeviceVector, srcDeviceVector, destDeviceVector, deviceFronts, deviceBacks, deviceFrontsNonConst, deviceBacksNonConst );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );

		bool passed = true;
		std::vector<int> hostResults;

		destDeviceVector >> hostResults;
		for( std::vector<int>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != i ) passed = false;
		hostResults.clear();

		deviceFronts >> hostResults;
		for( std::vector<int>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != 0 ) passed = false;
		hostResults.clear();

		deviceBacks >> hostResults;
		for( std::vector<int>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != 99 ) passed = false;
		hostResults.clear();

		deviceFrontsNonConst >> hostResults;
		for( std::vector<int>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != 0 ) passed = false;
		hostResults.clear();

		deviceBacksNonConst >> hostResults;
		for( std::vector<int>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != 99 ) passed = false;
		//hostResults.clear();

		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 5: device iterators
	//
	std::cerr << "Test 5" << std::endl;
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::vector<int> srcDeviceVector( hostVector.begin(), hostVector.end() );
		ecuda::vector<int> destDeviceVector( 100 );
		kernel_checkDeviceIterators<<<1,1>>>( srcDeviceVector, destDeviceVector );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::fill( hostVector.begin(), hostVector.end(), -1 );
		destDeviceVector >> hostVector;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != i ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 6: host iterators
	//
	std::cerr << "Test 6" << std::endl;
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::vector<int> srcDeviceVector( hostVector.begin(), hostVector.end() );
		ecuda::vector<int> destDeviceVector( 100 );
		kernel_checkHostIterators<int><<<1,1>>>( srcDeviceVector.begin(), srcDeviceVector.end(), destDeviceVector.begin(), destDeviceVector.end() );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::fill( hostVector.begin(), hostVector.end(), -1 );
		destDeviceVector >> hostVector;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != i ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 7: device reverse iterators
	//
	std::cerr << "Test 7" << std::endl;
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::vector<int> srcDeviceVector( hostVector.begin(), hostVector.end() );
		ecuda::vector<int> destDeviceVector( 100 );
		kernel_checkDeviceReverseIterators<<<1,1>>>( srcDeviceVector, destDeviceVector );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::fill( hostVector.begin(), hostVector.end(), -1 );
		destDeviceVector >> hostVector;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != i ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 8: host reverse iterators
	//
	std::cerr << "Test 8" << std::endl;
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::vector<int> srcDeviceVector( hostVector.begin(), hostVector.end() );
		ecuda::vector<int> destDeviceVector( 100 );
		kernel_checkHostReverseIterators<int><<<1,1>>>( srcDeviceVector.rbegin(), srcDeviceVector.rend(), destDeviceVector.rbegin(), destDeviceVector.rend() );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::fill( hostVector.begin(), hostVector.end(), -1 );
		destDeviceVector >> hostVector;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != i ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 9: host swap and clear
	//
	std::cerr << "Test 9" << std::endl;
	{
		ecuda::vector<int> deviceVector1( 100, 3 );
		ecuda::vector<int> deviceVector2( 100, 5 );
		deviceVector1.swap( deviceVector2 );
		std::vector<int> hostVector1; deviceVector1 >> hostVector1;
		std::vector<int> hostVector2; deviceVector2 >> hostVector2;
		bool passed = true;
		if( hostVector1.size() != 100 ) passed = false;
		if( hostVector2.size() != 100 ) passed = false;
		for( std::vector<int>::size_type i = 0; i < hostVector1.size(); ++i ) if( hostVector1[i] != 5 ) passed = false;
		for( std::vector<int>::size_type i = 0; i < hostVector2.size(); ++i ) if( hostVector2[i] != 3 ) passed = false;
		deviceVector1.clear();
		deviceVector2.clear();
		if( !deviceVector1.empty() ) passed = false;
		if( !deviceVector2.empty() ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test A: device swap and clear
	//
	std::cerr << "Test A" << std::endl;
	{
		ecuda::vector<int> deviceVector1( 100, 3 );
		ecuda::vector<int> deviceVector2( 100, 5 );
		ecuda::vector<int> deviceDummyVector( 100 );
		ecuda::array<int,2> deviceArray( 0 );
		kernel_testSwapAndClear<<<1,1>>>( deviceVector1, deviceVector2, deviceDummyVector, deviceArray );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		//std::vector<int> hostVector1; deviceVector1 >> hostVector1;
		//std::vector<int> hostVector2; deviceVector2 >> hostVector2;
		std::vector<int> hostArray;
		deviceArray >> hostArray;
		//bool passed = true;
		const bool passed = hostArray.size() == 2 and hostArray.front() == 1 and hostArray.back() == 1;
		//for( std::vector<int>::size_type i = 0; i < hostVector1.size(); ++i ) if( hostVector1[i] != 5 ) passed = false;
		//for( std::vector<int>::size_type i = 0; i < hostVector2.size(); ++i ) if( hostVector2[i] != 3 ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test B: host comparison operators
	//
	std::cerr << "Test B" << std::endl;
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::vector<int> deviceVector1( hostVector.begin(), hostVector.end() );
		ecuda::vector<int> deviceVector2( hostVector.begin(), hostVector.end() );

		bool passed = true;
		if( !deviceVector1.operator==(deviceVector2) ) passed = false;
		if( deviceVector1.operator!=(deviceVector2) ) passed = false;
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i+10;
		deviceVector2 << hostVector;
		if( !deviceVector1.operator< (deviceVector2) ) passed = false;
		if(  deviceVector1.operator> (deviceVector2) ) passed = false;
		if( !deviceVector1.operator<=(deviceVector2) ) passed = false;
		if(  deviceVector1.operator>=(deviceVector2) ) passed = false;
		if(  deviceVector2.operator< (deviceVector1) ) passed = false;
		if( !deviceVector2.operator> (deviceVector1) ) passed = false;
		if(  deviceVector2.operator<=(deviceVector1) ) passed = false;
		if( !deviceVector2.operator>=(deviceVector1) ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test C: device comparison operators
	//
	std::cerr << "Test C" << std::endl;
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::vector<int> deviceVector1( hostVector.begin(), hostVector.end() );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i+10;
		ecuda::vector<int> deviceVector2( hostVector.begin(), hostVector.end() );
		ecuda::vector<int> deviceResults( 10, -1 );

		kernel_testComparisonOperators<<<1,1>>>( deviceVector1, deviceVector2, deviceResults );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );

		std::vector<int> hostResults;
		deviceResults >> hostResults;
		bool passed = true;
		for( std::vector<bool>::size_type i = 0; i < hostResults.size(); ++i ) if( hostResults[i] != 1 ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	const std::string outputText =
"\
.........................................................+-+-+-+-+-+-+-+-+-+-+-+-+-+\n\
.........................................................|1|2|3|4|5|6|7|8|9|A|B|C|D|\n\
.........................................................+-+-+-+-+-+-+-+-+-+-+-+-+-+\n\
vector( const value_type& )                     H        |X|X| |X| | | | |X|X| |X| |eol\n\
template vector( InputIterator, InputIterator ) H        | |X| |X|X|X|X|X| | |X|X| |eol\n\
vector( std::initializer_list<T> il )           H  C++11 | | |X| | | | | | | | | | |eol\n\
template vector( const array<T,N2>& )           H        | | | | | | | | | | |X| | |eol\n\
vector( vector<T,N>&& )                         HD C++11 | | | | | | | | | | | | |X|eol\n\
~vector()                                       HD       |X|X|X|X|X|X|X|X|X|X| | | |eol\n\
operator[]( size_type )                         D        | | | |X| | | | | | | |X| |eol\n\
operator[]( size_type ) const                   D        | | | |X| | | | | | | | | |eol\n\
front()                                         D        | | | |X| | | | | | | | | |eol\n\
back()                                          D        | | | |X| | | | | | | | | |eol\n\
front() const                                   D        | | | |X| | | | | | | | | |eol\n\
back() const                                    D        | | | |X| | | | | | | | | |eol\n\
empty() const                                   HD       |H|D| | | | | | | | | | | |eol\n\
size() const                                    HD       |H|D| | | | | | | | | | | |eol\n\
max_size() const                                H        |X| | | | | | | | | | | | |eol\n\
data()                                          HD       |H| | | | | | | | | | | | |eol\n\
data() const                                    HD       | |D| | | | | | | | | | | |eol\n\
begin()                                         HD       | | | | |D|H| | | | | | | |eol\n\
end()                                           HD       | | | | |D|H| | | | | | | |eol\n\
begin() const                                   HD       | | | | |D|H| | | | | | | |eol\n\
end() const                                     HD       | | | | |D|H| | | | | | | |eol\n\
rbegin()                                        HD       | | | | | | |D|H| | | | | |eol\n\
rend()                                          HD       | | | | | | |D|H| | | | | |eol\n\
rbegin() const                                  HD       | | | | | | |D|H| | | | | |eol\n\
rend() const                                    HD       | | | | | | |D|H| | | | | |eol\n\
fill( const value_type& )                       HD       | | | | | | | | |H|D| | | |eol\n\
swap( vector<T,N>& )                            HD       | | | | | | | | |H|D| | | |eol\n\
operator==( const vector<T,N>& ) const          HD       | | | | | | | | | | |H|D| |eol\n\
operator!=( const vector<T,N>& ) const          HD       | | | | | | | | | | |H|D| |eol\n\
operator<( const vector<T,N>& ) const           HD       | | | | | | | | | | |H|D| |eol\n\
operator>( const vector<T,N>& ) const           HD       | | | | | | | | | | |H|D| |eol\n\
operator<=( const vector<T,N>& ) const          HD       | | | | | | | | | | |H|D| |eol\n\
operator>=( const vector<T,N>& ) const          HD       | | | | | | | | | | |H|D| |eol\n\
operator>>( std::vector<value_type>& ) const    H        |X|X| |X|X|X|X|X|X|X| |X| |eol\n\
operator<<( std::vector<value_type>& )          H        | |X| | | | | | | | |X|X| |eol\n\
operator>>( std::array<value_type,N>& ) const   H  C++11 | | |X| | | | | | | | | | |eol\n\
operator<<( std::array<value_type,N>& )         H  C++11 | | |X| | | | | | | | | | |eol\n\
vector<T,N>& operator=( const vector<T,N>& )    D        | |X| | |X|X|X|X|X|X| | | |eol\n\
.........................................................+-+-+-+-+-+-+-+-+-+-+-+-+-+\n\
.........................................................|";

	std::cout << outputText;
	for( std::vector<bool>::size_type i = 0; i < testResults.size(); ++i ) std::cout << ( testResults[i] == 1 ? "P" : ( testResults[i] == -1 ? "?" : "F" ) ) << "|";
	std::cout << std::endl;

	return EXIT_SUCCESS;

}
