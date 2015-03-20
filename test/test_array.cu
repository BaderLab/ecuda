#include <algorithm>
#include <iostream>
#include <string>
#include <cstdio>
#include "../include/ecuda/array.hpp"

#ifdef __CPP11_SUPPORTED__
#include <array>
#endif

template<typename T,std::size_t U> __global__
void kernel_checkArrayProperties(
	const ecuda::array<T,U> array,
	ecuda::array<int,U> empties,
	ecuda::array<typename ecuda::array<T,U>::size_type,U> sizes,
	ecuda::array<typename ecuda::array<T,U>::const_pointer,U> pointers
)
{
	const int threadNumber = threadIdx.x;
	empties[threadNumber] = array.empty() ? 1 : 0;
	sizes[threadNumber] = array.size();
	pointers[threadNumber] = array.data();
}

template<typename T,std::size_t U> __global__
void kernel_checkArrayAccessors(
	const ecuda::array<T,U> srcArray,
	ecuda::array<T,U> srcArrayNonConst,
	ecuda::array<T,U> destArray,
	ecuda::array<T,U> srcFronts,
	ecuda::array<T,U> srcBacks,
	ecuda::array<T,U> srcFrontsNonConst,
	ecuda::array<T,U> srcBacksNonConst
)
{
	const int threadNumber = threadIdx.x;
	destArray[threadNumber] = srcArray[threadNumber];
	srcFronts[threadNumber] = srcArray.front();
	srcBacks[threadNumber] = srcArray.back();
	srcFrontsNonConst[threadNumber] = srcArrayNonConst.front();
	srcBacksNonConst[threadNumber] = srcArrayNonConst.back();
}

template<typename T,std::size_t U> __global__
void kernel_checkDeviceIterators(
	const ecuda::array<T,U> srcArray,
	ecuda::array<T,U> destArray
)
{
	typename ecuda::array<T,U>::const_iterator srcIterator = srcArray.begin();
	typename ecuda::array<T,U>::iterator destIterator = destArray.begin();
	for( ; srcIterator != srcArray.end() and destIterator != destArray.end(); ++srcIterator, ++destIterator )
		*destIterator = *srcIterator;
}

template<typename T,std::size_t U> __global__
void kernel_checkHostIterators(
	typename ecuda::array<T,U>::const_iterator srcBegin,
	typename ecuda::array<T,U>::const_iterator srcEnd,
	typename ecuda::array<T,U>::iterator destBegin,
	typename ecuda::array<T,U>::iterator destEnd
)
{
	for( ; srcBegin != srcEnd and destBegin != destEnd; ++srcBegin, ++destBegin ) *destBegin = *srcBegin;
}

template<typename T,std::size_t U> __global__
void kernel_checkDeviceReverseIterators(
	const ecuda::array<T,U> srcArray,
	ecuda::array<T,U> destArray
)
{
	typename ecuda::array<T,U>::const_reverse_iterator srcIterator = srcArray.rbegin();
	typename ecuda::array<T,U>::reverse_iterator destIterator = destArray.rbegin();
	for( ; srcIterator != srcArray.rend() and destIterator != destArray.rend(); ++srcIterator, ++destIterator )
		*destIterator = *srcIterator;
}

template<typename T,std::size_t U> __global__
void kernel_checkHostReverseIterators(
	typename ecuda::array<T,U>::const_reverse_iterator srcBegin,
	typename ecuda::array<T,U>::const_reverse_iterator srcEnd,
	typename ecuda::array<T,U>::reverse_iterator destBegin,
	typename ecuda::array<T,U>::reverse_iterator destEnd
)
{
	for( ; srcBegin != srcEnd and destBegin != destEnd; ++srcBegin, ++destBegin ) *destBegin = *srcBegin;
}

template<typename T,std::size_t U> __global__
void kernel_testFillAndSwap(
	ecuda::array<T,U> array1,
	ecuda::array<T,U> array2
)
{
	array1.fill( 3 );
	array1.swap( array2 );
}

template<typename T,std::size_t U> __global__
void kernel_testComparisonOperators(
	const ecuda::array<T,U> array1,
	const ecuda::array<T,U> array2,
	ecuda::array<int,10> results
)
{
	results[0] = array1 == array2 ? 0 : 1;
	results[1] = array1 != array2 ? 1 : 0;
	results[2] = array1 < array2 ? 1 : 0;
	results[3] = array1 > array2 ? 0 : 1;
	results[4] = array1 <= array2 ? 1 : 0;
	results[5] = array1 >= array2 ? 0 : 1;
	results[6] = array2 < array1 ? 0 : 1;
	results[7] = array2 > array1 ? 1 : 0;
	results[8] = array2 <= array1 ? 0 : 1;
	results[9] = array2 >= array1 ? 1 : 0;
}

int main( int argc, char* argv[] ) {

	std::cout << "Testing ecuda::array..." << std::endl;

	std::vector<int> testResults;

	//
	// Test 1: default constructor, copy to host, and general info
	//
	std::cerr << "Test 1" << std::endl;
	{
		ecuda::array<int,100> deviceArray( 3 ); // array filled with number 3
		std::vector<int> hostVector;
		deviceArray >> hostVector;
		bool passed = true;
		if( deviceArray.size() == 100 and hostVector.size() == deviceArray.size() ) {
			for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) if( hostVector[i] != 3 ) passed = false;
		}
		// make sure device array memory is somewhere
		if( !deviceArray.data() ) passed = false;
		if( deviceArray.max_size() < deviceArray.size() ) passed = false;
		if( deviceArray.empty() ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 2: information is correct on device
	//
	std::cerr << "Test 2" << std::endl;
	{
		std::vector<int> hostVector( 100 );
		for( std::vector<int>::size_type i = 0; i < hostVector.size(); ++i ) hostVector[i] = i;
		ecuda::array<int,100> deviceArray( hostVector.begin(), hostVector.end() );
		ecuda::array<int,100> deviceEmpties( -1 );
		ecuda::array<ecuda::array<int,100>::size_type,100> deviceSizes;
		ecuda::array<ecuda::array<int,100>::const_pointer,100> devicePointers;
		kernel_checkArrayProperties<<<1,100>>>( deviceArray, deviceEmpties, deviceSizes, devicePointers );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::vector<int> hostEmpties( 100, -1 );
		std::vector<ecuda::array<int,100>::size_type> hostSizes( 100 );
		std::vector<ecuda::array<int,100>::const_pointer> hostPointers( 100 );
		deviceEmpties >> hostEmpties;
		deviceSizes >> hostSizes;
		devicePointers >> hostPointers;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostEmpties.size(); ++i ) if( hostEmpties[i] != 0 ) passed = false;
		for( std::vector<ecuda::array<int,100>::size_type>::size_type i = 0; i < hostSizes.size(); ++i ) if( hostSizes[i] != 100 ) passed = false;
		for( std::vector<ecuda::array<int,100>::const_pointer>::size_type i = 0; i < hostPointers.size(); ++i ) if( hostPointers[i] != deviceArray.data() ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 3: construction using a C++11 intializer list
	//
	#ifdef __CPP11_SUPPORTED__
	std::cerr << "Test 3" << std::endl;
	{
		ecuda::array<int,10> deviceArray( { 0,1,2,3,4,5,6,7,8,9 } );
		std::array<int,10> hostArray;
		deviceArray >> hostArray;
		bool passed = true;
		for( std::array<int,10>::size_type i = 0; i < hostArray.size(); ++i ) if( hostArray[i] != i ) passed = false;
		std::reverse( hostArray.begin(), hostArray.end() );
		deviceArray << hostArray;
		std::fill( hostArray.begin(), hostArray.end(), 0 );
		deviceArray >> hostArray;
		for( std::array<int,10>::size_type i = 0; i < hostArray.size(); ++i ) if( hostArray[hostArray.size()-1-i] != i ) passed = false;
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
		std::vector<int> hostArray( 100 );
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) hostArray[i] = i;
		ecuda::array<int,100> srcDeviceArray( hostArray.begin(), hostArray.end() );
		ecuda::array<int,100> destDeviceArray;
		ecuda::array<int,100> deviceFronts( -1 ), deviceBacks( -1 ), deviceFrontsNonConst( -1 ), deviceBacksNonConst( -1 );
		kernel_checkArrayAccessors<<<1,100>>>( srcDeviceArray, srcDeviceArray, destDeviceArray, deviceFronts, deviceBacks, deviceFrontsNonConst, deviceBacksNonConst );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );

		bool passed = true;
		std::vector<int> hostResults;

		destDeviceArray >> hostResults;
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
		std::vector<int> hostArray( 100 );
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) hostArray[i] = i;
		ecuda::array<int,100> srcDeviceArray( hostArray.begin(), hostArray.end() );
		ecuda::array<int,100> destDeviceArray;
		kernel_checkDeviceIterators<<<1,1>>>( srcDeviceArray, destDeviceArray );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::fill( hostArray.begin(), hostArray.end(), -1 );
		destDeviceArray >> hostArray;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) if( hostArray[i] != i ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 6: host iterators
	//
	std::cerr << "Test 6" << std::endl;
	{
		std::vector<int> hostArray( 100 );
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) hostArray[i] = i;
		ecuda::array<int,100> srcDeviceArray( hostArray.begin(), hostArray.end() );
		ecuda::array<int,100> destDeviceArray;
		kernel_checkHostIterators<int,100><<<1,1>>>( srcDeviceArray.begin(), srcDeviceArray.end(), destDeviceArray.begin(), destDeviceArray.end() );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::fill( hostArray.begin(), hostArray.end(), -1 );
		destDeviceArray >> hostArray;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) if( hostArray[i] != i ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 7: device reverse iterators
	//
	std::cerr << "Test 7" << std::endl;
	{
		std::vector<int> hostArray( 100 );
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) hostArray[i] = i;
		ecuda::array<int,100> srcDeviceArray( hostArray.begin(), hostArray.end() );
		ecuda::array<int,100> destDeviceArray;
		kernel_checkDeviceReverseIterators<<<1,1>>>( srcDeviceArray, destDeviceArray );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::fill( hostArray.begin(), hostArray.end(), -1 );
		destDeviceArray >> hostArray;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) if( hostArray[i] != i ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 8: host reverse iterators
	//
	std::cerr << "Test 8" << std::endl;
	{
		std::vector<int> hostArray( 100 );
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) hostArray[i] = i;
		ecuda::array<int,100> srcDeviceArray( hostArray.begin(), hostArray.end() );
		ecuda::array<int,100> destDeviceArray;
		kernel_checkHostReverseIterators<int,100><<<1,1>>>( srcDeviceArray.rbegin(), srcDeviceArray.rend(), destDeviceArray.rbegin(), destDeviceArray.rend() );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::fill( hostArray.begin(), hostArray.end(), -1 );
		destDeviceArray >> hostArray;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) if( hostArray[i] != i ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test 9: host fill and swap
	//
	std::cerr << "Test 9" << std::endl;
	{
		ecuda::array<int,100> deviceArray1;
		ecuda::array<int,100> deviceArray2;
		deviceArray1.fill( 3 );
		deviceArray1.swap( deviceArray2 );
		std::vector<int> hostArray;
		deviceArray2 >> hostArray;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) if( hostArray[i] != 3 ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test A: device fill and swap
	//
	std::cerr << "Test A" << std::endl;
	{
		ecuda::array<int,100> deviceArray1;
		ecuda::array<int,100> deviceArray2;
		kernel_testFillAndSwap<<<1,1>>>( deviceArray1, deviceArray2 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::vector<int> hostArray;
		deviceArray2 >> hostArray;
		bool passed = true;
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) if( hostArray[i] != 3 ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test B: host comparison operators
	//
	std::cerr << "Test B" << std::endl;
	{
		std::vector<int> hostArray( 100 );
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) hostArray[i] = i;
		ecuda::array<int,100> deviceArray1( hostArray.begin(), hostArray.end() );
		ecuda::array<int,100> deviceArray2( deviceArray1 );
		bool passed = true;
		if( !deviceArray1.operator==(deviceArray2) ) passed = false;
		if( deviceArray1.operator!=(deviceArray2) ) passed = false;
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) hostArray[i] = i+10;
		deviceArray2 << hostArray;
		if( !deviceArray1.operator< (deviceArray2) ) passed = false;
		if(  deviceArray1.operator> (deviceArray2) ) passed = false;
		if( !deviceArray1.operator<=(deviceArray2) ) passed = false;
		if(  deviceArray1.operator>=(deviceArray2) ) passed = false;
		if(  deviceArray2.operator< (deviceArray1) ) passed = false;
		if( !deviceArray2.operator> (deviceArray1) ) passed = false;
		if( !deviceArray2.operator<=(deviceArray1) ) passed = false;
		if(  deviceArray2.operator>=(deviceArray1) ) passed = false;
		testResults.push_back( passed ? 1 : 0 );
	}

	//
	// Test C: device comparison operators
	//
	std::cerr << "Test C" << std::endl;
	{
		std::vector<int> hostArray( 100 );
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) hostArray[i] = i;
		ecuda::array<int,100> deviceArray1( hostArray.begin(), hostArray.end() );
		for( std::vector<int>::size_type i = 0; i < hostArray.size(); ++i ) hostArray[i] = i+10;
		ecuda::array<int,100> deviceArray2( hostArray.begin(), hostArray.end() );
		ecuda::array<int,10> deviceResults( -1 );

		kernel_testComparisonOperators<<<1,1>>>( deviceArray1, deviceArray2, deviceResults );
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
........................................................+-+-+-+-+-+-+-+-+-+-+-+-+-+\n\
........................................................|1|2|3|4|5|6|7|8|9|A|B|C|D|\n\
........................................................+-+-+-+-+-+-+-+-+-+-+-+-+-+\n\
array( const value_type& )                     H        |X|X| |X| | | | |X|X| |X| |eol\n\
template array( InputIterator, InputIterator ) H        | |X| |X|X|X|X|X| | |X|X| |eol\n\
array( std::initializer_list<T> il )           H  C++11 | | |X| | | | | | | | | | |eol\n\
template array( const array<T,N2>& )           H        | | | | | | | | | | |X| | |eol\n\
array( array<T,N>&& )                          HD C++11 | | | | | | | | | | | | |X|eol\n\
~array()                                       HD       |X|X|X|X|X|X|X|X|X|X| | | |eol\n\
operator[]( size_type )                        D        | | | |X| | | | | | | |X| |eol\n\
operator[]( size_type ) const                  D        | | | |X| | | | | | | | | |eol\n\
front()                                        D        | | | |X| | | | | | | | | |eol\n\
back()                                         D        | | | |X| | | | | | | | | |eol\n\
front() const                                  D        | | | |X| | | | | | | | | |eol\n\
back() const                                   D        | | | |X| | | | | | | | | |eol\n\
empty() const                                  HD       |H|D| | | | | | | | | | | |eol\n\
size() const                                   HD       |H|D| | | | | | | | | | | |eol\n\
max_size() const                               H        |X| | | | | | | | | | | | |eol\n\
data()                                         HD       |H| | | | | | | | | | | | |eol\n\
data() const                                   HD       | |D| | | | | | | | | | | |eol\n\
begin()                                        HD       | | | | |D|H| | | | | | | |eol\n\
end()                                          HD       | | | | |D|H| | | | | | | |eol\n\
begin() const                                  HD       | | | | |D|H| | | | | | | |eol\n\
end() const                                    HD       | | | | |D|H| | | | | | | |eol\n\
rbegin()                                       HD       | | | | | | |D|H| | | | | |eol\n\
rend()                                         HD       | | | | | | |D|H| | | | | |eol\n\
rbegin() const                                 HD       | | | | | | |D|H| | | | | |eol\n\
rend() const                                   HD       | | | | | | |D|H| | | | | |eol\n\
fill( const value_type& )                      HD       | | | | | | | | |H|D| | | |eol\n\
swap( array<T,N>& )                            HD       | | | | | | | | |H|D| | | |eol\n\
operator==( const array<T,N>& ) const          HD       | | | | | | | | | | |H|D| |eol\n\
operator!=( const array<T,N>& ) const          HD       | | | | | | | | | | |H|D| |eol\n\
operator<( const array<T,N>& ) const           HD       | | | | | | | | | | |H|D| |eol\n\
operator>( const array<T,N>& ) const           HD       | | | | | | | | | | |H|D| |eol\n\
operator<=( const array<T,N>& ) const          HD       | | | | | | | | | | |H|D| |eol\n\
operator>=( const array<T,N>& ) const          HD       | | | | | | | | | | |H|D| |eol\n\
operator>>( std::vector<value_type>& ) const   H        |X|X| |X|X|X|X|X|X|X| |X| |eol\n\
operator<<( std::vector<value_type>& )         H        | |X| | | | | | | | |X|X| |eol\n\
operator>>( std::array<value_type,N>& ) const  H  C++11 | | |X| | | | | | | | | | |eol\n\
operator<<( std::array<value_type,N>& )        H  C++11 | | |X| | | | | | | | | | |eol\n\
array<T,N>& operator=( const array<T,N>& )     D        | |X| | |X|X|X|X|X|X| | | |eol\n\
........................................................+-+-+-+-+-+-+-+-+-+-+-+-+-+\n\
........................................................|";

	std::cout << outputText;
	for( std::vector<bool>::size_type i = 0; i < testResults.size(); ++i ) std::cout << ( testResults[i] == 1 ? "P" : ( testResults[i] == -1 ? "?" : "F" ) ) << "|";
	std::cout << std::endl;

	return EXIT_SUCCESS;

}
