#include <iostream>
#include <list>
#include <vector>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/array.hpp"

#ifndef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
template<typename T,std::size_t N>
//__global__ void testIterators( const typename ecuda::array<T,N>::argument src, ecuda::array<T,N> dest ) {
__global__ void testIterators( const typename ecuda::array<T,N>::argument src, typename ecuda::array<T,N>::argument dest ) {
	typename ecuda::array<T,N>::iterator result = dest.begin();
	for( typename ecuda::array<T,N>::const_iterator iter = src.begin(); iter != src.end(); ++iter, ++result ) *result = *iter;
}
#endif

int main( int argc, char* argv[] ) {

	{
		std::cerr << "TESTING CONSTRUCTORS" << std::endl;
		std::cerr << "--------------------" << std::endl;
		{
			const std::size_t N = 1000;
			ecuda::array<double,N> deviceArray;
			std::vector<double> hostVector( N );
			std::cerr << "ecuda::array() : " << std::boolalpha << ecuda::equal( deviceArray.begin(), deviceArray.end(), hostVector.begin() ) << std::endl;
		}
		{
			const std::size_t N = 1000;
			ecuda::array<double,N> deviceArray1;
			ecuda::array<double,N> deviceArray2( deviceArray1 );
			ecuda::fill( deviceArray1.begin(), deviceArray1.end(), 99.0 ); // filling deviceArray1 should be reflected in deviceArray2
			std::vector<double> hostVector( N, 99.0 );
			std::cerr << "ecuda::array( const ecuda::array& ) : " << std::boolalpha << ecuda::equal( deviceArray2.begin(), deviceArray2.end(), hostVector.begin() ) << std::endl;
		}
		#ifdef __CPP11_SUPPORTED__
		{
			std::cerr << "ecuda::array( ecuda::array&& ) : TEST NOT IMPLEMENTED" << std::endl;
		}
		#endif
		std::cerr << std::endl;
	}
	{
		std::cerr << "TESTING ACCESSORS" << std::endl;
		std::cerr << "-----------------" << std::endl;
		{
			const std::size_t N = 1000;
			ecuda::array<double,N> deviceArray;
			#ifdef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
			for( typename ecuda::array<double,N>::size_type i = 0; i < deviceArray.size(); ++i ) deviceArray[i] = static_cast<double>(i);
			std::cerr << "ecuda::array::operator[] : " << std::boolalpha << ( deviceArray[10] == static_cast<double>(10) ) << std::endl;
			std::cerr << "ecuda::array::front()    : " << std::boolalpha << ( deviceArray.front() == static_cast<double>(0) ) << std::endl;
			std::cerr << "ecuda::array::back()     : " << std::boolalpha << ( deviceArray.back() == static_cast<double>(N-1) ) << std::endl;
			//std::vector<double> hostVector( N );
			//ecuda::copy( deviceArray.rbegin(), deviceArray.rend(), hostVector.begin() );
			//std::cerr << "ecuda::array::rbegin(),rend() : " << std::boolalpha << ( deviceArray.front() == static_cast<double>(N-1) ) << "," << std::boolalpha << ( deviceArray.back() == static_cast<double>(0) ) << std::endl;
			#else
//			ECUDA_STATIC_ASSERT(false,MUST_IMPLEMENT_ACCESSOR_AS_KERNEL);
			#endif
			std::cerr << "ecuda::array::empty()    : " << std::boolalpha << ( !deviceArray.empty() ) << std::endl;
			std::cerr << "ecuda::array::size()     : " << std::boolalpha << ( deviceArray.size() == N ) << std::endl;
			std::cerr << "ecuda::array::data()     : " << std::boolalpha << ( deviceArray.data() > 0 ) << std::endl;
		}
		std::cerr << std::endl;
	}
	{
		std::cerr << "TESTING TRANSFORMS" << std::endl;
		std::cerr << "------------------" << std::endl;
		{
			const std::size_t N = 1000;
			ecuda::array<double,N> deviceArray1;
			deviceArray1.fill( static_cast<double>(99) );
			std::vector<double> hostVector1( N, static_cast<double>(99) );
			std::cerr << "ecuda::array::fill() : " << std::boolalpha << ecuda::equal( deviceArray1.begin(), deviceArray1.end(), hostVector1.begin() ) << std::endl;
			ecuda::array<double,N> deviceArray2;
			deviceArray2.fill( static_cast<double>(66) );
			deviceArray1.swap( deviceArray2 );
			std::cerr << "ecuda::array::swap() : " << std::boolalpha << ecuda::equal( deviceArray2.begin(), deviceArray2.end(), hostVector1.begin() ) << std::endl;
		}
		std::cerr << std::endl;
	}

	std::vector<int> hostVector( 100 );	for( unsigned i = 0; i < 100; ++i ) hostVector[i] = i;

	//ecuda::array<int,100> deviceArray; deviceArray.operator<<( hostVector );
	//if( !ecuda::equal( hostVector.begin(), hostVector.end(), deviceArray.begin() ) ) throw std::runtime_error( "operator<< failed" );

	ecuda::array<int,100> deviceArray;
	ecuda::copy( hostVector.begin(), hostVector.end(), deviceArray.begin() );

	#ifndef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
	{
		ecuda::array<int,100> deviceArray2;
		testIterators<int,100><<<1,1>>>( deviceArray, deviceArray2 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		std::cout << "EQUAL " << ( deviceArray == deviceArray2 ? "true" : "false" ) << std::endl;
		std::cout << "LESSTHAN " << ( deviceArray < deviceArray2 ? "true" : "false" ) << std::endl;
	}
	#endif

	ecuda::reverse( deviceArray.begin(), deviceArray.end() );

	std::cout << "HOST   VECTOR ="; for( unsigned i = 0; i < hostVector.size(); ++i ) std::cout << " " << hostVector[i]; std::cout << std::endl;
	{
		std::vector<int> tmp( 100 );
		ecuda::copy( deviceArray.begin(), deviceArray.end(), tmp.begin() );
		std::cout << "DEVICE VECTOR ="; for( unsigned i = 0; i < tmp.size(); ++i ) std::cout << " " << tmp[i]; std::cout << std::endl;
	}

	//int* p = 0;
	//typename ecuda::pointer_traits<int*>::unmanaged_pointer q = ecuda::pointer_traits<int*>().make_unmanaged(p);
	//typename ecuda::pointer_traits<int*>::unmanaged_pointer r = ecuda::pointer_traits<int*>::cast_unmanaged(q);

	return EXIT_SUCCESS;

}

