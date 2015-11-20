#include <iostream>
#include <list>
#include <vector>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/array.hpp"

#ifndef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
template<typename T,std::size_t N>
__global__ void kernel_test_iterators( const typename ecuda::array<T,N>::kernel_argument src, typename ecuda::array<T,N>::kernel_argument dest )
{
	typename ecuda::array<T,N>::iterator result = dest.begin();
	for( typename ecuda::array<T,N>::const_iterator iter = src.begin(); iter != src.end(); ++iter, ++result ) *result = *iter;
}
#endif

int main( int argc, char* argv[] ) {

	{
		std::cout << "TESTING CONSTRUCTORS" << std::endl;
		std::cout << "--------------------" << std::endl;
		{
			const std::size_t N = 1000;
			ecuda::array<double,N> deviceArray;
			std::vector<double> hostVector( N );
			std::cout << "ecuda::array() : " << std::boolalpha << ecuda::equal( deviceArray.begin(), deviceArray.end(), hostVector.begin() ) << std::endl;
		}
		{
			const std::size_t N = 1000;
			ecuda::array<double,N> deviceArray1;
			ecuda::fill( deviceArray1.begin(), deviceArray1.end(), 99.0 );
			ecuda::array<double,N> deviceArray2( deviceArray1 );
			std::vector<double> hostVector( N, 99.0 );
			std::cout << "ecuda::array( const ecuda::array& ) : " << std::boolalpha << ecuda::equal( deviceArray2.begin(), deviceArray2.end(), hostVector.begin() ) << std::endl;
		}
		#ifdef __CPP11_SUPPORTED__
		{
			const std::size_t N = 1000;
			ecuda::array<double,N> deviceArray1;
			ecuda::fill( deviceArray1.begin(), deviceArray1.end(), 99.0 );
			ecuda::array<double,N> deviceArray2( std::move(deviceArray1) );
			std::vector<double> hostVector( N, 99.0 );
			std::cout << "ecuda::array( ecuda::array&& ) :" << std::endl;
			std::cout << "  destination has contents : " << std::boolalpha << ecuda::equal( deviceArray2.begin(), deviceArray2.end(), hostVector.begin() ) << std::endl;
			std::cout << "  source now empty         : " << std::boolalpha << !static_cast<bool>(deviceArray1.data()) << std::endl;
		}
		#endif
		std::cout << std::endl;
	}
	{
		std::cout << "TESTING ASSIGNMENT OPERATORS" << std::endl;
		std::cout << "----------------------------" << std::endl;
		{
			const std::size_t N = 1000;
			ecuda::array<double,N> deviceArray1;
			ecuda::fill( deviceArray1.begin(), deviceArray1.end(), 99.0 );
			ecuda::array<double,N> deviceArray2 = deviceArray1;
			ecuda::fill( deviceArray1.begin(), deviceArray1.end(), 0.0 );
			std::vector<double> hostVector( N, 99.0 );
			std::cout << "ecuda::array::operator=(const array&) :" << std::endl;
			std::cout << "  assignment destination equal : " << std::boolalpha << ecuda::equal( deviceArray2.begin(), deviceArray2.end(), hostVector.begin() ) << std::endl;
			std::cout << "  assignment source unequal    : " << std::boolalpha << !ecuda::equal( deviceArray1.begin(), deviceArray1.end(), hostVector.begin() ) << std::endl;
		}
		#ifdef __CPP11_SUPPORTED__
		{
			const std::size_t N = 1000;
			ecuda::array<double,N> deviceArray1;
			ecuda::fill( deviceArray1.begin(), deviceArray1.end(), 99.0 );
			ecuda::array<double,N> deviceArray2 = std::move(deviceArray1);
			std::vector<double> hostVector( N, 99.0 );
			std::cout << "ecuda::array::operator=(array&&) : " << std::endl;
			std::cout << "  destination has contents : " << std::boolalpha << ecuda::equal( deviceArray2.begin(), deviceArray2.end(), hostVector.begin() ) << std::endl;
			std::cout << "  source now empty         : " << std::boolalpha << !static_cast<bool>(deviceArray1.data()) << std::endl;
		}
		#endif
		std::cout << std::endl;
	}
	{
		std::cout << "TESTING ACCESSORS" << std::endl;
		std::cout << "-----------------" << std::endl;
		{
			const std::size_t N = 1000;
			ecuda::array<double,N> deviceArray;
			#ifdef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
			for( typename ecuda::array<double,N>::size_type i = 0; i < deviceArray.size(); ++i ) deviceArray[i] = static_cast<double>(i);
			try {
				std::cout << "ecuda::array::at()       : " << std::boolalpha << ( deviceArray.at(10) == static_cast<double>(10) ) << std::endl;
				deviceArray.at( N );
				std::cout << "ecuda::array::at(N)      : exception NOT thrown as expected" << std::endl;
			} catch( std::out_of_range& ex ) {
				std::cout << "ecuda::array::at(N)      : exception thrown as expected" << std::endl;
			}
			std::cout << "ecuda::array::operator[] : " << std::boolalpha << ( deviceArray[10] == static_cast<double>(10) ) << std::endl;
			std::cout << "ecuda::array::front()    : " << std::boolalpha << ( deviceArray.front() == static_cast<double>(0) ) << std::endl;
			std::cout << "ecuda::array::back()     : " << std::boolalpha << ( deviceArray.back() == static_cast<double>(N-1) ) << std::endl;
			#else
			//ECUDA_STATIC_ASSERT(false,MUST_IMPLEMENT_ACCESSOR_AS_KERNEL);
			#endif
			std::cout << "ecuda::array::data()     : " << std::boolalpha << static_cast<bool>(deviceArray.data()) << std::endl;
			std::cout << "ecuda::array::empty()    : " << std::boolalpha << ( !deviceArray.empty() ) << std::endl;
			std::cout << "ecuda::array::size()     : " << std::boolalpha << ( deviceArray.size() == N ) << std::endl;
		}
		std::cout << std::endl;
	}
	{
		std::cout << "TESTING ITERATORS" << std::endl;
		std::cout << "-----------------" << std::endl;
		{
			const std::size_t N = 1000;
			ecuda::array<double,N> deviceArray;
			std::cout << "ecuda::array::end()-ecuda::array::begin()   : " << std::boolalpha << ( ( deviceArray.end()-deviceArray.begin() ) == N ) << std::endl;
			std::cout << "ecuda::array::rend()-ecuda::array::rbegin() : " << std::boolalpha << ( ( deviceArray.rbegin()-deviceArray.rend() ) == N ) << std::endl;
			//std::vector<double> hostVector( N );
			//ecuda::copy( deviceArray.rbegin(), deviceArray.rend(), hostVector.begin() );
		}
		std::cout << std::endl;
	}
	{
		std::cout << "TESTING TRANSFORMS" << std::endl;
		std::cout << "------------------" << std::endl;
		{
			const std::size_t N = 1000;
			ecuda::array<double,N> deviceArray1;
			deviceArray1.fill( static_cast<double>(99) );
			std::vector<double> hostVector1( N, static_cast<double>(99) );
			std::cout << "ecuda::array::fill() : " << std::boolalpha << ecuda::equal( deviceArray1.begin(), deviceArray1.end(), hostVector1.begin() ) << std::endl;
			ecuda::array<double,N> deviceArray2;
			deviceArray2.fill( static_cast<double>(66) );
			deviceArray1.swap( deviceArray2 );
			std::cout << "ecuda::array::swap() : " << std::boolalpha << ecuda::equal( deviceArray2.begin(), deviceArray2.end(), hostVector1.begin() ) << std::endl;
		}
		std::cout << std::endl;
	}

	const std::size_t N = 1000;
	std::vector<int> hostVector( N ); for( int i = 0; i < static_cast<int>(N); ++i ) hostVector[i] = i;
	ecuda::array<int,N> deviceArray;
	ecuda::copy( hostVector.begin(), hostVector.end(), deviceArray.begin() );

	#ifndef ECUDA_EMULATE_CUDA_WITH_HOST_ONLY
	{
		std::cout << "TESTING KERNELS" << std::endl;
		std::cout << "---------------" << std::endl;
		{
			ecuda::array<int,N> deviceArray2;
			CUDA_CALL_KERNEL_AND_WAIT( kernel_test_iterators<int,N><<<1,1>>>( deviceArray, deviceArray2 ) );
			//CUDA_CHECK_ERRORS();
			//CUDA_CALL( cudaDeviceSynchronize() );
			std::cout << "ecuda::array::iterator : " << std::boolalpha << ecuda::equal( deviceArray.begin(), deviceArray.end(), deviceArray2.begin() ) << std::endl;
		}
		std::cout << std::endl;
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

