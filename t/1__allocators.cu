#include "../include/ecuda/allocators.hpp"

template<typename T,class Alloc>
__global__ void destroy( Alloc alloc, typename ecuda::device_allocator<T>::pointer p, std::size_t n ) {
	const std::size_t index = blockIdx.x*blockDim.x+threadIdx.x;
	if( index < n ) alloc.destroy( p+index );
}

int main( int argc, char* argv[] ) {

	{
		ecuda::host_allocator<double> hostAllocator1;
		ecuda::host_allocator<double> hostAllocator2( hostAllocator1 );
		ecuda::host_allocator<int> hostAllocator3( hostAllocator2 );
		typename ecuda::host_allocator<double>::pointer p = hostAllocator1.allocate( 1000 );
		typename ecuda::host_allocator<double>::pointer q( p );
		for( std::size_t i = 0; i < 1000; ++i, ++q ) hostAllocator1.construct( q, static_cast<typename ecuda::host_allocator<double>::value_type>(i) );
		q = p;
		for( std::size_t i = 0; i < 1000; ++i, ++q ) hostAllocator1.destroy( q );
		hostAllocator1.deallocate( p, 1000 );
	}

	{
		ecuda::device_allocator<double> deviceAllocator1;
		ecuda::device_allocator<double> deviceAllocator2( deviceAllocator1 );
		ecuda::device_allocator<int> deviceAllocator3( deviceAllocator2 );
		typename ecuda::device_allocator<double>::pointer p = deviceAllocator1.allocate( 1000 );
		typename ecuda::device_allocator<double>::pointer q( p );
		for( std::size_t i = 0; i < 1000; ++i, ++q ) deviceAllocator1.construct( q, static_cast<typename ecuda::device_allocator<double>::value_type>(i) );
		q = p;
		destroy<double><<<1,1000>>>( deviceAllocator1, p, 1000 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		deviceAllocator1.deallocate( p, 1000 );
	}

	{
		ecuda::device_pitch_allocator<double> deviceAllocator1;
		ecuda::device_pitch_allocator<double> deviceAllocator2( deviceAllocator1 );
		ecuda::device_pitch_allocator<int> deviceAllocator3( deviceAllocator2 );
		typename ecuda::device_pitch_allocator<double>::size_type pitch;
		typename ecuda::device_pitch_allocator<double>::pointer p = deviceAllocator1.allocate( 100, 100, pitch );
		typename ecuda::device_pitch_allocator<double>::pointer q( p );
		for( std::size_t i = 0; i < 1000; ++i, ++q ) deviceAllocator1.construct( q, static_cast<typename ecuda::device_pitch_allocator<double>::value_type>(i) );
		q = p;
		destroy<double><<<1,1000>>>( deviceAllocator1, p, 1000 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		deviceAllocator1.deallocate( p, 1000 );
	}

	return 0;

}
