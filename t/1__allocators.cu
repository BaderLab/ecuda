#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/type_traits.hpp"

#define THREADS 320

template<typename T,class Alloc>
__global__ void destroy( Alloc alloc, typename Alloc::pointer p, std::size_t n ) {
	const std::size_t index = blockIdx.x*blockDim.x+threadIdx.x;
	if( index < n ) alloc.destroy( p+index );
}

template<typename T>
__global__ void fill_with_sequential_values( T* ptr, const std::size_t n ) {
	const std::size_t index = blockIdx.x*blockDim.x+threadIdx.x;
	if( index < n ) ptr[index] = static_cast<T>(index);
}

//template<typename T> struct __change_type { typedef int type; };
//template<> struct __change_type<int> { typedef double type; };

template<class Alloc,class OtherAlloc>
void test_allocator_ctors() {

	// create an instance with default ctor
	Alloc allocator1;

	// create an instance using copy ctor
	Alloc allocator2( allocator1 );

	// create an instance with a different allocator template
	OtherAlloc allocator3( allocator2 );

}

template<typename P,typename Q>
struct deviceCopy {
	void copy( P* ptr, Q* dest, std::size_t n ) {
		if( cudaMemcpy( reinterpret_cast<void*>(dest), reinterpret_cast<const void*>(ptr), n*sizeof(P), cudaMemcpyDeviceToHost ) != cudaSuccess ) throw std::runtime_error("deviceCopy::copy failed" );
	}
};

template<typename P,typename Q,typename R>
struct deviceCopy<ecuda::padded_ptr<P,Q>,R> {
	void copy( ecuda::padded_ptr<P,Q>& ptr, R* dest, std::size_t n ) {
		if(
			cudaMemcpy2D(
				reinterpret_cast<void*>(dest),
				ptr.get_width()*sizeof(P),
				reinterpret_cast<const void*>(typename ecuda::pointer_traits< ecuda::padded_ptr<P,Q> >::undress(ptr)),
				ptr.get_pitch(),
				ptr.get_width()*sizeof(P),
				n/ptr.get_width(),
				cudaMemcpyDeviceToHost
			)
		!= cudaSuccess ) throw std::runtime_error("deviceCopy::copy failed" );
	}
};

template<typename PointerType>
void test_device_memory_validity( PointerType ptr ) {

	typedef typename ecuda::pointer_traits<PointerType>::element_type element_type;

	// expect 10,000 element allocation
	dim3 grid( 1, (10000+THREADS-1)/THREADS ), block( 1, THREADS );
	fill_with_sequential_values<element_type><<<grid,block>>>( ptr, 10000 );
	CUDA_CHECK_ERRORS();
	CUDA_CALL( cudaDeviceSynchronize() );

}


int main( int argc, char* argv[] ) {

	test_allocator_ctors< ecuda::host_allocator<double>, ecuda::host_allocator<int> >();
	test_allocator_ctors< ecuda::device_allocator<double>, ecuda::device_allocator<int> >();
	test_allocator_ctors< ecuda::device_pitch_allocator<double>, ecuda::device_pitch_allocator<int> >();

	{
		ecuda::host_allocator<double> hostAllocator;
		typename ecuda::host_allocator<double>::pointer p = hostAllocator.allocate( 1000 );
		typename ecuda::host_allocator<double>::pointer q( p );
		for( std::size_t i = 0; i < 1000; ++i, ++q ) hostAllocator.construct( q, static_cast<typename ecuda::host_allocator<double>::value_type>(i) );
		q = p;
		for( std::size_t i = 0; i < 1000; ++i, ++q ) hostAllocator.destroy( q );
		hostAllocator.deallocate( p, 1000 );
	}


	{
		ecuda::device_allocator<double> deviceAllocator;
		typename ecuda::device_allocator<double>::pointer p = deviceAllocator.allocate( 1000 );
		typename ecuda::device_allocator<double>::pointer q( p );
		for( std::size_t i = 0; i < 1000; ++i, ++q ) deviceAllocator.construct( q, static_cast<typename ecuda::device_allocator<double>::value_type>(i) );
		q = p;
		destroy<double><<<1,1000>>>( deviceAllocator, p, 1000 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		test_device_memory_validity( p );
		deviceAllocator.deallocate( p, 1000 );
	}

	{
		ecuda::device_pitch_allocator<double> deviceAllocator1;
		ecuda::device_pitch_allocator<double> deviceAllocator2( deviceAllocator1 );
		ecuda::device_pitch_allocator<int> deviceAllocator3( deviceAllocator2 );
		//typename ecuda::device_pitch_allocator<double>::size_type pitch;
		typename ecuda::device_pitch_allocator<double>::pointer p( deviceAllocator1.allocate( 100, 100 ) );
std::cerr << p << std::endl;
		typename ecuda::device_pitch_allocator<double>::pointer q( p );
		for( std::size_t i = 0; i < 1000; ++i, ++q ) {
			deviceAllocator1.construct( q, static_cast<typename ecuda::device_pitch_allocator<double>::value_type>(i) );
			std::cerr << i << "\t" << std::dec << q.get() << std::endl;
//			std::cerr << i << "\t" << q.get() << std::endl;
		}
		q = p;
		destroy<double><<<1,1000>>>( deviceAllocator1, p, 1000 );
		CUDA_CHECK_ERRORS();
		CUDA_CALL( cudaDeviceSynchronize() );
		deviceAllocator1.deallocate( p, 1000 );
	}

	return 0;

}
