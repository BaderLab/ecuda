#include <numeric>
#include <vector>

#include "../include/ecuda/event.hpp"
#include "../include/ecuda/matrix.hpp"

#define THREADS 480
#define N 10000000
#define ROWS 1000
#define COLUMNS 10000
#define ROUNDS 100

/*
template<typename T>
__global__
void fill_with_consecutive_values( typename ecuda::matrix<T>::kernel_argument matrix )
{
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x;
	const std::size_t y = blockIdx.y*blockDim.y+threadIdx.y;
	if( x < matrix.number_rows() and y < matrix.number_columns() ) {
		matrix[x][y] = x;
	}
}
*/

template<typename T,class Alloc>
__global__
void copy_matrices( const typename ecuda::matrix<T,Alloc>::kernel_argument src, typename ecuda::matrix<T,Alloc>::kernel_argument dest )
{
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x;
	const std::size_t y = blockIdx.y*blockDim.y+threadIdx.y;
	if( x < src.number_rows() and y < src.number_columns() ) dest(x,y) = src(x,y);
}

/*
template<typename T>
__global__
void fill_with_consecutive_values( T* ptr, const std::size_t rows, const std::size_t columns, const size_t pitch )
{
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x;
	const std::size_t y = blockIdx.y*blockDim.y+threadIdx.y;
	if( x < rows and y < columns ) {
		T* p = reinterpret_cast<T*>( reinterpret_cast<char*>(ptr) + (x*pitch) ) + y;
		*p = x;
	}
}
*/

template<typename T>
__global__
void copy_matrices( const T* src, T* dest, const std::size_t rows, const std::size_t columns, const size_t src_pitch, const size_t dest_pitch )
{
	const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x;
	const std::size_t y = blockIdx.y*blockDim.y+threadIdx.y;
	if( x < rows and y < columns ) {
		src  = reinterpret_cast<const T*>( reinterpret_cast<const char*>(src) + (x*src_pitch) ) + y;
		dest = reinterpret_cast<T*>( reinterpret_cast<char*>(dest) + (x*dest_pitch) ) + y;
		*dest = *src;
	}
}

template<typename T,class Alloc>
__global__
void copy_columns( const typename ecuda::matrix<T,Alloc>::kernel_argument src, typename ecuda::matrix<T,Alloc>::kernel_argument dest )
{
	const std::size_t t = blockIdx.x*blockDim.x+threadIdx.x;
	if( t < src.number_columns() ) {
		typename ecuda::matrix<T>::const_column_type srcColumn = src.get_column(t);
		typename ecuda::matrix<T>::column_type destColumn = dest.get_column(t);
		ecuda::copy( srcColumn.begin(), srcColumn.end(), destColumn.begin() );
	}
}

template<typename T>
__global__
void copy_columns( const T* src, T* dest, const std::size_t rows, const std::size_t columns, const size_t src_pitch, const size_t dest_pitch )
{
	const std::size_t t = blockIdx.x*blockDim.x+threadIdx.x;
	if( t < columns ) {
		src += t;
		dest += t;
		for( std::size_t i = 0; i < rows; ++i ) {
			src  = reinterpret_cast<const T*>( reinterpret_cast<const char*>(src) + src_pitch );
			dest = reinterpret_cast<T*>( reinterpret_cast<char*>(dest) + dest_pitch );
			*dest = *src;
		}
	}
}


void perform_tasks_with_ecuda();
void perform_tasks_old_school();

int main( int argc, char* argv[] )
{

	{
		ecuda::event start, stop;
		start.record();
		perform_tasks_old_school();
		stop.record();
		stop.synchronize();
		std::cout << "EXECUTION TIME (CUDA API): " << (stop-start) << "ms" << std::endl;
	}

	{
		ecuda::event start, stop;
		start.record();
		perform_tasks_with_ecuda();
		stop.record();
		stop.synchronize();
		std::cout << "EXECUTION TIME (ECUDA)   : " << (stop-start) << "ms" << std::endl;
	}

	return 0;

}

/// ecuda::event start, stop;
///
/// // ... specify thread grid/blocks
///
/// start.record();
/// kernelFunction<<<grid,block>>>( ... ); // call the kernel
/// stop.record();
/// stop.synchronize(); // wait until kernel finishes executing
///
/// std::cout << "EXECUTION TIME: " << ( stop - start ) << "ms" << std::endl;

void perform_tasks_with_ecuda()
{

	const ecuda::matrix<double> deviceMatrix1( ROWS, COLUMNS );
	ecuda::matrix<double> deviceMatrix2( ROWS, COLUMNS );

	// fill with values many times
	std::vector<double> times( ROUNDS );
	for( unsigned i = 0; i < ROUNDS; ++i ) {
		ecuda::event start, stop;
		start.record();
		dim3 grid( ROWS, (COLUMNS+THREADS-1)/THREADS ), threads( THREADS, 1 );
		//CUDA_CALL_KERNEL_AND_WAIT( fill_with_consecutive_values<double><<<grid,threads>>>( deviceMatrix ) );
		//CUDA_CALL_KERNEL_AND_WAIT( copy_matrices<double,typename ecuda::matrix<double>::allocator_type><<<grid,threads>>>( deviceMatrix1, deviceMatrix2 ) );
		CUDA_CALL_KERNEL_AND_WAIT( copy_columns<double,typename ecuda::matrix<double>::allocator_type><<<grid,threads>>>( deviceMatrix1, deviceMatrix2 ) );
		stop.record();
		stop.synchronize();
		times[i] = (stop-start);
	}

	const float totalTime = std::accumulate( times.begin(), times.end(), static_cast<float>(0) );
	std::cout << "AVERAGE KERNEL TIME: " << std::fixed << (totalTime/static_cast<float>(ROUNDS)) << std::endl;

}

void perform_tasks_old_school()
{

	double* ptr;
	size_t pitch;
	cudaMallocPitch( reinterpret_cast<void**>(&ptr), &pitch, COLUMNS*sizeof(double), ROWS );

	std::vector<float> times( ROUNDS );
	for( unsigned i = 0; i < ROUNDS; ++i ) {
		ecuda::event start, stop;
		start.record();
		dim3 grid( 1, (N+THREADS-1)/THREADS ), threads( THREADS, 1 );
		//fill_with_consecutive_values<double><<<grid,threads>>>( ptr, ROWS, COLUMNS, pitch );
		//copy_matrices<<<grid,threads>>>( ptr, ptr, ROWS, COLUMNS, pitch, pitch );
		copy_columns<<<grid,threads>>>( ptr, ptr, ROWS, COLUMNS, pitch, pitch );
		cudaDeviceSynchronize();
		stop.record();
		stop.synchronize();
		times[i] = (stop-start);
	}
	const float totalTime = std::accumulate( times.begin(), times.end(), static_cast<float>(0) );
	std::cout << "AVERAGE KERNEL TIME: " << std::fixed << (totalTime/static_cast<float>(ROUNDS)) << std::endl;

	cudaFree( ptr );

}
