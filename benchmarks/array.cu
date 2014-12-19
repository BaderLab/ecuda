#include <ctime>
#include <vector>
#include "../include/ecuda/array.hpp"

template<typename T>
__global__
void squareArray( ecuda::array<T> input ) {
	const int index = blockIdx.x*blockDim.x+threadIdx.x;
	if( index < input.size() ) {
		T& value = input[index];
		value *= value;
	}
}

template<typename T>
__global__
void squareArray( T* input, std::size_t n ) {
	const int index = blockIdx.x*blockDim.x+threadIdx.x;
	if( index < n ) {
		T& value = input[index];
		value *= value;
	}
}

int main( int argc, char* argv[] ) {

	std::time_t start, end;
	std::time(&start);

	const std::size_t N = 100000;
	const std::size_t THREADS = 800;

	std::vector<int> hostData( N );

	ecuda::array<int> deviceData( N );
	deviceData << hostData;

	int* rawData = NULL;
	CUDA_CALL( cudaMalloc( reinterpret_cast<void**>(&rawData), N*sizeof(int) ) );
	CUDA_CALL( cudaMemcpy( reinterpret_cast<void*>(rawData), reinterpret_cast<const void*>(&hostData.front()), N*sizeof(int), cudaMemcpyHostToDevice ) );

	dim3 grid( (N+THREADS-1)/THREADS ), threads( THREADS );
	squareArray<int><<<grid,threads>>>( deviceData );
	CUDA_CALL( cudaDeviceSynchronize() );
	CUDA_CHECK_ERRORS();

	squareArray<int><<<grid,threads>>>( rawData, N );
	CUDA_CALL( cudaDeviceSynchronize() );
	CUDA_CHECK_ERRORS();

	return EXIT_SUCCESS;

}
