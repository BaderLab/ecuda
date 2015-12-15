#include <iostream>
#include <list>
#include <vector>

#include "../include/ecuda/algorithm.hpp"
#include "../include/ecuda/allocators.hpp"
#include "../include/ecuda/models.hpp"
#include "../include/ecuda/matrix.hpp"

template<typename T,typename U>
void testMatrix( const ecuda::device_contiguous_row_matrix<T,U>& matrix ) {
	std::vector<T> v( matrix.number_columns() );
	for( unsigned i = 0; i < matrix.number_rows(); ++i ) {
		ecuda::copy( matrix[i].begin(), matrix[i].end(), v.begin() );
		std::cout << "ROW[" << i << "]";
		for( unsigned j = 0; j < matrix.number_columns(); ++j ) std::cout << " " << v[i];
		std::cout << std::endl;
	}
}

template<class InputContainer,class OutputContainer>
inline bool testCopy( InputContainer& input, OutputContainer& output ) {
	ecuda::fill( output.begin(), output.end(), typename OutputContainer::value_type() );
	ecuda::copy( input.begin(), input.end(), output.begin() );
	return ecuda::equal( input.begin(), input.end(), output.begin() );
}

template<typename T,typename U>
__global__ void testKernel( const ecuda::device_contiguous_row_matrix<T,U> matrix, ecuda::device_sequence<T,U> vector ) {
	ecuda::copy( matrix.get_column(0).begin(), matrix.get_column(0).end(), vector.begin() );
}

int main( int argc, char* argv[] ) {

	ecuda::device_sequence< int, ecuda::unique_ptr<int> > device_sequence_noncontiguous1( ecuda::unique_ptr<int>( ecuda::device_allocator<int>().allocate( 100 ) ), 100 );
	ecuda::device_contiguous_sequence< int, ecuda::unique_ptr<int> > device_sequence_contiguous1( ecuda::unique_ptr<int>( ecuda::device_allocator<int>().allocate( 100 ) ), 100 );
	std::vector<int> host_sequence_contiguous1( 100 ); for( std::size_t i = 0; i < host_sequence_contiguous1.size(); ++i ) host_sequence_contiguous1[i] = i;
	std::list<int> host_sequence_noncontiguous1( 100 );

	if( !testCopy( host_sequence_contiguous1, device_sequence_noncontiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( host_sequence_contiguous1, device_sequence_contiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( host_sequence_contiguous1, host_sequence_contiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( host_sequence_contiguous1, host_sequence_noncontiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );

	if( !testCopy( host_sequence_noncontiguous1, device_sequence_noncontiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( host_sequence_noncontiguous1, device_sequence_contiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( host_sequence_noncontiguous1, host_sequence_contiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( host_sequence_noncontiguous1, host_sequence_noncontiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );

	if( !testCopy( device_sequence_noncontiguous1, device_sequence_noncontiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( device_sequence_noncontiguous1, device_sequence_contiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( device_sequence_noncontiguous1, host_sequence_contiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( device_sequence_noncontiguous1, host_sequence_noncontiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );

	if( !testCopy( device_sequence_contiguous1, device_sequence_noncontiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( device_sequence_contiguous1, device_sequence_contiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( device_sequence_contiguous1, host_sequence_contiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( device_sequence_contiguous1, host_sequence_noncontiguous1 ) ) throw std::runtime_error( EXCEPTION_MSG("") );

	ecuda::device_sequence< double, ecuda::unique_ptr<double> > device_sequence_noncontiguous2( ecuda::unique_ptr<double>( ecuda::device_allocator<double>().allocate( 100 ) ), 100 );
	ecuda::device_contiguous_sequence< double, ecuda::unique_ptr<double> > device_sequence_contiguous2( ecuda::unique_ptr<double>( ecuda::device_allocator<double>().allocate( 100 ) ), 100 );
	std::vector<double> host_sequence_contiguous2( 100 );
	std::list<double> host_sequence_noncontiguous2( 100 );

	if( !testCopy( host_sequence_contiguous1, device_sequence_noncontiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( host_sequence_contiguous1, device_sequence_contiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( host_sequence_contiguous1, host_sequence_contiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( host_sequence_contiguous1, host_sequence_noncontiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );

	if( !testCopy( host_sequence_noncontiguous1, device_sequence_noncontiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( host_sequence_noncontiguous1, device_sequence_contiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( host_sequence_noncontiguous1, host_sequence_contiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( host_sequence_noncontiguous1, host_sequence_noncontiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );

	if( !testCopy( device_sequence_noncontiguous1, device_sequence_noncontiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( device_sequence_noncontiguous1, device_sequence_contiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( device_sequence_noncontiguous1, host_sequence_contiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( device_sequence_noncontiguous1, host_sequence_noncontiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );

	if( !testCopy( device_sequence_contiguous1, device_sequence_noncontiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( device_sequence_contiguous1, device_sequence_contiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( device_sequence_contiguous1, host_sequence_contiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );
	if( !testCopy( device_sequence_contiguous1, host_sequence_noncontiguous2 ) ) throw std::runtime_error( EXCEPTION_MSG("") );

	{
		std::size_t pitch;
		int* p = ecuda::device_pitch_allocator<int>().allocate( 20, 10, pitch );
		ecuda::padded_ptr<int> paddedPointer( p, pitch, 20, p );
		ecuda::device_contiguous_row_matrix< int, ecuda::padded_ptr<int> > matrix( paddedPointer, 10, 20 );
		testMatrix( matrix );
		matrix.get_row(0);
		matrix.get_column(0);
	}


	ecuda::unique_ptr<int> ptr;
	ecuda::device_contiguous_row_matrix< int, ecuda::unique_ptr<int> > matrix1( ptr, 10, 20 );
	testMatrix( matrix1 );
	matrix1.get_row(0);
	matrix1.get_column(0);

	ecuda::device_contiguous_row_matrix< int, ecuda::naked_ptr<int> > matrix2( ecuda::naked_ptr<int>(ptr.get()), 10, 20 );
	testMatrix( matrix2 );
	matrix2.get_row(0);
	matrix2.get_column(0);

	ecuda::device_contiguous_row_matrix< int, ecuda::padded_ptr< int, ecuda::shared_ptr<int> > > matrix3( ecuda::padded_ptr< int, ecuda::shared_ptr<int> >( ecuda::shared_ptr<int>(), 100, 100, ecuda::shared_ptr<int>() ) );
	testMatrix( matrix3 );
	matrix3.get_row(0);
	matrix3.get_column(0);
	testCopy( matrix3, matrix3 );

	testKernel<<<1,1>>>( matrix1, device_sequence_noncontiguous1 );

	return EXIT_SUCCESS;

}

