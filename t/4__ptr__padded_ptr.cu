#include <iostream>
#include <vector>

#include "../include/ecuda/memory.hpp"
#include "../include/ecuda/impl/models.hpp"
#include "../include/ecuda/iterator.hpp"

int main( int argc, char* argv[] ) {

	char* mem = new char[92*5];
	{
		char* p = mem;
		unsigned counter = 0;
		for( std::size_t i = 0; i < 5; ++i ) {
			for( std::size_t j = 0; j < 10; ++j, p += sizeof(double), ++counter ) {
				*reinterpret_cast<double*>(p) = static_cast<double>(counter);
			}
			p += 12; // padding
		}
	}

	ecuda::padded_ptr<double> p( reinterpret_cast<double*>(mem), 92 );

	ecuda::impl::device_contiguous_row_matrix<double,double*> mat( p, 5, 10 );

	ecuda::device_contiguous_block_iterator<double,double*> begin( p, 10 );

	ecuda::padded_ptr<double> q( p );
	q.skip_bytes( p.get_pitch()*5 );
	ecuda::device_contiguous_block_iterator<double,double*> end( q, 10 );

	std::cout << "begin=" << begin.operator->() << std::endl;
	std::cout << "end  =" << end.operator->() << std::endl;
	std::cout << "distance=" << (end-begin) << std::endl;

	for( std::size_t i = 0; i < 5; ++i ) {
		for( std::size_t j = 0; j < 10; ++j ) {
			if( j ) std::cout << " ";
			std::cout << mat.at(i,j);
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	for( std::size_t i = 0; i < 5; ++i ) {
		typename ecuda::impl::device_contiguous_row_matrix<double,double*>::row_type row = mat.get_row(i);
		for( std::size_t j = 0; j < 10; ++j ) {
			if( j ) std::cout << " ";
			std::cout << row[j];
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	for( typename ecuda::impl::device_contiguous_row_matrix<double,double*>::iterator iter = mat.begin(); iter != mat.end(); ++iter ) std::cout << " " << *iter;
	std::cout << std::endl;
	std::cout << std::endl;

	ecuda::fill( mat.begin(), mat.end(), 99.0 );
	for( typename ecuda::impl::device_contiguous_row_matrix<double,double*>::iterator iter = mat.begin(); iter != mat.end(); ++iter ) std::cout << " " << *iter;
	std::cout << std::endl;
	std::cout << std::endl;

	/*
	std::vector<int> v( 100 );
	for( std::size_t i = 0; i < v.size(); ++i ) v[i] = i;

	ecuda::padded_ptr<int> p( &v.front(), sizeof(int)*10, 8, &v.front() );
	for( std::size_t i = 0; i < 10; ++i, ++p ) std::cout << "padded_ptr[" << p.get() << "]=" << (p.get()-&v.front()) << std::endl;
	*/

	delete [] mem;

	return 0;

}

