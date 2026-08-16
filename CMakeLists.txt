cmake_minimum_required( VERSION 4.3.1 )
project( ecuda LANGUAGES CXX )

option( ECUDA_ENABLE_CUDA "Build ecuda CUDA support" ON )

include( CheckLanguage )

set( ECUDA_HAVE_CUDA OFF )

if( ECUDA_ENABLE_CUDA )
        check_language( CUDA )
        if( CMAKE_CUDA_COMPILER )
                enable_language( CUDA )
                if( NOT DEFINED CMAKE_CUDA_ARCHITECTURES )
                        set( CMAKE_CUDA_ARCHITECTURES native )
                endif()
                message( STATUS "CUDA found: building ecuda with CUDA support" )
                set( ECUDA_HAVE_CUDA ON )
        else()
                message( STATUS "CUDA not found: building ecuda without CUDA support" )
                set( ECUDA_HAVE_CUDA OFF )
        endif()
endif()

add_library( ecuda INTERFACE )
add_library( ecuda::ecuda ALIAS ecuda )

target_compile_definitions(
	ecuda
	INTERFACE
		ECUDA_USE_CUDA_RUNTIME=$<BOOL:${ECUDA_HAVE_CUDA}>
)

# the SYSTEM keyword marks these as system includes in order to
# suppress warnings about iterators (the specification has changed in C++17)
target_include_directories(
	ecuda
	SYSTEM INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

option( ECUDA_BUILD_BENCHMARKS "Build ecuda benchmarks" OFF )

if( ECUDA_BUILD_BENCHMARKS AND ECUDA_HAVE_CUDA )
	add_subdirectory(benchmarks)
endif()

option( ECUDA_BUILD_TESTS "Build ecuda tests" OFF )

if( ECUDA_BUILD_TESTS )
	enable_testing()
	add_subdirectory(test)
endif()
