cmake_minimum_required( VERSION 4.3.1 )
project( ecuda LANGUAGES CXX CUDA )

add_library( ecuda INTERFACE )
add_library( ecuda::ecuda ALIAS ecuda )

# the SYSTEM keyword marks these as system includes in order to
# suppress warnings about iterators (the specification has changed in C++17)
target_include_directories(
	ecuda
	SYSTEM INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

option( ECUDA_BUILD_TESTS "Build ecuda tests" OFF )

if( ECUDA_BUILD_TESTS )
	enable_testing()
	add_subdirectory(test)
endif()
