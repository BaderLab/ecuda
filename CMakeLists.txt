cmake_minimum_required( VERSION 4.3.1 )
project( ecuda LANGUAGES CXX CUDA )

add_library( ecuda INTERFACE )
add_library( ecuda::ecuda ALIAS ecuda )

target_include_directories( ecuda INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include> )

target_compile_options(
	ecuda INTERFACE
	$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
)

option( ECUDA_BUILD_TESTS "Build ecuda tests" OFF )

if( ECUDA_BUILD_TESTS )
	enable_testing()
	add_subdirectory(test)
endif()
