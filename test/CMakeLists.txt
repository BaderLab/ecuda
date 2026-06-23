include( FetchContent )

FetchContent_Declare(
	doctest
	GIT_REPOSITORY https://github.com/doctest/doctest.git
	GIT_TAG v2.5.2
)

FetchContent_MakeAvailable( doctest )

set( ESTD_SOURCE_DIR "" CACHE PATH "existing estd source checkout" )
set( ESTD_ROOT "" CACHE PATH "existing installed estd prefix" )
set( ESTD_GIT_REPOSITORY "https://github.com/BaderLab/estd.git" CACHE STRING "Git repository for estd" )
set( ESTD_GIT_TAG "master" CACHE STRING "Git tag, branch, or commit for estd" )

function( require_estd )
	if( TARGET estd::estd )
		return()
	endif()

	if( ESTD_ROOT )
		list( PREPEND CMAKE_PREFIX_PATH "${ESTD_ROOT}" )
		find_package( estd CONFIG REQUIRED )
		return()
	endif()

	if( ESTD_SOURCE_DIR )
		add_subdirectory( "${ESTD_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/estd-build" )
		return()
	endif()

	FetchContent_Declare(
		estd
		GIT_REPOSITORY "${ESTD_GIT_REPOSITORY}"
		GIT_TAG        "${ESTD_GIT_TAG}"
		SOURCE_DIR     "${CMAKE_BINARY_DIR}/_deps/estd-src"
		BINARY_DIR     "${CMAKE_BINARY_DIR}/_deps/estd-build"		
	)

	FetchContent_MakeAvailable( estd )
endfunction()

require_estd()

file( GLOB ECUDA_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu" )

find_package( CUDAToolkit REQUIRED )

set( ECUDA_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" )
if( NOT ECUDA_CUDA_ARCHITECTURES )
	set( ECUDA_CUDA_ARCHITECTURES native )
endif()

foreach( test_src IN LISTS ECUDA_TEST_SOURCES )
	get_filename_component( test_name "${test_src}" NAME_WE )
	add_executable( "${test_name}" "${test_src}" )

	set_target_properties(
		"${test_name}" PROPERTIES
		CUDA_ARCHITECTURES ${ECUDA_CUDA_ARCHITECTURES}
		CUDA_STANDARD 14
		CUDA_STANDARD_REQUIRED ON
		CUDA_SEPARABLE_COMPILATION ON
		POSITION_INDEPENDENT_CODE ON
	)

	target_link_libraries( "${test_name}" PRIVATE ecuda::ecuda estd::estd doctest::doctest )

	target_compile_options(
		"${test_name}"
		PRIVATE
		$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
	)

	add_test(
		NAME "${test_name}"
		COMMAND "${test_name}"
	)

endforeach()
