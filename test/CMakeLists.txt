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

	target_link_libraries( "${test_name}" PRIVATE ecuda::ecuda )

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
