file( GLOB ECUDA_BENCHMARK_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu" )

find_package( CUDAToolkit REQUIRED )

set( ECUDA_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" )
if( NOT ECUDA_CUDA_ARCHITECTURES )
	set( ECUDA_CUDA_ARCHITECTURES native )
endif()

foreach( benchmark_src IN LISTS ECUDA_BENCHMARK_SOURCES )
	get_filename_component( benchmark_name ${benchmark_src} NAME_WE )
	add_executable( "${benchmark_name}" "${benchmark_src}" )

	target_link_libraries( "${benchmark_name}" PRIVATE ecuda::ecuda )

	set_target_properties(
		"${benchmark_name}" PROPERTIES
		CUDA_ARCHITECTURES ${ECUDA_CUDA_ARCHITECTURES}
		CUDA_STANDARD 14
		CUDA_STANDARD_REQUIRED ON
		CUDA_SEPARABLE_COMPILATION ON
		POSITION_INDEPENDENT_CODE ON
	)

	target_compile_options(
		"${benchmark_name}"
		PRIVATE
		$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
	)

	set_target_properties(
		"${benchmark_name}" PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
	)

endforeach()
