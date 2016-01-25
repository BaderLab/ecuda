CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ecuda)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)

#LIST(APPEND CUDA_NVCC_FLAGS -arch=sm_21 -O3 -Xptxas -v)
#LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -pedantic -flto")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

INCLUDE( ../local-config.cmake OPTIONAL )

INCLUDE_DIRECTORIES(../../Catch/include)

FILE( GLOB CUDA_SOURCES "*.cu" )
FOREACH( cudasrcfile ${CUDA_SOURCES} )
  GET_FILENAME_COMPONENT( cudaprogname ${cudasrcfile} NAME_WE )
  CUDA_ADD_EXECUTABLE( ${cudaprogname} ${cudasrcfile} )
ENDFOREACH( cudasrcfile ${CUDA_SOURCES} )
