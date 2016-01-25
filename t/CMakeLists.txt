CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ecuda)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

INCLUDE( ../local-config.cmake OPTIONAL )

#INCLUDE_DIRECTORIES(../../Catch/include)
#MESSAGE( "CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}" )

FILE( GLOB CUDA_SOURCES "*.cu" )
FOREACH( cudasrcfile ${CUDA_SOURCES} )
  GET_FILENAME_COMPONENT( cudaprogname ${cudasrcfile} NAME_WE )
  CUDA_ADD_EXECUTABLE( ${cudaprogname} ${cudasrcfile} )
  FILE( COPY ${cudasrcfile} DESTINATION host )
  FILE( RENAME ${CMAKE_BINARY_DIR}/host/${cudaprogname}.cu ${CMAKE_BINARY_DIR}/host/${cudaprogname}.cpp )
  ADD_EXECUTABLE( ${cudaprogname}_host ${CMAKE_BINARY_DIR}/host/${cudaprogname}.cpp )
#  ADD_EXECUTABLE( ${cudaprogname}_host ${cudasrcfile} )
ENDFOREACH( cudasrcfile ${CUDA_SOURCES} )

#FILE( GLOB HOST_SOURCES "*.cpp" )
#FOREACH( hostsrcfile ${HOST_SOURCES} )
#  GET_FILENAME_COMPONENT( hostprogname ${hostsrcfile} NAME_WE )
#ENDFOREACH( hostsrcfile ${HOST_SOURCES} )
