CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ecuda)

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_21 -O3 -Xptxas -v")
#LIST(APPEND CMAKE_CXX_FLAGS "-O3 -Wall -pedantic")

CUDA_ADD_EXECUTABLE(array_add_cuda array_add_cuda.cu)
CUDA_ADD_EXECUTABLE(array_add_ecuda array_add_ecuda.cu)
