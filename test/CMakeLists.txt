CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ecuda)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(../../Catch/include)

CUDA_ADD_EXECUTABLE(test_algorithms test_algorithms.cu)
CUDA_ADD_EXECUTABLE(test_array      test_array.cu)
CUDA_ADD_EXECUTABLE(test_ctors      test_ctors.cu)
CUDA_ADD_EXECUTABLE(test_cube       test_cube.cu)
CUDA_ADD_EXECUTABLE(test_iters      test_iters.cu)
CUDA_ADD_EXECUTABLE(test_matrix     test_matrix.cu)
CUDA_ADD_EXECUTABLE(test_models     test_models.cu)
CUDA_ADD_EXECUTABLE(test_ptr        test_ptr.cu)
CUDA_ADD_EXECUTABLE(test_vector     test_vector.cu)

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -pedantic -flto")
LIST(APPEND CUDA_NVCC_FLAGS -arch=sm_21 -O3 -Xptxas -v)

