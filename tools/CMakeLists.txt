CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ecuda)

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

#INCLUDE_DIRECTORIES(/usr/local/cuda/include)

#FILE(GLOB SOURCES "*.cu" "*.cpp" "*.hpp")
#FILE(GLOB SOURCES "*.cu")
CUDA_ADD_EXECUTABLE(print_device_info print_device_info.cu)

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -pedantic -flto")
LIST(APPEND CUDA_NVCC_FLAGS -arch=sm_21 -O3 -Xptxas -v)

#TARGET_LINK_LIBRARIES(print_device_info /usr/local/cuda/lib64/libcudart.so)
