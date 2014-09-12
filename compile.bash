/usr/local/cuda/bin/nvcc -arch=sm_21 -Xcompiler -fopenmp -c test/test_array.cu -o obj/test/test_array.cu.o
g++ -O3 -flto -march=native -fopenmp -L/usr/local/cuda/lib64 obj/test/test_array.cu.o -lcudart -o bin/test/test_array
