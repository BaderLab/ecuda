SHELL = /bin/sh

packagename = ecuda

prefix = /usr/local
exec_prefix = $(prefix)
bindir = $(exec_prefix)/bin
sbindir = $(exec_prefix)/sbin
libexecdir = $(exec_prefix)/libexec
datarootdir = $(exec_prefix)/share
datadir = $(datarootdir)
sysconfdir = $(prefix)/etc
sharedstatedir = $(prefix)/com
localstatedir = $(prefix)/var
runstatedir = $(localstatedir)/run
includedir = $(prefix)/include
oldincludedir = /usr/include
docdir = $(datarootdir)/doc/$(packagename)
infodir = $(datarootdir)/info
htmldir = $(docdir)
dvidir = $(docdir)
pdfdir = $(docdir)
psdir = $(docdir)
libbdir = $(exec_prefix)/lib
lispdir = $(datarootdir)/emacs/site-lisp
localedir = $(datarootdir)/locale
mandir = $(datarootdir)/man
man1dir = $(mandir)/man1
man2dir = $(mandir)/man2
manext = .1
man1ext = .1
man2ext = .2
#srcdir = ./src

AR = ar
CC = gcc -x c
CFLAGS = -Wall
CXX = g++
CXXFLAGS = -Wall -flto -L/usr/local/cuda/lib64 -pedantic
CXXFLAGS += -I../Catch/include
#-std=c++11 
FC = gfortran
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -arch=sm_21 -O3
NVCCFLAGS += -Xptxas -v
NVCCFLAGS += -I../Catch/include
# for OpenMP support add to above: -X compiler -fopenmp
# for C++11 support add to above: -std=c++11
LDLIBS = -lcudart

-include local-config.cfg

ifeq ($(std),c++11)
	CXXFLAGS += -std=c++11
	NVCCFLAGS += -std=c++11
endif

ifeq ($(mode),debug)
	CXXFLAGS += -g
	CFLAGS += -g
else
	mode = release
	CXXFLAGS += -O3 -march=native
	CFLAGS += -O3 -march=native
endif

install:
	@mkdir -p $(includedir)
	cp -a include/ecuda $(includedir)/ecuda

test/% :: test/%.cu
	@mkdir -p bin/test
	@mkdir -p obj/test
	$(NVCC) $(NVCCFLAGS) -c $< -o obj/$@.cu.o
	$(CXX) $(CXXFLAGS) obj/$@.cu.o $(LDLIBS) -o bin/$@

test/cpu/% :: test/%.cu
	@mkdir -p bin/test/cpu
	cp $< $<.cpp
	$(CXX) -g $(CXXFLAGS) -Wno-variadic-macros -Wno-long-long $<.cpp -o bin/$@
	rm $<.cpp

test/ptx/% :: test/%.cu
	@mkdir -p ptx/test
	$(NVCC) -ptx $< -o ptx/$<.ptx

T_FILES = $(basename $(shell find t -name '*.cu'))

t/% :: t/%.cu
	@mkdir -p bin/t
	@mkdir -p obj/t
	$(NVCC) $(NVCCFLAGS) -c $< -o obj/$@.cu.o
	$(CXX) $(CXXFLAGS) obj/$@.cu.o $(LDLIBS) -o bin/$@
	$(NVCC) $(NVCCFLAGS) -std=c++11 -c $< -o obj/$@_c++11.cu.o
	$(CXX) $(CXXFLAGS) -std=c++11 obj/$@.cu.o $(LDLIBS) -o bin/$@_c++11

t/cpu/% :: t/%.cu
	@mkdir -p bin/t/cpu
	cp $< $<.cpp
	$(CXX) -g $(CXXFLAGS) -Wno-variadic-macros -Wno-long-long $<.cpp -o bin/$@
	rm $<.cpp

unittests :: $(T_FILES)
.PHONY: unittests

benchmark/% :: benchmark/%.cu
	@mkdir -p bin/benchmark
	@mkdir -p obj/benchmark
	$(NVCC) $(NVCCFLAGS) -c $< -o obj/$@.cu.o
	$(CXX) $(CXXFLAGS) obj/$@.cu.o $(LDLIBS) -o bin/$@

benchmark/ptx/% :: benchmark/%.cu
	@mkdir -p ptx/benchmark
	$(NVCC) -ptx $< -o ptx/$<.ptx

dist:
	tar zcvf ecuda-dist.tar.gz LICENSE.txt include/ecuda/*.hpp test/*.cu benchmarks/*.cu

docs: FORCE
	doxygen doxygen.cfg

tests: test/test_array test/test_cube test/test_matrix test/test_vector

cpu_tests: test/cpu/test_array test/cpu/test_cube test/cpu/test_matrix test/cpu/test_vector

benchmarks: benchmarks/array benchmarks/matrix

FORCE:

