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
CXXFLAGS = -O3 -Wall -flto -L/usr/local/cuda/lib64 -pedantic
# for C++11 support append to above: -std=c++11
FC = gfortran
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -arch=sm_21 -O3
# for OpenMP support append to above: -X compiler -fopenmp
# for CUDA 7.0 C++11 support append to above: -std=c++11
LDLIBS = -lcudart

-include local-config.cfg

ifeq ($(mode),debug)
	CXXFLAGS += -g
	CFLAGS += -g
else
	mode = release
	CXXFLAGS += -O3 -march=native
	CFLAGS += -O3 -march=native
endif

VERSION = $(shell cat VERSION)

install:
	@mkdir -p $(includedir)
	cp -a include/ecuda $(includedir)/ecuda

test/% :: test/%.cu
	@mkdir -p bin/test
	@mkdir -p obj/test
	$(NVCC) $(NVCCFLAGS) -c $< -o obj/$@.cu.o
	$(CXX) $(CXXFLAGS) obj/$@.cu.o $(LDLIBS) -o bin/$@

benchmarks/% :: benchmarks/%.cu
	@mkdir -p bin/benchmarks
	@mkdir -p obj/benchmarks
	$(NVCC) $(NVCCFLAGS) -c $< -o obj/$@.cu.o
	$(CXX) $(CXXFLAGS) obj/$@.cu.o $(LDLIBS) -o bin/$@

examples/% :: examples/%.cu
	@mkdir -p bin/examples
	@mkdir -p obj/examples
	$(NVCC) $(NVCCFLAGS) -c $< -o obj/$@.cu.o
	$(CXX) $(CXXFLAGS) obj/$@.cu.o $(LDLIBS) -o bin/$@

dist:
	tar zcvf ecuda-dist-${VERSION}.tar.gz LICENSE.txt Makefile MANIFEST doxygen.cfg include/ecuda/*.hpp test/*.cu benchmarks/*.cu examples/*.cu docs/*.html docs/*.css docs/*.dox docs/images/*.png

docs: FORCE
	doxygen doxygen.cfg

tests: test/test_array test/test_cube test/test_matrix test/test_vector

benchmarks: benchmarks/array benchmarks/matrix

FORCE:

