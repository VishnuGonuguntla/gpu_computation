# Build serial and parallel binaries for every project from the repo root.
#
#   make              # all serial + parallel
#   make serial       # CPU binaries
#   make parallel     # OpenMP + CUDA binaries
#   make stream / julia / molDyn / molDynOpt / dem / mppi

CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -O3 -march=native -std=c++20
C_FLAGS ?= -O3 -march=native -std=c++20
NVCCFLAGS ?= -O3 -std=c++20 -arch=sm_80
CUDA_FLAGS ?= -O3 -std=c++20 -arch=sm_80
export CXX NVCC CXXFLAGS C_FLAGS NVCCFLAGS CUDA_FLAGS

.PHONY: all serial parallel clean \
	stream julia molDyn molDynOpt dem mppi \
	stream-serial stream-parallel \
	julia-serial julia-parallel \
	molDyn-serial molDyn-parallel \
	molDynOpt-serial molDynOpt-parallel \
	dem-serial dem-parallel \
	mppi-serial mppi-parallel

all: serial parallel

serial: stream-serial julia-serial molDyn-serial molDynOpt-serial dem-serial mppi-serial

parallel: stream-parallel julia-parallel molDyn-parallel molDynOpt-parallel dem-parallel mppi-parallel

stream: stream-serial stream-parallel
julia: julia-serial julia-parallel
molDyn: molDyn-serial molDyn-parallel
molDynOpt: molDynOpt-serial molDynOpt-parallel
dem: dem-serial dem-parallel
mppi: mppi-serial mppi-parallel

stream-serial:
	$(MAKE) -C 1_1_stream_benchmark build-serial

stream-parallel:
	$(MAKE) -C 1_1_stream_benchmark build-parallel

julia-serial:
	$(MAKE) -C 1_2_julia_set build-serial

julia-parallel:
	$(MAKE) -C 1_2_julia_set build-parallel

molDyn-serial:
	$(MAKE) -C 2_molecularDyn build-serial

molDyn-parallel:
	$(MAKE) -C 2_molecularDyn build-parallel

molDynOpt-serial:
	$(MAKE) -C 3_molecularDynOpt build-serial

molDynOpt-parallel:
	$(MAKE) -C 3_molecularDynOpt build-parallel

dem-serial:
	$(MAKE) -C 4_DEM build-serial

dem-parallel:
	$(MAKE) -C 4_DEM build-parallel

mppi-serial:
	$(MAKE) -C mini_project build-serial

mppi-parallel:
	$(MAKE) -C mini_project build-parallel

clean:
	$(MAKE) -C 1_1_stream_benchmark clean
	$(MAKE) -C 1_2_julia_set clean
	$(MAKE) -C 2_molecularDyn clean
	$(MAKE) -C 3_molecularDynOpt clean
	$(MAKE) -C 4_DEM clean
	$(MAKE) -C mini_project clean
