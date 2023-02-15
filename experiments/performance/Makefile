mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_path := $(patsubst %/,%,$(dir $(mkfile_path)))
cuda_version := $(shell nvidia-smi | grep "CUDA Version" | cut -d: -f 3 | cut -d" " -f 2)

EXPERIMENT = echo Starting container... && docker run --rm --gpus device=0 -v $(current_path)/../../:/experiment/ nvidia/cuda:${cuda_version}.0-devel-ubuntu20.04 /bin/bash -c
NVCC = nvcc --expt-relaxed-constexpr --generate-code=arch=compute_61,code=[compute_61,sm_61] -Xcompiler=-fPIC -O2 -std=c++17
G++ = g++ -O2 -std=c++17

################################################################################
sort-array:
	@$(EXPERIMENT) "cd /performance && make docker-sort-array"

docker-sort-array:
	$(NVCC) -o sort-array.out sort-array.cu
	./sort-array.out


################################################################################
sort-array-cpu:
	$(G++) -o sort-array.cpp.out sort-array.cpp
	time ./sort-array.cpp.out


################################################################################
sort-matrix-columns:
	@$(EXPERIMENT) "cd /performance && make docker-sort-matrix-columns"

docker-sort-matrix-columns:
	$(NVCC) -o sort-matrix-columns.out sort-matrix-columns.cu
	./sort-matrix-columns.out


################################################################################
bitonic-sort-general:
	@$(EXPERIMENT) "cd /performance && make docker-bitonic-sort-general"

docker-bitonic-sort-general:
	$(NVCC) -o bitonic-sort-general.out bitonic-sort-general.cu
	./bitonic-sort-general.out


################################################################################
transpose:
	@$(EXPERIMENT) "cd /performance && make docker-transpose"

docker-transpose:
	$(NVCC) -o transpose.out transpose.cu
	./transpose.out


################################################################################
evolve:
	@$(EXPERIMENT) "cd /experiment/experiments/performance/ && make docker-evolve"

docker-evolve:
	$(NVCC) -lcurand -o evolve.out evolve.cu
	./evolve.out


################################################################################
abstract:
	@$(EXPERIMENT) "cd /performance && make docker-abstract"

docker-abstract:
	$(NVCC) -o abstract.out abstract.cu
	./abstract.out


################################################################################
decoder:
	@$(EXPERIMENT) "cd /performance && make docker-decoder"

docker-decoder:
	$(NVCC) -o decoder.out decoder.cu
	./decoder.out
