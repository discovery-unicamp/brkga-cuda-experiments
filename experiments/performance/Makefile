mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_path := $(patsubst %/,%,$(dir $(mkfile_path)))
cuda_version := $(shell nvidia-smi | grep "CUDA Version" | cut -d: -f 3 | cut -d" " -f 2)

EXPERIMENT = echo Starting container... && docker run --rm --gpus device=0 -v $(current_path)/:/performance/ nvidia/cuda:${cuda_version}.0-devel-ubuntu20.04 /bin/bash -c
NVCC = nvcc --generate-code=arch=compute_61,code=[compute_61,sm_61] -Xcompiler=-fPIC -O3 -std=c++17
G++ = g++ -O2 -std=c++17


sort-array:
	@$(EXPERIMENT) "cd /performance && make docker-sort-array"

docker-sort-array: sort-array.out
	./sort-array.out

sort-array.out: sort-array.cu
	$(NVCC) -o sort-array.out sort-array.cu


sort-array-cpu: sort-array.cpp.out
	time ./sort-array.cpp.out

sort-array.cpp.out: sort-array.cpp
	$(G++) -o sort-array.cpp.out sort-array.cpp


bitonic-sort-general:
	@$(EXPERIMENT) "cd /performance && make docker-bitonic-sort-general"

docker-bitonic-sort-general: bitonic-sort-general.out
	./bitonic-sort-general.out

bitonic-sort-general.out: bitonic-sort-general.cu
	$(NVCC) -o bitonic-sort-general.out bitonic-sort-general.cu
