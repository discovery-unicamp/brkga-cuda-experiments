PROJECT_NAME = cuda-tsp

NVCC = nvcc

SOURCE_DIR = src
BUILD_DIR = build

GPU_FLAGS = --device-c  -Xcompiler -fopenmp -O3 
NVCCFLAGS = -O3 -lcurand -std=c++11 -Xcompiler -fopenmp
#GPU_ARCH = -gencode arch=compute_61,code=sm_61

GPU_FLAGS_DEBUG = --device-c 
NVCCFLAGS_DEBUG  = -g -lcurand -std=c++11 -Xcompiler -fopenmp

gpu_objects = BRKGA.o Decoder.o 

all: build clean

build: build_dir gpu
	$(NVCC) $(gpu_objects) $(SOURCE_DIR)/*.cpp $(NVCCFLAGS) $(GPU_ARCH) -o $(BUILD_DIR)/$(PROJECT_NAME)

build_dir:
	mkdir -p $(BUILD_DIR)

gpu:
	$(NVCC) $(GPU_FLAGS) $(SOURCE_DIR)/*.cu

clean:
	rm *.o

run:
	./$(BUILD_DIR)/$(PROJECT_NAME)

debug: build_debug clean

build_debug: build_dir_debug gpu_debug
	$(NVCC) $(gpu_objects) $(SOURCE_DIR)/*.cpp $(NVCCFLAGS_DEBUG) -o $(BUILD_DIR)/$(PROJECT_NAME)

build_dir_debug:
	mkdir -p $(BUILD_DIR)

gpu_debug:
	$(NVCC) $(GPU_FLAGS_DEBUG) $(SOURCE_DIR)/*.cu
