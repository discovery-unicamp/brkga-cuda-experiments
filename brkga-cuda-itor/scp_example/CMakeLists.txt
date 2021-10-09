include_directories(../brkga_cuda_api)
include_directories(src)

add_executable(brkga-scp
  src/main.cpp
        src/SetCoveringDecoder.cu
  src/Node.cpp
  src/SetCoveringDecoder.cpp
  src/SetCoveringSolution.cpp
  ${API_FILES}
)

set_target_properties(brkga-scp PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-scp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-scp PROPERTY CUDA_STANDARD 17)
set_property(TARGET brkga-scp PROPERTY CUDA_ARCHITECTURES 61)
target_compile_options(brkga-scp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -Xcompiler
                       -fopenmp
                       >)

if(OpenMP_CXX_FOUND)
  if(USE_NVTX)
    target_link_libraries(brkga-scp PUBLIC OpenMP::OpenMP_CXX PUBLIC ${LIBNVTOOLSEXT} -lcurand -fopenmp)
  else()
    target_link_libraries(brkga-scp PUBLIC OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)
  endif()
else()
	target_link_libraries(brkga-scp PUBLIC -lcurand)
endif()
