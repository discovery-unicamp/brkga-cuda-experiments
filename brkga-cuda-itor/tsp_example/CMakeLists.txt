include_directories(../brkga_cuda_api)
include_directories(src)

add_executable(brkga-tsp
  src/main.cpp
  src/TSPInstance.cpp
  src/Decoder.cu
  ${API_FILES}
)

set_target_properties(brkga-tsp PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-tsp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-tsp PROPERTY CUDA_STANDARD 11)
target_compile_options(brkga-tsp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -Xcompiler
                       -fopenmp
                       >)

if(OpenMP_CXX_FOUND)
  target_link_libraries(brkga-tsp PUBLIC OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)
else()
	target_link_libraries(brkga-tsp PUBLIC -lcurand)
endif()
