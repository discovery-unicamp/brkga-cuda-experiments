#ifndef COMMON_UTILS_FUNCTOR_CUH
#define COMMON_UTILS_FUNCTOR_CUH

#include "../CudaCheck.cuh"

namespace device {
namespace detail {
template <class FunctorImpl, class... Args>
__global__ void newFunctor(FunctorImpl** functor, Args... args) {
  *functor = new FunctorImpl(args...);
}

template <class FunctorImpl>
__global__ void deleteFunctor(FunctorImpl** functor) {
  delete *functor;
}
}  // namespace detail

template <class... Args>
class Functor {
public:
  __device__ Functor() {}
  __device__ virtual ~Functor() {}
  __device__ virtual void operator()(Args... args) = 0;
};

template <class FunctorImpl>
class FunctorPointer {
public:
  template <class... BuildArgs>
  FunctorPointer(BuildArgs... args) : functor(nullptr) {
    CUDA_CHECK(cudaMalloc(&functor, sizeof(FunctorImpl**)));
    detail::newFunctor<<<1, 1>>>(functor, args...);
    CUDA_CHECK_LAST();
  }

  ~FunctorPointer() { detail::deleteFunctor<<<1, 1>>>(functor); }

  FunctorImpl** functor;
};
}  // namespace device

#endif  // COMMON_UTILS_FUNCTOR_CUH
