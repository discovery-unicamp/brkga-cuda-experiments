// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#ifndef SRC_BRKGA_OPENCL_HPP
#define SRC_BRKGA_OPENCL_HPP

#include <CL/cl2.hpp>
#include <cassert>

class OpenCL {
public:

  class Pipeline;

  [[nodiscard]]
  Pipeline startPipeline();

  template <class T>
  [[nodiscard]]
  inline cl::Buffer buffer(size_t n) const {
    assert(n > 0);
    return cl::Buffer(context, CL_MEM_READ_WRITE, n * sizeof(T));
  }

  template <class T>
  [[nodiscard]]
  inline cl::Buffer buffer(std::vector<T>& data, bool readOnly = false) const {
    assert(!data.empty());
    return cl::Buffer(context, data.begin(), data.end(), readOnly);
  }

  template <class T>
  inline cl::Event read(const cl::Buffer& src, T* dst, size_t begin, size_t n,
                        const std::vector<cl::Event>& dependencies = {}) const {
    assert(n > 0);
    cl::Event event;
    commands.enqueueReadBuffer(src, true, begin * sizeof(T), n * sizeof(T), dst, &dependencies, &event);
    return event;
  }

  template <class T>
  [[nodiscard]]
  inline std::vector<T> read(const cl::Buffer& src, size_t begin, size_t n,
                             const std::vector<cl::Event>& dependencies = {}) const {
    std::vector<T> dst(n);
    read(src, dst.data(), begin, n, dependencies);
    return dst;
  }

  template <class T>
  [[nodiscard]]
  inline cl::Event copy(cl::Buffer& dst, size_t beginDst, const cl::Buffer& src, size_t beginSrc, size_t n,
                        const std::vector<cl::Event>& dependencies = {}) const {
    cl::Event event;
    commands.enqueueCopyBuffer(src, dst, beginSrc, beginDst, n * sizeof(T), &dependencies, &event);
    return event;
  }

protected:

  OpenCL(const cl::Device& device, const char* source, const char* flags);

  template <class... T>
  [[nodiscard]]
  inline cl::Kernel kernel(const char* name, const T& ... args) const {
    cl::Kernel kernel(program, name);
    setKernelArgs(kernel, 0, args...);
    return kernel;
  }

private:

  cl::Event run(const cl::Kernel& kernel, cl::size_type blocks, cl::size_type threads,
                const std::vector<cl::Event>& dependencies);

  static inline void setKernelArgs(cl::Kernel&, unsigned) {}

  template <class T, class... U>
  static inline void setKernelArgs(cl::Kernel& kernel, unsigned index, const T& nextArg, const U& ... otherArgs) {
    kernel.setArg(index, nextArg);
    setKernelArgs(kernel, index + 1, otherArgs...);
  }

  cl::size_type maxWorkSize;
  cl::Context context;
  cl::Program program;
  cl::CommandQueue commands;
};

class OpenCL::Pipeline {
public:

  explicit Pipeline(OpenCL& _openCl, const std::vector<cl::Event>& _dependencies = {}) :
      openCl(_openCl),
      dependencies(_dependencies) {
  }

  inline Pipeline& then(const cl::Kernel& kernel) {
    executionList.push_back(kernel);
    return *this;
  }

  template <class T>
  inline Pipeline& thenReadFromTo(const cl::Buffer& src, T* dst, size_t begin, size_t n) {
    assert(executionList.empty());
    dependencies = {openCl.read(src, dst, begin, n, dependencies)};
    return *this;
  }

  Pipeline& then(size_t n, const std::function<void(size_t, Pipeline&)>& definition);

  inline Pipeline& run(cl::size_type blocks, cl::size_type threads) {
    for (const auto& kernel: executionList)
      dependencies = {openCl.run(kernel, blocks, threads, dependencies)};
    executionList.clear();
    return *this;
  }

  inline void wait() {
    for (auto& dep: dependencies)
      dep.wait();
    dependencies.clear();
  }

private:

  OpenCL& openCl;
  std::vector<cl::Event> dependencies;
  std::vector<cl::Kernel> executionList;
};

#endif //SRC_BRKGA_OPENCL_HPP
