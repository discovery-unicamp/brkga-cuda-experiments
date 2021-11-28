#include "OpenCL.hpp"

OpenCL::Pipeline OpenCL::startPipeline() {
  return Pipeline(*this);
}

OpenCL::OpenCL(const cl::Device& device, const char* source, const char* flags) :
    maxWorkSize(-1),
    context(device),
    program(context, source),
    commands(context, device) {
  device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkSize);

  try {
    program.build({device}, flags);
  } catch (const cl::BuildError& err) {
    std::string errors;
    for (const auto& log: err.getBuildLog())
      errors += log.second;
    throw std::runtime_error("Build failed\n" + errors);
  }
}

cl::Event OpenCL::run(const cl::Kernel& kernel, cl::size_type blocks, cl::size_type threads,
                      const std::vector<cl::Event>& dependencies) {
  assert(0 < threads && threads <= maxWorkSize);
  // TODO is this really necessary?
  --threads;
  const auto global = threads - threads % blocks + blocks;  // next multiple of blocks
  const auto local = blocks;
  cl::Event event;
  commands.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(local), &dependencies, &event);
  return event;
}

OpenCL::Pipeline& OpenCL::Pipeline::then(size_t n, const std::function<void(size_t, Pipeline&)>& definition) {
  assert(executionList.empty());
  std::vector<cl::Event> newDependencies;
  for (size_t i = 0; i < n; ++i) {
    Pipeline pipeline(openCl, dependencies);
    definition(i, pipeline);
    assert(pipeline.executionList.empty());
    newDependencies.insert(newDependencies.end(), pipeline.dependencies.begin(),  pipeline.dependencies.end());
  }

  dependencies = std::move(newDependencies);
  return *this;
}
