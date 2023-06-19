#include "environment.h"

#include <mutex>
#include <thread>
#include "common.h"
#include "shared_library.h"
#include "log.h"
#include "util.h"
#include "gemm_kernel.h"

namespace llama {

// -- class Environment::Impl ----------

class Environment::Impl {
 public:
  Impl();
  ~Impl();

  nn::CPUMathBackend getCpuMathBackend() const;
  int getCpuMathNumThreads() const;
  void setCpuMathNumThreads(int numThreads);

 private:
  nn::CPUMathBackend _cpuMathBackend;
  int _cpuMathNumThreads;

  void init();
};

Environment::Impl::Impl() :
    _cpuMathBackend(nn::CPUMathBackend::DEFAULT),
    _cpuMathNumThreads(8) {
  init();
}
Environment::Impl::~Impl() {}

void Environment::Impl::init() {
  _cpuMathBackend = nn::findBestCpuMathBackend();
}

nn::CPUMathBackend Environment::Impl::getCpuMathBackend() const {
  return _cpuMathBackend;
}

int Environment::Impl::getCpuMathNumThreads() const {
  return _cpuMathNumThreads;
}
void Environment::Impl::setCpuMathNumThreads(int numThreads) {
  _cpuMathNumThreads = numThreads;
}

// -- class Environment::Impl ----------

Environment::Impl *Environment::_instance = nullptr;

void Environment::init() {
  _instance = new Impl();
}

void Environment::destroy() {
  delete _instance;
}

nn::CPUMathBackend Environment::getCpuMathBackend() {
  CHECK(_instance);
  return _instance->getCpuMathBackend();
}

int Environment::getCpuMathNumThreads() {
  CHECK(_instance);
  return _instance->getCpuMathNumThreads();
}

void Environment::setCpuMathNumThreads(int numThreads) {
  _instance->setCpuMathNumThreads(numThreads);
}

}  // namespace llama
