#include "environment.h"

#include <mutex>
#include <thread>
#include "common.h"
#include "shared_library.h"
#include "log.h"
#include "util.h"
#include "llmrt_blas.h"

namespace llama {

// -- class Environment::Impl ----------

class Environment::Impl {
 public:
  Impl();
  ~Impl();

  nn::LLmRTBlasBackend getLLmRTBlasBackend() const;
  int getLLmRTBlasNumThreads() const;
  void setLLmRTBlasNumThreads(int numThreads);

 private:
  nn::LLmRTBlasBackend _llmrtBlasBackend;
  int _llmrtBlasNumThreads;

  void init();
};

Environment::Impl::Impl() :
    _llmrtBlasBackend(nn::LLmRTBlasBackend::DEFAULT),
    _llmrtBlasNumThreads(8) {
  init();
}
Environment::Impl::~Impl() {}

void Environment::Impl::init() {
  _llmrtBlasBackend = nn::LLmRTBlas::findBestBackend();
}

nn::LLmRTBlasBackend Environment::Impl::getLLmRTBlasBackend() const {
  return _llmrtBlasBackend;
}

int Environment::Impl::getLLmRTBlasNumThreads() const {
  return _llmrtBlasNumThreads;
}
void Environment::Impl::setLLmRTBlasNumThreads(int numThreads) {
  _llmrtBlasNumThreads = numThreads;
}

// -- class Environment::Impl ----------

Environment::Impl *Environment::_instance = nullptr;

void Environment::init() {
  _instance = new Impl();
}

void Environment::destroy() {
  delete _instance;
}

nn::LLmRTBlasBackend Environment::getLLmRTBlasBackend() {
  CHECK(_instance);
  return _instance->getLLmRTBlasBackend();
}

int Environment::getLLmRTBlasNumThreads() {
  CHECK(_instance);
  return _instance->getLLmRTBlasNumThreads();
}

void Environment::setLLmRTBlasNumThreads(int numThreads) {
  _instance->setLLmRTBlasNumThreads(numThreads);
}

}  // namespace llama
