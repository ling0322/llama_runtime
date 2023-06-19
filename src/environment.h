#pragma once

#include <memory>
#include <mutex>
#include "util.h"

namespace llama {

namespace nn {
class Operators;
enum class CPUMathBackend;

}  // namespace nn

// stores all global function and objects required in finley
class Environment : private util::NonCopyable {
 public:
  class Impl;
  
  // initialize and destryy the global environment
  static void init();
  static void destroy();

  // get the best backend implementation of LLmRT-BLAS.
  static nn::CPUMathBackend getCpuMathBackend();

  // get or set the num-threads for CPU math operators.
  static int getCpuMathNumThreads();
  static void setCpuMathNumThreads(int numThreads);
    
 private:
  // global pointer of Env as well as its Init() and Destroy() mutex
  static Impl *_instance;
};

}  // namespace llama
