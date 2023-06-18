#pragma once

#include <memory>
#include <mutex>
#include "util.h"

namespace llama {

namespace nn {
class Operators;
enum class LLmRTBlasBackend;

}  // namespace nn

// stores all global function and objects required in finley
class Environment : private util::NonCopyable {
 public:
  class Impl;
  
  // initialize and destryy the global environment
  static void init();
  static void destroy();

  // get the best backend implementation of LLmRT-BLAS.
  static nn::LLmRTBlasBackend getLLmRTBlasBackend();

  // get or set the num-threads for LLmRT-BLAS.
  static int getLLmRTBlasNumThreads();
  static void setLLmRTBlasNumThreads(int numThreads);
  
  static int getBlasNumThreads();
  
 private:
  // global pointer of Env as well as its Init() and Destroy() mutex
  static Impl *_instance;
};

}  // namespace llama
