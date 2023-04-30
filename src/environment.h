#ifndef LLAMA_CC_ENV_H_
#define LLAMA_CC_ENV_H_

#include <memory>
#include <mutex>
#include "util.h"

namespace llama {

namespace nn {
class Operators;
}  // namespace nn


// stores all global function and objects required in finley
class Environment : private util::NonCopyable {
 public:
  class Impl;

  // destructor.
  ~Environment();
  
  // initialize and destryy the global environment
  static void Init();
  static void Destroy();

  // get an instance of Env. Before calling this function Init() should be
  // called. This function is lock-free
  static const Environment *instance();
  
 private:
  // global pointer of Env as well as its Init() and Destroy() mutex
  static Environment *env_;
  static std::mutex mutex_;

  // singleton constructor
  Environment();
};

}  // namespace llama

#endif  // LLAMA_CC_ENV_H_
