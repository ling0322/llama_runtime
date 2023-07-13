#pragma once

#include <memory>
#include <mutex>
#include "llyn/noncopyable.h"

namespace llmpp {

namespace nn {
class Operators;
enum class CPUMathBackend;

}  // namespace nn

// stores all global function and objects required in finley
class Environment : private ly::NonCopyable {
 public:
  class Impl;
  
  // initialize and destryy the global environment
  static void init();
  static void destroy();
    
 private:
  // global pointer of Env as well as its Init() and Destroy() mutex
  static Impl *_instance;
};

}  // namespace llmpp
