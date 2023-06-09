#include "environment.h"

#include <mutex>
#include <thread>
#include "common.h"
#include "shared_library.h"
#include "log.h"
#include "util.h"

namespace llama {

// ----------------------------------------------------------------------------
// Env
// ----------------------------------------------------------------------------

Environment *Environment::env_ = nullptr;
std::mutex Environment::mutex_;

Environment::Environment() {}
Environment::~Environment() {}

void Environment::Init() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (env_) {
    return;
  }
}

void Environment::Destroy() {
  std::lock_guard<std::mutex> guard(mutex_);
}

const Environment *Environment::instance() {
  CHECK(env_) << "Env::instance() called before Init() or after Destroy()";
  return env_;
}


}  // namespace llama
