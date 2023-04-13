#ifndef LLAMA_CC_ENV_H_
#define LLAMA_CC_ENV_H_

struct OrtApi;
struct OrtEnv;
struct OrtMemoryInfo;

#include <mutex>

namespace llama {

// stores all global function and objects required in finley
class Env {
 public:
  class Impl;

  Env(Env &) = delete;
  Env(Env &&) = delete;
  Env &operator=(Env &) = delete;
  Env &operator=(Env &&) = delete;
  ~Env();
  
  // initialize and destryy the global environment
  static void Init() noexcept;
  static void Destroy() noexcept;

  // get an instance of Env. Before calling this function Init() should be
  // called. This function is lock-free
  static const Env *instance() noexcept;

  // environment for onnxruntime
  const OrtApi *ort_api() const noexcept;
  OrtEnv *ort_env() const noexcept;

 private:
  // global pointer of Env as well as its Init() and Destroy() mutex
  static Env *env_;
  static std::mutex mutex_;

  // internal inolementation
  std::unique_ptr<Impl> impl_;

  // singleton constructor
  Env();
};

}  // namespace llama

#endif  // LLAMA_CC_ENV_H_
