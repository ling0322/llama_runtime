#include "env.h"

#include <mutex>
#include <thread>
#include "common.h"
#include "status.h"
#include "shared_library.h"
#include "log.h"
#include "util.h"
#include "third_party/onnxruntime/onnxruntime_c_api.h"

namespace llama {

// global variable and its once_flag for Env
Env *gEnv = nullptr;
std::once_flag gEnvOnceFlag;

class Env::Impl {
 public:
  typedef const  OrtApiBase* (ORT_API_CALL GetApiBaseFunc)();

  Impl();
  ~Impl();
  
  // initialize the environment, this function designed to be call only once
  void Init() noexcept;

  const OrtApi *ort_api() const noexcept { return ort_api_; }
  OrtEnv *ort_env() const noexcept { return ort_env_; }

 private:
  SharedLibrary ort_library_;
  const OrtApi *ort_api_;
  OrtEnv *ort_env_;

  Status InitOnnxRuntime();
};

// ----------------------------------------------------------------------------
// Env::Impl
// ----------------------------------------------------------------------------

Env::Impl::Impl()
    : ort_api_(nullptr),
      ort_env_(nullptr) {}

Env::Impl::~Impl() {
  if (ort_env_) {
    LL_CHECK(ort_api_) << "call ReleaseEnv() on NULL ort_api_";
    ort_api_->ReleaseEnv(ort_env_);
    ort_env_ = nullptr;
  }
}

void Env::Impl::Init() noexcept {
  Status status = InitOnnxRuntime();
  if (!status.ok()) {
    LL_LOG_INFO() << "onnxruntime not initialized: " << status.what();
  }
}

Status Env::Impl::InitOnnxRuntime() {
  RETURN_IF_ERROR(ort_library_.Open("onnxruntime"));

  GetApiBaseFunc *get_apibase = ort_library_.GetFunc<GetApiBaseFunc>(
      "OrtGetApiBase");
  if (!get_apibase) {
    RETURN_ABORTED() << "funtion not found: OrtGetApiBase";
  }
  const OrtApiBase *api_base = get_apibase();
  LL_CHECK(api_base);

  // ort_api_
  ort_api_ = api_base->GetApi(ORT_API_VERSION);
  if (!ort_api_) {
    RETURN_ABORTED() << "api_base->GetApi() failed: runtime version "
                     << api_base->GetVersionString() << " request version "
                     << ORT_API_VERSION;
  }

  // ort_env_
  AutoCPtr<OrtEnv> ort_env = {nullptr, ort_api_->ReleaseEnv};
  OrtStatus *status = ort_api_->CreateEnv(
      ORT_LOGGING_LEVEL_WARNING,
      PROJECT_NAME,
      ort_env.get_pp());
  if (status) {
    std::string message = ort_api_->GetErrorMessage(status);
    ort_api_->ReleaseStatus(status);
    RETURN_ABORTED() << "create OrtEnv failed: " << message;
  }

  LL_CHECK(ort_env.get());
  ort_env_ = ort_env.Release();

  return OkStatus();
}

// ----------------------------------------------------------------------------
// Env
// ----------------------------------------------------------------------------

Env *Env::env_ = nullptr;
std::mutex Env::mutex_;

Env::Env() {}
Env::~Env() {}

void Env::Init() noexcept {
  std::lock_guard<std::mutex> guard(mutex_);
  if (env_) {
    return;
  }

  env_ = new Env();
  env_->impl_ = std::make_unique<Impl>();
  env_->impl_->Init();
}

void Env::Destroy() noexcept {
  std::lock_guard<std::mutex> guard(mutex_);

  delete env_;
  env_ = nullptr;
}

const Env *Env::instance() noexcept {
  LL_CHECK(env_) << "Env::instance() called before Init() or after Destroy()";
  return env_;
}

const OrtApi *Env::ort_api() const noexcept {
  return impl_->ort_api();
}
OrtEnv *Env::ort_env() const noexcept {
  return impl_->ort_env();
}

}  // namespace llama
