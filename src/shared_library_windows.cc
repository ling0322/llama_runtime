#include "shared_library.h"

#include <functional>
#include <memory>
#include <windows.h>

#include "common.h"
#include "status.h"
#include "util.h"

namespace llama {

class SharedLibrary::Impl {
 public:
  Impl();
  ~Impl();

  // implements SharedLibrary
  Status Open(const std::string &name);
  void *GetRawFuncPtr(const std::string &filename);

 private:
  HMODULE module_;
};

SharedLibrary::Impl::Impl() : module_(nullptr) {}
SharedLibrary::Impl::~Impl() {
  if (module_) {
    FreeLibrary(module_);
    module_ = nullptr;
  }
}

Status SharedLibrary::Impl::Open(const std::string &name) {
  util::Path filename = std::string(name) + ".dll";

  // first try to load the dll from same folder as current module
  util::Path module_path = util::Path::CurrentModulePath();
  module_path = module_path.dirname();
  util::Path abs_filename = module_path / filename;

  DWORD code =  S_OK;
  std::wstring ws_filename;
  RETURN_IF_ERROR(abs_filename.AsWString(&ws_filename)) 
      << "invalid library name: " << name;
  module_ = LoadLibraryW(ws_filename.c_str());
  if (!module_) {
    code = GetLastError();
    LOG(INFO) << "Load library " << abs_filename.string()
              << " failed with code " << code
              << " fall back to system search";

    // fallback to system search
    RETURN_IF_ERROR(filename.AsWString(&ws_filename))
        << "invalid library name: " << name ;
    module_ = LoadLibraryW(ws_filename.c_str());
    if (!module_) {
      code = GetLastError();
      RETURN_ABORTED() << "Load library " << abs_filename.string()
                       << " failed with code " << code;
    }
  }

  return OkStatus();
}

void *SharedLibrary::Impl::GetRawFuncPtr(const std::string &func_name) {
  CHECK(module_) << "call GetRawFuncPtr() on empty SharedLibrary";
  FARPROC func = GetProcAddress(module_, std::string(func_name).c_str());
  return reinterpret_cast<void *>(func);
}

// ----------------------------------------------------------------------------
// SharedLibrary
// ----------------------------------------------------------------------------

SharedLibrary::SharedLibrary() {}
SharedLibrary::~SharedLibrary() {}

Status SharedLibrary::Open(const std::string &name) {
  impl_ = std::make_unique<Impl>();
  return impl_->Open(name);
}
void *SharedLibrary::GetRawFuncPtr(const std::string &name) {
  return impl_->GetRawFuncPtr(name);
}

}  // namespace llama
