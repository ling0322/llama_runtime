#include "shared_library.h"

#include <functional>
#include <memory>
#include <windows.h>

#include "common.h"
#include "strings.h"
#include "util.h"

namespace llama {

class SharedLibrary::Impl {
 public:
  ~Impl();

  static std::unique_ptr<Impl> open(const std::string &name);
  void *getFuncPtr(const std::string &filename);

 private:
  HMODULE _module;

  Impl();
};

SharedLibrary::Impl::Impl() : _module(nullptr) {}
SharedLibrary::Impl::~Impl() {
  if (_module) {
    FreeLibrary(_module);
    _module = nullptr;
  }
}

std::unique_ptr<SharedLibrary::Impl> SharedLibrary::Impl::open(const std::string &name) {
  std::unique_ptr<Impl> impl{new Impl()};
  util::Path filename = std::string(name) + ".dll";

  // first try to load the dll from same folder as current module
  util::Path modulePath = util::Path::currentModulePath();
  modulePath = modulePath.dirname();
  util::Path absFilename = modulePath / filename;

  DWORD code =  S_OK;
  impl->_module = LoadLibraryW(absFilename.wstring().c_str());
  if (!impl->_module) {
    code = GetLastError();
    LOG(INFO) << "Load library " << absFilename.string()
              << " failed with code " << code
              << " fall back to system search.";

    // fallback to system search
    impl->_module = LoadLibraryW(filename.wstring().c_str());
    if (!impl->_module) {
      code = GetLastError();
      throw AbortedException(str::sprintf(
          "Load library %s failed with code 0x%x.", absFilename.string(), code));
    }
  }

  return impl;
}

void *SharedLibrary::Impl::getFuncPtr(const std::string &func_name) {
  CHECK(_module) << "call GetRawFuncPtr() on empty SharedLibrary";
  FARPROC func = GetProcAddress(_module, std::string(func_name).c_str());
  return reinterpret_cast<void *>(func);
}

// -- class SharedLibrary ----------

SharedLibrary::SharedLibrary() {}
SharedLibrary::~SharedLibrary() {}

std::unique_ptr<SharedLibrary> SharedLibrary::open(const std::string &name) {
  std::unique_ptr<SharedLibrary> library{new SharedLibrary()};
  library->_impl = Impl::open(name);
  return library;
}
void *SharedLibrary::getFuncPtr(const std::string &name) {
  return _impl->getFuncPtr(name);
}

}  // namespace llama
