#ifndef CHIWAUKUM_UTIL_READABLE_FILE_H_
#define CHIWAUKUM_UTIL_READABLE_FILE_H_

#include <functional>
#include <memory>

#include "status.h"

namespace llama {

// Dynamic library loader. It stores the instance of dynamic library and 
// supports get function from it.
//
// Example:
//
// SharedLibrary library("foo");  // foo.dll for Windows
// std::function<int(float)> func = library.GetFunc<int(float)>("bar");
class SharedLibrary {
 public:
  class Impl;

  SharedLibrary();
  SharedLibrary(SharedLibrary &) = delete;
  SharedLibrary(SharedLibrary &&) = delete;
  SharedLibrary &operator=(SharedLibrary &) = delete;
  SharedLibrary &operator=(SharedLibrary &&) = delete;
  ~SharedLibrary();

  // load a library by name from OS. Firstly, it will search the same directory
  // as caller module. Then, fallback to system search. In windows, the actual
  // library name would be `name`.dll. In Linux, it would be lib`name`.so
  Status Open(const std::string &name);

  // get raw function pointer by name. throw error if function not exist or
  // other errors occured
  void *GetRawFuncPtr(const std::string& name);

  // get function by name. return nullptr if function not found
  template<typename T>
  T *GetFunc(const std::string& name) {
    return reinterpret_cast<T *>(GetRawFuncPtr(name));
  }

  // returns true if the shared library is loaded
  bool empty() const;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace llama


#endif  // CHIWAUKUM_UTIL_READABLE_FILE_H_
