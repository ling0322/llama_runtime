#ifndef CHIWAUKUM_UTIL_READABLE_FILE_H_
#define CHIWAUKUM_UTIL_READABLE_FILE_H_

#include <functional>
#include <memory>

#include "util.h"

namespace llama {

// Dynamic library loader. It stores the instance of dynamic library and 
// supports get function from it.
//
// Example:
//
// auto library = SharedLibrary::open("foo");  // foo.dll for Windows
// std::function<int(float)> func = library.GetFunc<int(float)>("bar");
class SharedLibrary : private util::NonCopyable {
 public:
  class Impl;

  ~SharedLibrary();

  // load a library by name from OS. Firstly, it will search the same directory
  // as caller module. Then, fallback to system search. In windows, the actual
  // library name would be `name`.dll. In Linux, it would be lib`name`.so
  static std::unique_ptr<SharedLibrary> open(const std::string &name);

  // get function by name. return nullptr if function not found
  template<typename T>
  T *getFunc(const std::string& name) {
    return reinterpret_cast<T *>(getFuncPtr(name));
  }

 private:
  std::unique_ptr<Impl> _impl;

  SharedLibrary();

  // get raw function pointer by name. throw error if function not exist or
  // other errors occured
  void *getFuncPtr(const std::string& name);
};

}  // namespace llama


#endif  // CHIWAUKUM_UTIL_READABLE_FILE_H_
