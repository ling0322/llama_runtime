#include "util.h"

#include <windows.h>
#include "common.h"
#include "log.h"
#include "strings.h"

namespace llama {
namespace util {

Path Path::CurrentModulePath() {
  char filename[MAX_PATH + 1];

  HMODULE hm = NULL;
  BOOL b = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | 
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCWSTR)&CurrentModulePath,
                              &hm);
  CHECK(b);

  DWORD dw = GetModuleFileNameA(hm, filename, sizeof(filename));
  CHECK(dw != 0);

  return filename;
}

Path Path::CurrentExecutablePath() {
  char filename[MAX_PATH + 1];
  DWORD charsWritten = GetModuleFileNameA(NULL, filename, sizeof(filename));
  CHECK(charsWritten);

  return filename;
}

bool Path::isabs() const {
  if (path_.size() <= 1) return false;

  std::string path = strings::Trim(path_);
  char disk = tolower(path.front());
  if (disk > 'z' || disk < 'a') {
    return false;
  }

  if (path[1] == ':') {
    return true;
  }

  return false;
}

std::string Path::NormPath(const std::string &path) {
  return strings::Replace(path, "/", "\\");
}

}  // namespace util
}  // namespace llama
