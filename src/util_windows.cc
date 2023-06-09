#include "util.h"

#include <windows.h>
#include "common.h"
#include "log.h"
#include "strings.h"

namespace llama {
namespace util {

Path Path::currentModulePath() {
  char filename[MAX_PATH + 1];

  HMODULE hm = NULL;
  BOOL b = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | 
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCWSTR)&currentModulePath,
                              &hm);
  CHECK(b);

  DWORD dw = GetModuleFileNameA(hm, filename, sizeof(filename));
  CHECK(dw != 0);

  return filename;
}

Path Path::currentExecutablePath() {
  char filename[MAX_PATH + 1];
  DWORD charsWritten = GetModuleFileNameA(NULL, filename, sizeof(filename));
  CHECK(charsWritten);

  return filename;
}

bool Path::isabs() const {
  if (_path.size() <= 1) return false;

  std::string path = str::trim(_path);
  char disk = tolower(path.front());
  if (disk > 'z' || disk < 'a') {
    return false;
  }

  if (path[1] == ':') {
    return true;
  }

  return false;
}

std::string Path::normPath(const std::string &path) {
  return str::replace(path, "/", "\\");
}

}  // namespace util
}  // namespace llama
