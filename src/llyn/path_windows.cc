#include "llyn/path.h"

#include <windows.h>
#include "llyn/log.h"
#include "llyn/strings.h"

namespace ly {

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

  std::string path = trim(_path);
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
  return replace(path, "/", "\\");
}

}  // namespace ly
