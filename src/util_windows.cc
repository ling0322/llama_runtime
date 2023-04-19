#include "util.h"

#include <windows.h>
#include "common.h"
#include "log.h"

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

}  // namespace util
}  // namespace llama
