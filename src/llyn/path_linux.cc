#include "llyn/path.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include "llyn/log.h"
#include "llyn/strings.h"

extern char *program_invocation_name;

namespace ly {

Path Path::currentExecutablePath() {
  return Path(program_invocation_name);
}

Path Path::currentModulePath() {
  Dl_info info;
  int success = dladdr(reinterpret_cast<const void *>(&currentModulePath), &info);
  CHECK(success);

  return Path(info.dli_fname);
}

bool Path::isabs() const {
  if (_path.size() == 0) return false;
  if (_path[0] == '/') return true;

  return false;
}

std::string Path::normPath(const std::string &path) {
  return path;
}

}  // namespace ly
