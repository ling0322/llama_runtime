#include "util.h"

#include "strings.h"



namespace llama {
namespace util {

//
// class Path
//

#ifdef AL_PLATFORM_WINDOWS
#define PATH_DELIM "\\"
#else
#define PATH_DELIM "/"
#endif

Path::Path(const std::string &path) : _path(normPath(path)) {}
Path::Path(std::string &&path): _path(normPath(path)) {}

Path Path::dirname() const {
  int lastDelimIdx = _path.find_last_of(PATH_DELIM);
  if (lastDelimIdx == std::string::npos) {
    return "";
  }

  std::string name = std::string(_path.begin(), _path.begin() + lastDelimIdx);
  name = str::trimRight(name, PATH_DELIM);
  return name;
}

Path Path::basename() const {
  int lastDelimIdx = _path.find_last_of(PATH_DELIM);
  if (lastDelimIdx == std::string::npos) {
    return _path;
  }

  return std::string(_path.begin() + lastDelimIdx + 1, _path.end());
}

bool Path::operator==(const Path &r) const {
  return _path == r._path;
}

bool Path::operator==(const std::string &r) const {
  return _path == r;
}

Path Path::operator/(const Path &path) const {
  std::string left = _path;
  if (left.empty()) {
    return path;
  }

  if ((!left.empty()) && left.back() == PATH_DELIM[0]) {
    left = str::trimRight(left, PATH_DELIM);
  }

  std::string right = path._path;
  if ((!right.empty()) && right.front() == PATH_DELIM[0]) {
    right = str::trimLeft(right, PATH_DELIM);
  }

  return left + PATH_DELIM + right;
}

Path Path::operator/(const std::string &path) const {
  return *this / Path(path);
}

std::string Path::string() const {
  return _path;
}

std::wstring Path::wstring() const {
  return str::toWide(_path);
}

}  // namespace util
}  // namespace llama
