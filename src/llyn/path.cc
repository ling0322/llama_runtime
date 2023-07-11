#include "llyn/path.h"

#include "llyn/log.h"
#include "llyn/platform.h"
#include "llyn/strings.h"

namespace ly {

Path::Path(const std::string &path) : _path(normPath(path)) {}
Path::Path(std::string &&path): _path(normPath(path)) {}

Path Path::dirname() const {
  const char *delim = getPathDelim();
  int lastDelimIdx = _path.find_last_of(delim);
  if (lastDelimIdx == std::string::npos) {
    return Path();
  }

  std::string name = std::string(_path.begin(), _path.begin() + lastDelimIdx);
  name = trimRight(name, delim);
  return name;
}

Path Path::basename() const {
  const char *delim = getPathDelim();
  int lastDelimIdx = _path.find_last_of(delim);
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
  const char *delim = getPathDelim();
  std::string left = _path;
  if (left.empty()) {
    return path;
  }

  if ((!left.empty()) && left.back() == delim[0]) {
    left = trimRight(left, delim);
  }

  std::string right = path._path;
  if ((!right.empty()) && right.front() == delim[0]) {
    right = trimLeft(right, delim);
  }

  return left + delim + right;
}

Path Path::operator/(const std::string &path) const {
  return *this / Path(path);
}

std::string Path::string() const {
  return _path;
}

std::wstring Path::wstring() const {
  return toWide(_path);
}

}  // namespace ly
