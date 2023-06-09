#include "util.h"

#include "strings.h"



namespace llama {
namespace util {

//
// class Path
//

#ifdef LL_PLATFORM_WINDOWS
#define PATH_DELIM "\\"
#else
#define PATH_DELIM "/"
#endif

Path::Path(const std::string &path) : path_(NormPath(path)) {}
Path::Path(std::string &&path): path_(NormPath(path)) {}

Path Path::dirname() const {
  int last_delim_idx = path_.find_last_of(PATH_DELIM);
  if (last_delim_idx == std::string::npos) {
    return "";
  }

  std::string name = std::string(path_.begin(), path_.begin() + last_delim_idx);
  name = str::trimRight(name, PATH_DELIM);
  return name;
}

Path Path::basename() const {
  int last_delim_idx = path_.find_last_of(PATH_DELIM);
  if (last_delim_idx == std::string::npos) {
    return path_;
  }

  return std::string(path_.begin() + last_delim_idx + 1, path_.end());
}

bool Path::operator==(const Path &r) const {
  return path_ == r.path_;
}

bool Path::operator==(const std::string &r) const {
  return path_ == r;
}

Path Path::operator/(const Path &path) const {
  std::string left = path_;
  if (left.empty()) {
    return path;
  }

  if ((!left.empty()) && left.back() == PATH_DELIM[0]) {
    left = str::trimRight(left, PATH_DELIM);
  }

  std::string right = path.path_;
  if ((!right.empty()) && right.front() == PATH_DELIM[0]) {
    right = str::trimLeft(right, PATH_DELIM);
  }

  return left + PATH_DELIM + right;
}

Path Path::operator/(const std::string &path) const {
  return *this / Path(path);
}

std::string Path::string() const {
  return path_;
}

Status Path::AsWString(std::wstring *ws) const {
  *ws = str::toWide(path_);
  return OkStatus();
}

}  // namespace util
}  // namespace llama
