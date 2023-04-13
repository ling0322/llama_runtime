#include "path.h"

#include "strings.h"

#ifdef _WIN32
#define PATH_DELIM "\\"
#else
#define PATH_DELIM "/"
#endif

namespace llama {

Path::Path(const std::string &path) : path_(path) {}
Path::Path(std::string &&path): path_(std::move(path)) {}

Path Path::dirname() const {
  int last_delim_idx = path_.find_last_of(PATH_DELIM);
  if (last_delim_idx == std::string::npos) {
    return "";
  }

  std::string name = std::string(path_.begin(), path_.begin() + last_delim_idx);
  name = strings::TrimRight(name, PATH_DELIM);
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
    left = strings::TrimRight(left, PATH_DELIM);
  }

  std::string right = path.path_;
  if ((!right.empty()) && right.front() == PATH_DELIM[0]) {
    right = strings::TrimLeft(right, PATH_DELIM);
  }

  return left + PATH_DELIM + right;
}

Path Path::operator/(const std::string &path) const {
  return *this / Path(path);
}

std::string Path::string() const {
  return path_;
}

Status Path::wstring(std::wstring *ws) const {
  RETURN_IF_ERROR(strings::ToWide(path_, ws));
  return OkStatus();
}

}  // namespace llama
