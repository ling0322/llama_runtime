#pragma once

#include <string>

namespace ly {

// provide base functions for path. For example, path::Join, dirname, basename
// and convert to wstring, ...
class Path {
 public:
  static Path currentModulePath();
  static Path currentExecutablePath();

  Path() = default;
  Path(const std::string &path);
  Path(std::string &&path);

  bool operator==(const Path &r) const;
  bool operator==(const std::string &r) const;

  Path dirname() const;
  Path basename() const;
  bool isabs() const;

  Path operator/(const Path &path) const;
  Path operator/(const std::string &path) const;

  std::string string() const;
  std::wstring wstring() const;

 private:
  std::string _path;

  // normalize path string.
  static std::string normPath(const std::string &path);
};

}  // namespace ly