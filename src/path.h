#ifndef LLAMA_CC_PATH_H_
#define LLAMA_CC_PATH_H_

#include <sstream>
#include <string>
#include "common.h"
#include "status.h"

namespace llama {

class Path {
 public:
  static Path CurrentModulePath();
  static Path CurrentExecutablePath();

  Path(const std::string &path);
  Path(std::string &&path);

  bool operator==(const Path &r) const;
  bool operator==(const std::string &r) const;

  Path dirname() const;
  Path basename() const;

  Path operator/(const Path &path) const;
  Path operator/(const std::string &path) const;

  std::string string() const;
  Status wstring(std::wstring *ws) const;

 private:
  std::string path_;
};

}  // namespace llama

#endif  // LLAMA_CC_PATH_H_
