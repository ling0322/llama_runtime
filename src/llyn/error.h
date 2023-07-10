#pragma once

#include <exception>
#include <string>

namespace ly {

enum class ErrorCode : int {
  OK = 0,
  Aborted = 1,
  OutOfRange = 2,
};

class Error : public std::exception {
 public:
  Error(ErrorCode code, const std::string &what);
  ~Error();
 
  // get error code.
  ErrorCode getCode() const;

  // implement std::exception.
  const char* what() const noexcept override;

 private:
  ErrorCode _code;
  std::string _what;
};

class AbortedError : public Error {
 public:
  AbortedError(const std::string &what);
};

class OutOfRangeError : public Error {
 public:
  OutOfRangeError(const std::string &what);
};

}  // namespace ly
