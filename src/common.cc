#include "common.h"

namespace llama {

Exception::Exception(StatusCode code, const std::string &what) : _code(code), _what(what) {}
Exception::~Exception() {}

StatusCode Exception::getCode() const {
  return _code;
}

const char *Exception::what() const noexcept {
  return _what.c_str();
}

AbortedException::AbortedException(const std::string &what)
    : Exception(StatusCode::kAborted, what) {}

}  // namespace llama
