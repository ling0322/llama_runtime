#include "llyn/error.h"

namespace ly {

Error::Error(ErrorCode code, const std::string &what) : _code(code), _what(what) {}
Error::~Error() {}

ErrorCode Error::getCode() const {
  return _code;
}

const char *Error::what() const noexcept {
  return _what.c_str();
}

AbortedError::AbortedError(const std::string &what)
    : Error(ErrorCode::Aborted, what) {}

OutOfRangeError::OutOfRangeError(const std::string &what)
    : Error(ErrorCode::OutOfRange, what) {}

}
