#ifndef LLAMA_CC_LOG_H_
#define LLAMA_CC_LOG_H_

#include <sstream>
#include "common.h"

#define LL_LOG_INTERNAL(severity) \
    llama::LogWrapper(llama::LogSeverity::severity, \
                                FUNCTION_NAME)
#define LL_LOG_INFO() LL_LOG_INTERNAL(kInfo)
#define LL_LOG_FATAL() LL_LOG_INTERNAL(kFatal)

#define LL_CHECK(cond) \
    if (cond) {} else LL_LOG_FATAL().DefaultMessage("Check " #cond " failed")
#define LL_CHECK_OK(expr) \
    do { auto s = (expr); if (!s.ok()) { LL_LOG_FATAL() << s.what(); } } \
    while (0)

namespace llama {

enum class LogSeverity {
  kDebug = 0,
  kInfo = 1,
  kWarning = 2,
  kError = 4,
  kFatal = 3
};

class LogWrapper {
 public:
  LogWrapper(LogSeverity severity, PCStrType location);
  ~LogWrapper();

  LogWrapper(LogWrapper &) = delete;
  LogWrapper &operator=(LogWrapper &) = delete;

  template <typename T>
  LogWrapper& operator<<(const T &value) { os_ << value; return *this; }

  // set the default message to LogWrapper. If no message appended, it will
  // log the `message` instead
  LogWrapper &DefaultMessage(const std::string &message);

 private:
  std::ostringstream os_;
  std::string default_message_;

  LogSeverity severity_;
  PCStrType location_;
  char time_[200];

  PCStrType Time();
  PCStrType Severity() const;
};

}  // namespace llama

#endif  // LLAMA_CC_LOG_H_
