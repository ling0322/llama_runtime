#ifndef LLAMA_CC_LOG_H_
#define LLAMA_CC_LOG_H_

#include <sstream>
#include "common.h"

#define LOG(severity) llama::LogWrapper ## severity(FUNCTION_NAME)
#define CHECK(cond) \
    if (cond) {} else LOG(FATAL).DefaultMessage("Check " #cond " failed")
#define CHECK_OK(expr) \
    do { auto s = (expr); if (!s.ok()) { LOG(FATAL) << s.what(); } } \
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

// log wrappers for each severity
class LogWrapperINFO : public LogWrapper {
 public:
  LogWrapperINFO(PCStrType location) : 
      LogWrapper(LogSeverity::kInfo, location) {}
};
class LogWrapperFATAL : public LogWrapper {
 public:
  LogWrapperFATAL(PCStrType location) : 
      LogWrapper(LogSeverity::kFatal, location) {}
};

}  // namespace llama

#endif  // LLAMA_CC_LOG_H_
