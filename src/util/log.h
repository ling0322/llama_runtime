#ifndef LLAMA_CC_LOG_H_
#define LLAMA_CC_LOG_H_

#include <sstream>
#include "common/common.h"

#define LOG(severity) llama::LogWrapper ## severity(__FILE__, __LINE__)
#define CHECK(cond) \
    if (cond) {} else LOG(FATAL).DefaultMessage("Check " #cond " failed.")
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
  LogWrapper(LogSeverity severity, PCStrType source_file, int source_line);
  ~LogWrapper();

  LogWrapper(LogWrapper &) = delete;
  LogWrapper &operator=(LogWrapper &) = delete;

  template <typename T>
  LogWrapper& operator<<(const T &value) { os_ << value; return *this; }

  // set the default message to LogWrapper. If no message appended, it will
  // log the `message` instead
  LogWrapper &DefaultMessage(PCStrType message);

 private:
  std::ostringstream os_;
  PCStrType default_message_;

  LogSeverity severity_;
  PCStrType source_file_;
  int source_line_;
  char time_[200];

  PCStrType Time();
  PCStrType Severity() const;
};

// log wrappers for each severity
class LogWrapperINFO : public LogWrapper {
 public:
  LogWrapperINFO(PCStrType source_file, int source_line) : 
      LogWrapper(LogSeverity::kInfo, source_file, source_line) {}
};
class LogWrapperWARN : public LogWrapper {
 public:
  LogWrapperWARN(PCStrType source_file, int source_line) : 
      LogWrapper(LogSeverity::kWarning, source_file, source_line) {}
};
class LogWrapperFATAL : public LogWrapper {
 public:
  LogWrapperFATAL(PCStrType source_file, int source_line) : 
      LogWrapper(LogSeverity::kFatal, source_file, source_line) {}
};

}  // namespace llama

#endif  // LLAMA_CC_LOG_H_
