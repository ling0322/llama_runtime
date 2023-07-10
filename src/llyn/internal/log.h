#pragma once 

#include "llyn/log.h"

namespace ly {
namespace internal {

class LogWrapper {
 public:
  LogWrapper(LogSeverity severity, const char *source_file, int source_line);
  ~LogWrapper();

  LogWrapper(LogWrapper &) = delete;
  LogWrapper &operator=(LogWrapper &) = delete;

  template <typename T>
  LogWrapper& operator<<(const T &value) { os_ << value; return *this; }

  // set the default message to LogWrapper. If no message appended, it will
  // log the `message` instead
  LogWrapper &DefaultMessage(const char *message);

 private:
  std::ostringstream os_;
  const char *default_message_;

  LogSeverity severity_;
  const char *source_file_;
  int source_line_;
  char time_[200];

  const char *Time();
  const char *Severity() const;
};

// log wrappers for each severity
class LogWrapperINFO : public LogWrapper {
 public:
  LogWrapperINFO(const char *source_file, int source_line) : 
      LogWrapper(LogSeverity::Info, source_file, source_line) {}
};
class LogWrapperWARN : public LogWrapper {
 public:
  LogWrapperWARN(const char *source_file, int source_line) : 
      LogWrapper(LogSeverity::Warning, source_file, source_line) {}
};
class LogWrapperFATAL : public LogWrapper {
 public:
  LogWrapperFATAL(const char *source_file, int source_line) : 
      LogWrapper(LogSeverity::Fatal, source_file, source_line) {}
};

}  // namespace internal
}  // namespace ly
