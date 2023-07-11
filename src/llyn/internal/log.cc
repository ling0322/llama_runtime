#include "llyn/internal/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <string>

namespace ly {
namespace internal {

LogWrapper::LogWrapper(LogSeverity severity,
                       const char *source_file,
                       int source_line) 
    : severity_(severity),
      source_line_(source_line) {
  const char *s = strrchr(source_file, '/');
  if (!s) {
    s = strrchr(source_file, '\\');
  }

  if (s) {
    source_file_ = s + 1;
  } else {
    source_file_ = s;
  }
}

LogWrapper::~LogWrapper() {
  std::string message = os_.str();
  if (message.empty()) message = default_message_;

  printf("%s %s %s:%d] %s\n", 
         Severity(),
         Time(),
         source_file_,
         source_line_,
         message.c_str());

  if (severity_ == LogSeverity::Fatal) {
    abort();
  }
}

const char *LogWrapper::Time() {
  time_t now = time(nullptr);
  
  std::strftime(time_, sizeof(time_), "%FT%TZ", std::gmtime(&now));
  return time_;
}

const char *LogWrapper::Severity() const {
  switch (severity_) {
    case LogSeverity::Debug:
      return "DEBUG";
    case LogSeverity::Info:
      return "INFO";
    case LogSeverity::Warning:
      return "WARNING";
    case LogSeverity::Error:
      return "ERROR";
    case LogSeverity::Fatal:
      return "FATAL";
    default:
      fputs("invalid log severity.", stderr);
      abort();
  }
}

LogWrapper &LogWrapper::DefaultMessage(const char *message) {
  default_message_ = message;
  return *this;
}

}  // namespace internal
}  // namespace ly
