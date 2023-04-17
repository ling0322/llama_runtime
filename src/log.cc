#include "log.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

#include "path.h"

namespace llama {

LogWrapper::LogWrapper(LogSeverity severity,
                       PCStrType source_file,
                       int source_line) 
    : severity_(severity),
      source_line_(source_line) {
  PCStrType s = strrchr(__FILE__, '/');
  if (!s) {
    s = strrchr(__FILE__, '\\');
  }

  if (s) {
    source_file_ = s + 1;
  } else {
    source_file_ = s;
  }
}

LogWrapper::~LogWrapper() {
  std::string message = os_.str();
  printf("%s %s %s:%d] %s\n", 
         Severity(),
         Time(),
         source_file_,
         source_line_,
         message.c_str());

  if (severity_ == LogSeverity::kFatal) {
    abort();
  }
}

PCStrType LogWrapper::Time() {
  time_t now = time(nullptr);
  
  std::strftime(time_, sizeof(time_), "%FT%TZ", std::gmtime(&now));
  return time_;
}

PCStrType LogWrapper::Severity() const {
  switch (severity_) {
    case LogSeverity::kDebug:
      return "DEBUG";
    case LogSeverity::kInfo:
      return "INFO";
    case LogSeverity::kWarning:
      return "WARNING";
    case LogSeverity::kError:
      return "ERROR";
    case LogSeverity::kFatal:
      return "FATAL";
    default:
      NOT_IMPL();
  }
}

LogWrapper &LogWrapper::DefaultMessage(PCStrType message) {
  default_message_ = message;
  return *this;
}

}  // namespace llama
