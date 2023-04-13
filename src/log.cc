#include "log.h"

#include <stdio.h>
#include <time.h>

namespace llama {

LogWrapper::LogWrapper(LogSeverity severity, PCStrType location) 
    : severity_(severity),
      location_(location) {}

LogWrapper::~LogWrapper() {
  std::string message = os_.str();
  printf("%s %s %s] %s\n", Severity(), Time(), location_, message.c_str());

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

LogWrapper &LogWrapper::DefaultMessage(const std::string &message) {
  default_message_ = message;
  return *this;
}

}  // namespace llama
