#pragma once

#include <sstream>

#define LOG(severity) ly::internal::LogWrapper ## severity(__FILE__, __LINE__)
#define NOT_IMPL() LOG(FATAL) << "not implemented"

// CHECK macro conflicts with catch2
#ifndef CATCH_TEST_MACROS_HPP_INCLUDED
#define CHECK(cond) \
    if (cond) {} else LOG(FATAL).DefaultMessage("Check " #cond " failed.")
#endif

namespace ly {

enum class LogSeverity {
  Debug = 0,
  Info = 1,
  Warning = 2,
  Error = 4,
  Fatal = 3
};

}  // namespace ly

#include "llyn/internal/log.h"
