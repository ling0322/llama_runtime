#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <exception>
#include <sstream>

#define PROJECT_NAME "llama_runtime"

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName &);              \
  TypeName &operator=(const TypeName &);

#define CONCAT2(a, b) a##b
#define NOT_IMPL()          \
  abort()

// to make some basic classes independent of any other code, we use ASSERT
// instead of CHECK in these files
#define ASSERT(x) do { if (!(x)) { abort(); }} while (0)

#define NAMEOF(x) #x

#ifdef __APPLE__
#define AL_PLATFORM_APPLE
#elif defined(linux) || defined(__linux) || defined(__linux__)
#define AL_PLATFORM_LINUX
#elif defined(WIN32) || defined(__WIN32__) || defined(_MSC_VER) || \
      defined(_WIN32) || defined(__MINGW32__)
#define AL_PLATFORM_WINDOWS
#else
#error unknown platform
#endif

#if defined(__cplusplus) && defined(__has_cpp_attribute)
#define LR_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define LR_HAS_CPP_ATTRIBUTE(x) 0
#endif

#if LR_HAS_CPP_ATTRIBUTE(clang::lifetimebound)
#define LR_LIFETIME_BOUND [[clang::lifetimebound]]
#else
#define LR_LIFETIME_BOUND
#endif

#ifdef __GNUC__
#define FUNCTION_NAME __PRETTY_FUNCTION__
#else
#define FUNCTION_NAME __FUNCTION__
#endif

#define DEBUG_EDGE_ITEMS 3

namespace llama {

typedef const char *PCStrType;
typedef unsigned char ByteType;
typedef const unsigned char CByteType;

enum class StatusCode : int {
  kOK = 0,
  kAborted = 1,
  kOutOfRange = 2,
};

class Exception : public std::exception {
 public:
  Exception(StatusCode code, const std::string &what);
  ~Exception();
 
  // get error code.
  StatusCode getCode() const;

  // implement std::exception.
  const char* what() const noexcept override;

 private:
  StatusCode _code;
  std::string _what;
};

class AbortedException : public Exception {
 public:
  AbortedException(const std::string &what);
};

};  // namespace llama

