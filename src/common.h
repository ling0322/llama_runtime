#ifndef LLAMA_CC_COMMON_H_
#define LLAMA_CC_COMMON_H_

#include <stdint.h>
#include <stdlib.h>

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
#define LL_PLATFORM_APPLE
#elif defined(linux) || defined(__linux) || defined(__linux__)
#define LL_PLATFORM_LINUX
#elif defined(WIN32) || defined(__WIN32__) || defined(_MSC_VER) || \
      defined(_WIN32) || defined(__MINGW32__)
#define LL_PLATFORM_WINDOWS
#else
#error unknown platform
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


};  // namespace llama

#endif  // LLAMA_CC_COMMON_H_
