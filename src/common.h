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

#define STRINGIZE(x) STRINGIZE_INTENRAL(x)
#define STRINGIZE_INTENRAL(x) #x

// to make some basic classes independent of any other code, we use ASSERT
// instead of LL_CHECK in these files
#define ASSERT(x) do { if (!(x)) { abort(); }} while (0)

#define NAMEOF(x) #x

#ifdef __APPLE__
#define BR_PLATFORM_APPLE
#elif defined(linux) || defined(__linux) || defined(__linux__)
#define BR_PLATFORM_LINUX
#elif defined(WIN32) || defined(__WIN32__) || defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32__)
#define BR_PLATFORM_WINDOWS
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

enum class DType : int16_t { kUnknown = 0, kFloat = 1 };

// get type-id
template <typename T>
DType TypeID();

// get the size of specific dtype
int SizeOfDType(DType dtype);

// return true of DType is valid
bool IsValidDType(DType dtype);

};  // namespace llama

#endif  // LLAMA_CC_COMMON_H_
