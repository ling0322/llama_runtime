// OS ot platform dependent functions.
#pragma once

#include <stdint.h>

#ifdef __APPLE__
#define LY_PLATFORM_APPLE
#elif defined(linux) || defined(__linux) || defined(__linux__)
#define LY_PLATFORM_LINUX
#elif defined(WIN32) || defined(__WIN32__) || defined(_MSC_VER) || \
      defined(_WIN32) || defined(__MINGW32__)
#define LY_PLATFORM_WINDOWS
#else
#error unknown platform
#endif

namespace ly {

bool isAvx512Available();
bool isAvx2Available();
void *alloc32ByteAlignedMem(int64_t nbytes);
void free32ByteAlignedMem(void *);
const char *getPathDelim();

}
