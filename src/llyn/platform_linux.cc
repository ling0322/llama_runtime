#include "llyn/platform.h"

#include <stdlib.h>

namespace ly {

void initCpuInfo() {
#if !defined(__clang__) || __clang_major__ >= 6
  __builtin_cpu_init();
#endif
}

bool isAvx512Available() {
  initCpuInfo();
  return __builtin_cpu_supports("avx512f") != 0;
}

bool isAvx2Available() {
  initCpuInfo();
  return __builtin_cpu_supports("avx2") != 0;
}


void *alloc32ByteAlignedMem(int64_t size) {
  if (size % 32 != 0) {
    size += (32 - size % 32);
  }
  return aligned_alloc(32, size);
}

void free32ByteAlignedMem(void *ptr) {
  free(ptr);
}

const char *getPathDelim() {
  return "/";
}

}
