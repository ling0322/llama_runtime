#include "llyn/platform.h"

#include <windows.h>

namespace ly {

bool isAvx512Available() {
  return IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE) == TRUE;
}

bool isAvx2Available() {
  return IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE) == TRUE;
}

void *alloc32ByteAlignedMem(int64_t size) {
  return _aligned_malloc(size, 32);
}

void free32ByteAlignedMem(void *ptr) {
  _aligned_free(ptr);
}

const char *getPathDelim() {
  return "\\";
}

}
