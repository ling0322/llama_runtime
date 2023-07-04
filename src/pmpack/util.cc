#include "pmpack/util.h"

namespace llama {
namespace nn {

// copy vector x to y.
void scopy(int n, const float *x, int incx, float *y, int incy) {
  for (int i = 0; i < n; ++i) {
    y[i * incy] = x[i * incx];
  }
}

// allocate n single float and returns the holder. the memory is 32 byte aligned.
util::AutoCPtr<float> salloc(int64_t n) {
  return util::AutoCPtr<float>(
      reinterpret_cast<float *>(util::alloc32ByteAlignedMem(sizeof(float) * n)),
      util::free32ByteAlignedMem);
}

}  // namespace nn
}  // namespace llama

