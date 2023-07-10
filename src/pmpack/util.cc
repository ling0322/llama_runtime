#include "pmpack/util.h"

#include "llyn/platform.h"

namespace pmpack {

// copy vector x to y.
void scopy(int n, const float *x, int incx, float *y, int incy) {
  for (int i = 0; i < n; ++i) {
    y[i * incy] = x[i * incx];
  }
}

// allocate n single float and returns the holder. the memory is 32 byte aligned.
ly::c_ptr<float> salloc(int64_t n) {
  return ly::c_ptr<float>(
      reinterpret_cast<float *>(ly::alloc32ByteAlignedMem(sizeof(float) * n)),
      ly::free32ByteAlignedMem);
}

}  // namespace pmpack

