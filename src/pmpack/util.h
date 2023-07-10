#pragma once

#include <stdint.h>
#include <memory>
#include "pmpack/block.h"
#include "llyn/c_ptr.h"

namespace pmpack {

// copy vector x to y.
void scopy(int n, const float *x, int incx, float *y, int incy);

// allocate n single float and returns the holder. the memory is 32 byte aligned.
ly::c_ptr<float> salloc(int64_t n);

}  // namespace pmpack
