#pragma once

#include <stdint.h>
#include <memory>
#include "pmpack/block.h"
#include "util/util.h"

namespace llama {
namespace nn {

// copy vector x to y.
void scopy(int n, const float *x, int incx, float *y, int incy);

// allocate n single float and returns the holder. the memory is 32 byte aligned.
util::AutoCPtr<float> salloc(int64_t n);

}  // namespace nn
}  // namespace llama
