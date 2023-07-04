#include "pmpack/gemm_kernel.h"

#include <stdlib.h>
#include "common/environment.h"
#include "util/log.h"
#include "util/util.h"

namespace llama {
namespace nn {

// -- fallback micro-kernels ---------

void sgemmKernel6x16Fallback(int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR
  constexpr int64_t MR = 6;
  constexpr int64_t NR = 16;

  for (int k = 0; k < kc; ++k) {
    float *Ak = a + k * MR;
    for (int m = 0; m < MR; ++m) {
      float *Cm = c + m * rs_c;
      float Akm = Ak[m];
      float *Bk = b + k * NR;
      
      for (int n = 0; n < NR; ++n) {
        Cm[n] += Akm * Bk[n];
      }
    }
  }
}

// dequantize n numbers from src to tgt with the specified scale.
void dequantizeInt4ToFloat32Fallback(const ByteType *src, float scale, int n, float *tgt) {
  CHECK(n % 2 == 0);
  int nb = n / 2;

  const int8_t *p = reinterpret_cast<const int8_t *>(src);
  for (int i = 0; i < nb; ++i) {
    *tgt++ = scale * (*p >> 4);
    *tgt++ = scale * ((*p << 4) >> 4);
    ++p;
  }
}

float DOTFp32Int4Fp32FallbackKernel::apply(
    int64_t n, const float *x, const uint8_t *y, float scale) {
  int64_t nb = n / 2;
  float sum = 0.0f;

  const uint8_t *py = y;
  for (int64_t i = 0; i < nb; ++i) {
    sum += *x++ * scale * (static_cast<int>(*py >> 4) - 8);
    sum += *x++ * scale * ((static_cast<int>(*py) & 0xf) - 8);
    ++py;
  }

  return sum;
}

}  // namespace nn
}  // namespace llama
