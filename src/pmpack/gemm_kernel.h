// A default cpu-based GEMM
#pragma once

#include <stdint.h>
#include <memory>

namespace pmpack {

// -- avx512 kernels ---------

void sgemmKernel12x32Avx512(int64_t kc, float *a, float *b, float *c, int64_t rs_c);

// -- avx2 kernels ---------

void sgemmKernel6x16Avx2(int64_t kc, float *a, float *b, float *c, int64_t rs_c);
void saxpyKernelAvx2(int64_t n, float a, const float *x, float *y);
float sdotKernelAvx2(int64_t n, const float *x, const float *y);
float dotFp32Int4Fp32KernelAvx2(int64_t n, const float *x, const uint8_t *y, float scale);

// -- fallback kernels ---------

void sgemmKernel6x16Fallback(int64_t kc, float *a, float *b, float *c, int64_t rs_c);
void dequantizeInt4ToFloat32Fallback(const int8_t *src, float scale, int n, float *tgt);
float dotFp32Int4Fp32KernelFallback(int64_t n, const float *x, const uint8_t *y, float scale);

// -- classes ----------

class SGEMM6x16DefaultKernel {
 public:
  static constexpr int MR = 6;
  static constexpr int NR = 16;
  static inline void callKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
    sgemmKernel6x16Fallback(kc, a, b, c, rs_c);
  }
};

class SGEMM6x16Avx2Kernel {
 public:
  static constexpr int MR = 6;
  static constexpr int NR = 16;
  static inline void callKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
    sgemmKernel6x16Avx2(kc, a, b, c, rs_c);
  }
};

class SGEMM12x32Avx512Kernel {
 public:
  static constexpr int MR = 12;
  static constexpr int NR = 32;
  static inline void callKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
    sgemmKernel12x32Avx512(kc, a, b, c, rs_c);
  }
};

class SAXPYAvx2Kernel {
 public:
  static inline void callKernel(int64_t n, float a, const float *x, float *y) {
    saxpyKernelAvx2(n, a, x, y);
  }
};

class SDOTAvx2Kernel {
 public:
  static inline float callKernel(int64_t n, const float *x, const float *y) {
    return sdotKernelAvx2(n, x, y);
  }
};

class DOTFp32Int4Fp32Avx2Kernel {
 public:
  static float apply(int64_t n, const float *x, const uint8_t *y, float scale);
};

class DOTFp32Int4Fp32FallbackKernel {
 public:
  static float apply(int64_t n, const float *x, const uint8_t *y, float scale);
};

}  // namespace pmpack
