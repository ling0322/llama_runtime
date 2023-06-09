#ifndef FASTALPACA_GEMM_KERNEL_H_
#define FASTALPACA_GEMM_KERNEL_H_

#include <stdint.h>
#include "log.h"

namespace llama {
namespace nn {

class SGEMM6x16DefaultKernel {
 public:
  static constexpr int MR = 6;
  static constexpr int NR = 16;
  static void callKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c);
};

class SGEMM6x16Avx2Kernel {
 public:
  static constexpr int MR = 6;
  static constexpr int NR = 16;
  static void callKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c);
};

class SGEMM6x16Avx512Kernel {
 public:
  static constexpr int MR = 12;
  static constexpr int NR = 32;
  static void callKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c);
};

}  // namespace nn
}  // namespace llama

#endif  // FASTALPACA_GEMM_KERNEL_H_
