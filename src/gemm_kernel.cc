#include "gemm.h"

#include "gemm_kernel.h"

namespace llama {
namespace nn {

void SGEMM6x16DefaultKernel::callKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR
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

}  // namespace nn
}  // namespace llama
