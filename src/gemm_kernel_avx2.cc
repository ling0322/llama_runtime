#include "gemm_common.h"

#include <immintrin.h>

namespace llama {
namespace nn {

void SGEMM6x16Avx2Kernel::GEMMKernel(
    int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR

  // C: MR x NR (6 x 2 ymmX)
  __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
  __m256 a00, b00, b01;

  float *pc = c;
  c00 = _mm256_load_ps(pc);
  c01 = _mm256_load_ps(pc + 8);
  pc += rs_c;

  c10 = _mm256_load_ps(pc);
  c11 = _mm256_load_ps(pc + 8);
  pc += rs_c;

  c20 = _mm256_load_ps(pc);
  c21 = _mm256_load_ps(pc + 8);
  pc += rs_c;

  c30 = _mm256_load_ps(pc);
  c31 = _mm256_load_ps(pc + 8);
  pc += rs_c;

  c40 = _mm256_load_ps(pc);
  c41 = _mm256_load_ps(pc + 8);
  pc += rs_c;

  c50 = _mm256_load_ps(pc);
  c51 = _mm256_load_ps(pc + 8);
  pc += rs_c;

  float *pa = a;
  float *pb = b;
  for (int k = 0; k < kc; ++k) {
    b00 = _mm256_load_ps(pb);
    b01 = _mm256_load_ps(pb + 8);
    a00 = _mm256_broadcast_ss(pa);

    c00 = _mm256_fmadd_ps(a00, b00, c00);
    c01 = _mm256_fmadd_ps(a00, b01, c01);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c10 = _mm256_fmadd_ps(a00, b00, c10);
    c11 = _mm256_fmadd_ps(a00, b01, c11);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c20 = _mm256_fmadd_ps(a00, b00, c20);
    c21 = _mm256_fmadd_ps(a00, b01, c21);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c30 = _mm256_fmadd_ps(a00, b00, c30);
    c31 = _mm256_fmadd_ps(a00, b01, c31);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c40 = _mm256_fmadd_ps(a00, b00, c40);
    c41 = _mm256_fmadd_ps(a00, b01, c41);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c50 = _mm256_fmadd_ps(a00, b00, c50);
    c51 = _mm256_fmadd_ps(a00, b01, c51);
    pa += 1;

    pb += 16;
  }

  pc = c;
  _mm256_store_ps(pc, c00);
  _mm256_store_ps(pc + 8, c01);
  pc += rs_c;

  _mm256_store_ps(pc, c10);
  _mm256_store_ps(pc + 8, c11);
  pc += rs_c;

  _mm256_store_ps(pc, c20);
  _mm256_store_ps(pc + 8, c21);
  pc += rs_c;

  _mm256_store_ps(pc, c30);
  _mm256_store_ps(pc + 8, c31);
  pc += rs_c;

  _mm256_store_ps(pc, c40);
  _mm256_store_ps(pc + 8, c41);
  pc += rs_c;

  _mm256_store_ps(pc, c50);
  _mm256_store_ps(pc + 8, c51);
  pc += rs_c;
}

}  // namespace nn
}  // namespace llama
