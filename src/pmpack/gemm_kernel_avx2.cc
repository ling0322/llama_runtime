#include "pmpack/gemm_kernel.h"

#include <immintrin.h>
#include <stdint.h>

namespace pmpack {

void sgemmKernel6x16Avx2(int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR

  // C: MR x NR (6 x 2 ymmX)
  __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
  __m256 a00, b00, b01;

  float *pc = c;
  c00 = _mm256_loadu_ps(pc);
  c01 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c10 = _mm256_loadu_ps(pc);
  c11 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c20 = _mm256_loadu_ps(pc);
  c21 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c30 = _mm256_loadu_ps(pc);
  c31 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c40 = _mm256_loadu_ps(pc);
  c41 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c50 = _mm256_loadu_ps(pc);
  c51 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  float *pa = a;
  float *pb = b;
  for (int k = 0; k < kc; ++k) {
    b00 = _mm256_loadu_ps(pb);
    b01 = _mm256_loadu_ps(pb + 8);
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
  _mm256_storeu_ps(pc, c00);
  _mm256_storeu_ps(pc + 8, c01);
  pc += rs_c;

  _mm256_storeu_ps(pc, c10);
  _mm256_storeu_ps(pc + 8, c11);
  pc += rs_c;

  _mm256_storeu_ps(pc, c20);
  _mm256_storeu_ps(pc + 8, c21);
  pc += rs_c;

  _mm256_storeu_ps(pc, c30);
  _mm256_storeu_ps(pc + 8, c31);
  pc += rs_c;

  _mm256_storeu_ps(pc, c40);
  _mm256_storeu_ps(pc + 8, c41);
  pc += rs_c;

  _mm256_storeu_ps(pc, c50);
  _mm256_storeu_ps(pc + 8, c51);
  pc += rs_c;
}

void saxpyKernelAvx2(int64_t n, float a, const float *x, float *y) {
  __m256 a00 = _mm256_broadcast_ss(&a);
  __m256 x00, y00;

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const float *px = x;
  float *py = y;
  for (int i = 0; i < nb; ++i) {
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_loadu_ps(py);

    y00 = _mm256_fmadd_ps(a00, x00, y00);
    _mm256_store_ps(py, y00);

    px += 8;
    py += 8;
  }

  for (int i = 0; i < nr; ++i) {
    *py++ += a * *px++;
  }
}

float sdotKernelAvx2(int64_t n, const float *x, const float *y) {
  __m256 x00, y00, a00;

  a00 = _mm256_setzero_ps();

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const float *px = x;
  const float *py = y;
  for (int i = 0; i < nb; ++i) {
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_loadu_ps(py);
    a00 = _mm256_fmadd_ps(x00, y00, a00);

    px += 8;
    py += 8;
  }

  // unroll a00
  __m128 r4 = _mm_add_ps(_mm256_extractf128_ps(a00, 1), _mm256_castps256_ps128(a00));
  __m128 r4h = _mm_movehl_ps(r4, r4);
  __m128 r2 = _mm_add_ps(r4, r4h);
  __m128 r2h = _mm_movehdup_ps(r2);
  __m128 r1 = _mm_add_ps(r2, r2h);
  float sum = _mm_cvtss_f32(r1);

  for (int i = 0; i < nr; ++i) {
    sum += *px++ * *py++;
  }

  return sum;
}

float DOTFp32Int4Fp32Avx2Kernel::apply(int64_t n, const float *x, const uint8_t *y, float scale) {
  __m256 x00, y00, a00, ymmScale;
  __m256i yint8x32, yint8x32odd, yint8x32even, ymm0xf, ymm0x8;
  __m128i yint8x16;

  a00 = _mm256_setzero_ps();
  ymm0xf = _mm256_set1_epi8(0xf);
  ymm0x8 = _mm256_set1_epi8(0x8);
  ymmScale = _mm256_broadcast_ss(&scale);

  int64_t nb = n / 32;

  const float *px = x;
  const uint8_t *py = y;
  for (int i = 0; i < nb; ++i) {
    // read 32 int4 (16 bytes), convert to 32 int8 and store to yint8x32 
    yint8x32 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(py)));
    yint8x32odd = _mm256_slli_epi16(yint8x32, 8);
    yint8x32even = _mm256_srli_epi16(yint8x32, 4);
    yint8x32 = _mm256_or_si256(yint8x32odd, yint8x32even);
    yint8x32 = _mm256_and_si256(yint8x32, ymm0xf);

    // uint4 range [0, 15] to int4 [-8, 7]
    yint8x32 = _mm256_sub_epi8(yint8x32, ymm0x8);

    // subblock 0
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm256_extracti128_si256(yint8x32, 0)));
    y00 = _mm256_mul_ps(y00, ymmScale);
    a00 = _mm256_fmadd_ps(x00, y00, a00);
    px += 8;

    // subblock 1
    yint8x16 = _mm256_extracti128_si256(yint8x32, 0);
    yint8x16 = _mm_srli_si128(yint8x16, 8);
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(yint8x16));
    y00 = _mm256_mul_ps(y00, ymmScale);
    a00 = _mm256_fmadd_ps(x00, y00, a00);
    px += 8;

    // subblock 2
    yint8x16 = _mm256_extracti128_si256(yint8x32, 1);
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(yint8x16));
    y00 = _mm256_mul_ps(y00, ymmScale);
    a00 = _mm256_fmadd_ps(x00, y00, a00);
    px += 8;

    // subblock 3
    yint8x16 = _mm_srli_si128(yint8x16, 8);
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(yint8x16));
    y00 = _mm256_mul_ps(y00, ymmScale);
    a00 = _mm256_fmadd_ps(x00, y00, a00);
    px += 8;

    py += 16;
  }

  // unroll a00
  __m128 r4 = _mm_add_ps(_mm256_extractf128_ps(a00, 1), _mm256_castps256_ps128(a00));
  __m128 r4h = _mm_movehl_ps(r4, r4);
  __m128 r2 = _mm_add_ps(r4, r4h);
  __m128 r2h = _mm_movehdup_ps(r2);
  __m128 r1 = _mm_add_ps(r2, r2h);
  float sum = _mm_cvtss_f32(r1);

  return sum;
}

}  // namespace pmpack
