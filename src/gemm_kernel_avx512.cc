#include "gemm_kernel.h"

#include <immintrin.h>

namespace llama {
namespace nn {

void SGEMM6x16Avx512Kernel::callKernel(
    int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR

  // MR=12, NR=32
  // C: MR x NR (6 zmmX registers)
  __m512 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61,
         c70, c71, c80, c81, c90, c91, ca0, ca1, cb0, cb1;
  __m512 a00, b00, b01;

  float *pc = c;
  c00 = _mm512_load_ps(pc);
  c01 = _mm512_load_ps(pc + 16);
  pc += rs_c;

  c10 = _mm512_load_ps(pc);
  c11 = _mm512_load_ps(pc + 16);
  pc += rs_c;

  c20 = _mm512_load_ps(pc);
  c21 = _mm512_load_ps(pc + 16);
  pc += rs_c;

  c30 = _mm512_load_ps(pc);
  c31 = _mm512_load_ps(pc + 16);
  pc += rs_c;

  c40 = _mm512_load_ps(pc);
  c41 = _mm512_load_ps(pc + 16);
  pc += rs_c;

  c50 = _mm512_load_ps(pc);
  c51 = _mm512_load_ps(pc + 16);
  pc += rs_c;

  c60 = _mm512_load_ps(pc);
  c61 = _mm512_load_ps(pc + 16);
  pc += rs_c;

  c70 = _mm512_load_ps(pc);
  c71 = _mm512_load_ps(pc + 16);
  pc += rs_c;

  c80 = _mm512_load_ps(pc);
  c81 = _mm512_load_ps(pc + 16);
  pc += rs_c;

  c90 = _mm512_load_ps(pc);
  c91 = _mm512_load_ps(pc + 16);
  pc += rs_c;

  ca0 = _mm512_load_ps(pc);
  ca1 = _mm512_load_ps(pc + 16);
  pc += rs_c;

  cb0 = _mm512_load_ps(pc);
  cb1 = _mm512_load_ps(pc + 16);
  pc += rs_c;

  float *pa = a;
  float *pb = b;
  for (int k = 0; k < kc; ++k) {
    b00 = _mm512_load_ps(pb);
    b01 = _mm512_load_ps(pb + 16);

    a00 = _mm512_set1_ps(pa[0]);
    c00 = _mm512_fmadd_ps(a00, b00, c00);
    c01 = _mm512_fmadd_ps(a00, b01, c01);

    a00 = _mm512_set1_ps(pa[1]);
    c10 = _mm512_fmadd_ps(a00, b00, c10);
    c11 = _mm512_fmadd_ps(a00, b01, c11);

    a00 = _mm512_set1_ps(pa[2]);
    c20 = _mm512_fmadd_ps(a00, b00, c20);
    c21 = _mm512_fmadd_ps(a00, b01, c21);

    a00 = _mm512_set1_ps(pa[3]);
    c30 = _mm512_fmadd_ps(a00, b00, c30);
    c31 = _mm512_fmadd_ps(a00, b01, c31);

    a00 = _mm512_set1_ps(pa[4]);
    c40 = _mm512_fmadd_ps(a00, b00, c40);
    c41 = _mm512_fmadd_ps(a00, b01, c41);

    a00 = _mm512_set1_ps(pa[5]);
    c50 = _mm512_fmadd_ps(a00, b00, c50);
    c51 = _mm512_fmadd_ps(a00, b01, c51);

    a00 = _mm512_set1_ps(pa[6]);
    c60 = _mm512_fmadd_ps(a00, b00, c60);
    c61 = _mm512_fmadd_ps(a00, b01, c61);

    a00 = _mm512_set1_ps(pa[7]);
    c70 = _mm512_fmadd_ps(a00, b00, c70);
    c71 = _mm512_fmadd_ps(a00, b01, c71);

    a00 = _mm512_set1_ps(pa[8]);
    c80 = _mm512_fmadd_ps(a00, b00, c80);
    c81 = _mm512_fmadd_ps(a00, b01, c81);

    a00 = _mm512_set1_ps(pa[9]);
    c90 = _mm512_fmadd_ps(a00, b00, c90);
    c91 = _mm512_fmadd_ps(a00, b01, c91);

    a00 = _mm512_set1_ps(pa[10]);
    ca0 = _mm512_fmadd_ps(a00, b00, ca0);
    ca1 = _mm512_fmadd_ps(a00, b01, ca1);

    a00 = _mm512_set1_ps(pa[11]);
    cb0 = _mm512_fmadd_ps(a00, b00, cb0);
    cb1 = _mm512_fmadd_ps(a00, b01, cb1);

    pb += 32;
    pa += 12;
  }

  pc = c;
  _mm512_store_ps(pc, c00);
  _mm512_store_ps(pc + 16, c01);
  pc += rs_c;

  _mm512_store_ps(pc, c10);
  _mm512_store_ps(pc + 16, c11);
  pc += rs_c;

  _mm512_store_ps(pc, c20);
  _mm512_store_ps(pc + 16, c21);
  pc += rs_c;

  _mm512_store_ps(pc, c30);
  _mm512_store_ps(pc + 16, c31);
  pc += rs_c;

  _mm512_store_ps(pc, c40);
  _mm512_store_ps(pc + 16, c41);
  pc += rs_c;

  _mm512_store_ps(pc, c50);
  _mm512_store_ps(pc + 16, c51);
  pc += rs_c;

  _mm512_store_ps(pc, c60);
  _mm512_store_ps(pc + 16, c61);
  pc += rs_c;

  _mm512_store_ps(pc, c70);
  _mm512_store_ps(pc + 16, c71);
  pc += rs_c;

  _mm512_store_ps(pc, c80);
  _mm512_store_ps(pc + 16, c81);
  pc += rs_c;

  _mm512_store_ps(pc, c90);
  _mm512_store_ps(pc + 16, c91);
  pc += rs_c;

  _mm512_store_ps(pc, ca0);
  _mm512_store_ps(pc + 16, ca1);
  pc += rs_c;

  _mm512_store_ps(pc, cb0);
  _mm512_store_ps(pc + 16, cb1);
  pc += rs_c;
}

}  // namespace nn
}  // namespace llama
