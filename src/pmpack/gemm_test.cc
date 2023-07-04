#include "common/test_helper.h"

#include "pmpack/gemm_kernel.h"
#include "nn/nn_test_helper.h"
#include "nn/operators.h"
#include "util/util.h"

using namespace llama;
using namespace nn;

void refGemmFp32QInt8Fp32(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const void *B,
    const float *scaleDataB,
    int groupSizeB,
    float *C,
    int ldc) {
  CHECK(groupSizeB == K);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      const float *Aj = A + j * lda;
      const uint8_t *Bi = reinterpret_cast<const uint8_t *>(B) + K / 2;
      C[j * ldc + i] = DOTFp32Int4Fp32FallbackKernel::apply(K, Aj, Bi, scaleDataB[i]);
    }
  }
}

TEST_CASE("test dotFp32Int4Fp32", "[core][gemm][avx2]") {
  constexpr int DIM = 1024;

  std::vector<float> x(DIM);
  std::vector<uint8_t> y(DIM / 2);

  util::Random random(0x55aa);

  random.random(util::makeSpan(x));
  random.randomUInt8(util::makeSpan(y));

  float rs = DOTFp32Int4Fp32FallbackKernel::apply(DIM, x.data(), y.data(), 0.1f);
  float s = DOTFp32Int4Fp32Avx2Kernel::apply(DIM, x.data(), y.data(), 0.1f);

  REQUIRE(fabs(rs - s) < 1e-5);
}
